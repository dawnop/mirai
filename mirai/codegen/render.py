import ast
from jinja2 import (
    Environment,
    StrictUndefined,
    Template,
    PackageLoader,
)
import json
import os
import argparse


class TensorShape:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class AsyncTritonAssignVisitor(ast.NodeVisitor):
    def __init__(self):
        self.kernels = []

    def visit_Module(self, node):
        for child in node.body:
            if isinstance(child, ast.Assign):
                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    target = child.targets[0]
                    value = child.value
                    # async_compile.triton(...)
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id == "async_compile"
                        and value.func.attr == "triton"
                    ):
                        self.kernels.append(target.id)


class TensorIOVisitor(ast.NodeVisitor):
    """Parse the ``call()`` function to extract input/output variable names.

    In dynamic mode the ``call()`` args contain a mix of int (shape symbol)
    and tensor arguments.  After visiting, extra attributes are populated:

    * ``symbol_bindings`` – ``{symbol_name: primals_name}`` e.g. ``{"s0": "primals_1"}``
    * ``int_arg_names`` – ordered list of primals that are shape ints
    * ``tensor_input_names`` – ordered list of primals that are tensors
    """

    def __init__(self):
        self.input_names = []
        self.output_names = []
        # Dynamic-mode extras
        self.symbol_bindings = {}  # s0 -> primals_1
        self.int_arg_names = []  # primals that are shape ints
        self.tensor_input_names = []  # primals that are tensors

    def visit_Module(self, node):
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "call":
                self._visit_call(child)

    def _visit_call(self, func_node):
        # Collect all primals names from args unpacking
        all_arg_names = []
        # Collect names that appear in assert_size_stride (these are tensors)
        asserted_names = set()
        # Collect symbol bindings: s0 = primals_1
        symbol_bindings = {}

        for stmt in func_node.body:
            # Input unpacking: primals_1, primals_2, ... = args
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
                and isinstance(stmt.value, ast.Name)
                and stmt.value.id == "args"
            ):
                all_arg_names = [elt.id for elt in stmt.targets[0].elts if isinstance(elt, ast.Name)]

            # assert_size_stride(primals_X, ...) -> primals_X is a tensor
            elif (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "assert_size_stride"
                and isinstance(stmt.value.args[0], ast.Name)
            ):
                asserted_names.add(stmt.value.args[0].id)

            # s0 = primals_1 -> symbol binding (int arg)
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Name)
                and stmt.value.id in (all_arg_names or [])
                and stmt.targets[0].id not in (all_arg_names or [])
            ):
                sym = stmt.targets[0].id
                primal = stmt.value.id
                symbol_bindings[sym] = primal

            # Output
            elif isinstance(stmt, ast.Return):
                self.output_names.extend(self._extract_return_vars(stmt.value))

        self.input_names = all_arg_names
        self.symbol_bindings = symbol_bindings
        int_arg_set = set(symbol_bindings.values())
        self.int_arg_names = [n for n in all_arg_names if n in int_arg_set]
        self.tensor_input_names = [n for n in all_arg_names if n not in int_arg_set]

    def _extract_return_vars(self, value):
        result = []
        if isinstance(value, ast.Tuple):
            for elt in value.elts:
                result.extend(self._extract_output_var(elt))
        else:
            result.extend(self._extract_output_var(value))
        return result

    def _extract_output_var(self, elt):
        result = []
        if isinstance(elt, ast.Name):
            result.append(elt.id)
        elif (
            isinstance(elt, ast.Call)
            and isinstance(elt.func, ast.Name)
            and elt.func.id == "reinterpret_tensor"
            and len(elt.args) > 0
            and isinstance(elt.args[0], ast.Name)
        ):
            result.append(elt.args[0].id)
        return result


def parse_grid_from_meta(name, base_dir="./"):
    file_name = os.path.join(base_dir, f"{name}_meta.txt")
    with open(file_name, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    values = lines[1].split()

    data = dict(zip(header, values))

    xgrid = data["XGRID"]
    ygrid = data["YGRID"]
    zgrid = data["ZGRID"]

    return xgrid, ygrid, zgrid


def strided_shape_to_contiguous(shape, stride):
    paired = list(zip(shape, stride))
    paired_sorted = sorted(paired, key=lambda x: -x[1])
    shape = [x[0] for x in paired_sorted]
    return shape


def _expr_to_cpp(node):
    """Convert a Python AST expression to a C++ expression string.

    Handles symbolic shape expressions like ``s0*s1*s2``, ``s0*s3``, etc.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp):
        left = _expr_to_cpp(node.left)
        right = _expr_to_cpp(node.right)
        if isinstance(node.op, ast.Mult):
            return f"({left} * {right})"
        if isinstance(node.op, ast.Add):
            return f"({left} + {right})"
        if isinstance(node.op, ast.Sub):
            return f"({left} - {right})"
        if isinstance(node.op, ast.FloorDiv):
            return f"({left} / {right})"
        if isinstance(node.op, ast.Mod):
            return f"({left} % {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_expr_to_cpp(node.operand)})"
    return ast.unparse(node)


def _shape_tuple_to_cpp(shape_node):
    """Convert a Python AST tuple of shape expressions to a list of C++ strings.

    E.g. ``(s1, s0, s2)`` → ``["s1", "s0", "s2"]``
    """
    if isinstance(shape_node, ast.Tuple):
        return [_expr_to_cpp(elt) for elt in shape_node.elts]
    return [_expr_to_cpp(shape_node)]


def _stride_tuple_to_list(stride_node):
    """Convert a Python AST tuple of stride expressions to a list of C++ strings."""
    if isinstance(stride_node, ast.Tuple):
        return [_expr_to_cpp(elt) for elt in stride_node.elts]
    return [_expr_to_cpp(stride_node)]


def _strided_shape_to_contiguous_dynamic(shape_exprs, stride_exprs):
    """Sort shape dims by descending stride to get contiguous allocation order.

    Both inputs are lists of C++ expression strings.
    For dynamic mode, strides are symbolic, so we attempt to parse integer
    constants and fall back to the original order.
    """

    def _try_int(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    int_strides = [_try_int(s) for s in stride_exprs]
    if all(v is not None for v in int_strides):
        paired = list(zip(shape_exprs, int_strides))
        paired_sorted = sorted(paired, key=lambda x: -x[1])
        return [x[0] for x in paired_sorted]
    # Cannot sort — return original order (already contiguous in most cases)
    return list(shape_exprs)


class CallLogicVisitor(ast.NodeVisitor):
    """Generates C++ code fragments from the ``call()`` function body.

    Supports both static (concrete shapes) and dynamic (symbolic shapes) modes.

    Args:
        input_shapes_map: ``{var_name: shape_list}`` for static mode.
        output_shapes_map: ``{var_name: shape_list}`` for static mode.
        output_names: list of output variable names.
        output_path: directory containing *_meta.txt files.
        dynamic: If True, emit runtime shape logic.
        symbol_bindings: ``{symbol_name: primals_name}`` e.g. ``{"s0": "primals_1"}``.
        tensor_input_names: list of primals that are tensors (dynamic mode).
        int_arg_names: list of primals that are shape ints (dynamic mode).
    """

    def __init__(
        self,
        input_shapes_map,
        output_shapes_map,
        output_names,
        output_path="./",
        dynamic=False,
        symbol_bindings=None,
        tensor_input_names=None,
        int_arg_names=None,
        meta_dicts=None,
    ):
        self.input_shapes_map = input_shapes_map
        self.output_shapes_map = output_shapes_map
        self.output_names = output_names
        self.output_path = output_path
        self.dynamic = dynamic
        self.symbol_bindings = symbol_bindings or {}
        self.tensor_input_names = tensor_input_names or []
        self.int_arg_names = int_arg_names or []
        self.meta_dicts = meta_dicts or {}

        # In dynamic mode, track which symbols / intermediates have already been emitted.
        self._emitted_vars = set()
        # Track known shape-symbol names (s0, s1, ...) and intermediates (ps0, ...)
        self._known_symbols = set(self.symbol_bindings.keys())

        self.codes = []

        self.allocate_temp_tpl = Template(
            """
        Tensor {{var_name}};
        TensorShape {{var_name}}_shape = { {% for n in shape %}{{ n }}, {% endfor %} };
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(get_type<Tp>(), {{var_name}}_shape, &{{var_name}}, alloc_attrs));
        auto {{var_name}}_ptr = (CUdeviceptr)({{var_name}}.data());
        """
        )

        self.allocate_output_tpl = Template(
            """
        Tensor *{{var_name}};
        TensorShape {{var_name}}_shape = { {% for n in shape %}{{ n }}, {% endfor %} };
        OP_REQUIRES_OK(ctx, ctx->allocate_output({{output_index}}, {{var_name}}_shape, &{{var_name}}, alloc_attrs));
        auto {{var_name}}_ptr = (CUdeviceptr)({{var_name}}->data());
        """
        )

        self.check_shape_tpl = Template(
            """
        OP_REQUIRES(ctx, check_tensor_shape_flexible( {{var_name}}, { {% for n in shape %}{{ n }}, {% endfor %} } ) ,
            errors::InvalidArgument("Input {{var_name}} shape should be [{% for n in shape %}{{ n }}, {% endfor %}], got: ", {{var_name}}.shape()));
        """
        )

        self.assign_tensor_tpl = Template(
            """
        auto {{target_name}}_ptr = {{value_name}}_ptr;
        """
        )

        self.kernel_call_tpl = Template(
            """
        auto {{kernel_name}}_{{count}}_config = launch_config_map_["{{kernel_name}}"];
        auto {{kernel_name}}_{{count}}_cufunc = cufunc_map_["{{kernel_name}}"];
        uint32_t {{kernel_name}}_{{count}}_grid_x = {{kernel_grid[0]}};
        uint32_t {{kernel_name}}_{{count}}_grid_y = {{kernel_grid[1]}};
        uint32_t {{kernel_name}}_{{count}}_grid_z = {{kernel_grid[2]}};
        {% for val, var in const_args_map.items() %}
        int64_t {{var}} = {{val}};
        {% endfor %}
        void *{{kernel_name}}_{{count}}_args[] = {
        {% for arg in kernel_args -%}
            &{{ arg }},
        {% endfor %}
        };
        CUDA_CHECK(cuLaunchKernel({{kernel_name}}_{{count}}_cufunc,
                                {{kernel_name}}_{{count}}_grid_x, {{kernel_name}}_{{count}}_grid_y, {{kernel_name}}_{{count}}_grid_z,  // grid dim
                                32 * {{kernel_name}}_{{count}}_config.num_warps, 1, 1,   // block dim
                                {{kernel_name}}_{{count}}_config.shared_memory, custream, // shared memory and stream
                                {{kernel_name}}_{{count}}_args, 0)); // arguments
        """
        )

        # Dynamic-mode kernel call template: grid is computed at runtime.
        self.kernel_call_dynamic_tpl = Template(
            """
        auto {{kernel_name}}_{{count}}_config = launch_config_map_["{{kernel_name}}"];
        auto {{kernel_name}}_{{count}}_cufunc = cufunc_map_["{{kernel_name}}"];
        uint32_t {{kernel_name}}_{{count}}_grid_x = {{grid_x_expr}};
        uint32_t {{kernel_name}}_{{count}}_grid_y = {{grid_y_expr}};
        uint32_t {{kernel_name}}_{{count}}_grid_z = {{grid_z_expr}};
        {% for var, expr in shape_args %}
        int64_t {{var}} = {{expr}};
        {% endfor %}
        {% for val, var in const_args_map.items() %}
        int64_t {{var}} = {{val}};
        {% endfor %}
        void *{{kernel_name}}_{{count}}_args[] = {
        {% for arg in kernel_args -%}
            &{{ arg }},
        {% endfor %}
        };
        CUDA_CHECK(cuLaunchKernel({{kernel_name}}_{{count}}_cufunc,
                                {{kernel_name}}_{{count}}_grid_x, {{kernel_name}}_{{count}}_grid_y, {{kernel_name}}_{{count}}_grid_z,  // grid dim
                                32 * {{kernel_name}}_{{count}}_config.num_warps, 1, 1,   // block dim
                                {{kernel_name}}_{{count}}_config.shared_memory, custream, // shared memory and stream
                                {{kernel_name}}_{{count}}_args, 0)); // arguments
        """
        )

        self.kernel_count = {}

    def _is_symbol_or_intermediate(self, name):
        """Check if a name is a shape symbol (s0, s1) or intermediate (ps0, ps1)."""
        return name in self._known_symbols

    def visit_Module(self, node):
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "call":
                for stmt in child.body:
                    self.dispatch_stmt(stmt)

    def dispatch_stmt(self, stmt):
        # Input variable unpacking — skip
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
            and isinstance(stmt.value, ast.Name)
            and stmt.value.id == "args"
        ):
            return

        # Return statement — skip
        if isinstance(stmt, ast.Return):
            return

        # args.clear() — skip
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "args"
            and stmt.value.func.attr == "clear"
        ):
            return

        # with statement — recurse into body
        if isinstance(stmt, ast.With):
            for inner in stmt.body:
                self.dispatch_stmt(inner)
            return

        # del statement — skip
        if isinstance(stmt, ast.Delete):
            return

        # torch.cuda.set_device(0)
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "set_device"
        ):
            return

        # stream0 = get_raw_stream(0)
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "get_raw_stream"
        ):
            return

        # assert_size_stride — runtime shape check
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "assert_size_stride"
        ):
            self.handle_shape_check(stmt)
            return

        # empty_strided_cuda — tensor allocation
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "empty_strided_cuda"
        ):
            self.handle_alloc(stmt)
            return

        # In dynamic mode: symbol binding  s0 = primals_1  (already handled by TensorIOVisitor, skip)
        if self.dynamic and isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name):
            target = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None
            value = stmt.value.id
            if target and target in self.symbol_bindings and self.symbol_bindings[target] == value:
                # Symbol binding — skip, already emitted by template
                return

        # In dynamic mode: intermediate shape computation  ps0 = s0*s3
        if self.dynamic and isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.BinOp):
            target = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None
            if target:
                self._known_symbols.add(target)
                if target not in self._emitted_vars:
                    expr = _expr_to_cpp(stmt.value)
                    self._emitted_vars.add(target)
                    self.codes.append(f"\n        int64_t {target} = {expr};")
                return

        # In dynamic mode: xnumel assignment like  triton_poi_fused_clone_0_xnumel = s0*s1*s2
        if self.dynamic and isinstance(stmt, ast.Assign):
            target = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None
            if target and "_xnumel" in target:
                self._known_symbols.add(target)
                if target not in self._emitted_vars:
                    expr = _expr_to_cpp(stmt.value)
                    self._emitted_vars.add(target)
                    self.codes.append(f"\n        int64_t {target} = {expr};")
                return

        # Variable assignment (rhs is a Name) — tensor aliasing
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name):
            self.handle_assign_tensor(stmt)
            return

        # Variable assignment (rhs is reinterpret_tensor)
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "reinterpret_tensor"
        ):
            self.handle_assign_reinterpret(stmt)
            return

        # kernel.run() call
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "run"
        ):
            self.handle_kernel(stmt)
            return

        # Expr-level comment (string literal) — skip
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return

        # extern_kernels.bmm(...) — unsupported, recommend max_autotune_gemm_backends='TRITON'
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "bmm"
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "extern_kernels"
        ):
            raise NotImplementedError(
                "extern_kernels.bmm is not supported. "
                "Add max_autotune_gemm_backends='TRITON' to COMPILE_OPTIONS to force Triton template kernels."
            )

        raise NotImplementedError(f"Unsupported statement: {ast.dump(stmt)}")

    # Handle shape check
    def handle_shape_check(self, stmt):
        if self.dynamic:
            # In dynamic mode, skip shape checks — shapes are symbolic
            return
        var_name = stmt.value.args[0].id
        shape = self.input_shapes_map[var_name]
        self.codes.append(self.check_shape_tpl.render(var_name=var_name, shape=shape))

    # Handle tensor allocation
    def handle_alloc(self, stmt):
        var_name = stmt.targets[0].id
        if self.dynamic:
            self._handle_alloc_dynamic(stmt)
            return
        if var_name in self.output_names:
            shape = self.output_shapes_map[var_name]
            self.codes.append(
                self.allocate_output_tpl.render(
                    var_name=var_name,
                    output_index=self.output_names.index(var_name),
                    shape=shape,
                )
            )
        else:
            shape_arg = ast.unparse(stmt.value.args[0])
            stride_arg = ast.unparse(stmt.value.args[1])
            shape_tuple = ast.literal_eval(shape_arg)
            stride_tuple = ast.literal_eval(stride_arg)
            shape = strided_shape_to_contiguous(shape_tuple, stride_tuple)
            self.codes.append(self.allocate_temp_tpl.render(var_name=var_name, shape=shape))

    def _handle_alloc_dynamic(self, stmt):
        """Handle empty_strided_cuda in dynamic mode with symbolic shapes."""
        var_name = stmt.targets[0].id
        shape_exprs = _shape_tuple_to_cpp(stmt.value.args[0])
        stride_exprs = _stride_tuple_to_list(stmt.value.args[1])
        contiguous_shape = _strided_shape_to_contiguous_dynamic(shape_exprs, stride_exprs)

        if var_name in self.output_names:
            self.codes.append(
                self.allocate_output_tpl.render(
                    var_name=var_name,
                    output_index=self.output_names.index(var_name),
                    shape=contiguous_shape,
                )
            )
        else:
            self.codes.append(self.allocate_temp_tpl.render(var_name=var_name, shape=contiguous_shape))

    def handle_assign_tensor(self, stmt):
        target_name = stmt.targets[0].id
        value_name = stmt.value.id
        self.codes.append(self.assign_tensor_tpl.render(target_name=target_name, value_name=value_name))

    def handle_assign_reinterpret(self, stmt):
        target_name = stmt.targets[0].id
        value_name = stmt.value.args[0].id
        self.codes.append(self.assign_tensor_tpl.render(target_name=target_name, value_name=value_name))

    # Handle kernel launch
    def handle_kernel(self, stmt):
        if self.dynamic:
            self._handle_kernel_dynamic(stmt)
            return
        call = stmt.value
        kernel_name = call.func.value.id
        kernel_args = []
        const_args_map = {}
        count = self.kernel_count.get(kernel_name, 0)
        for arg in call.args:
            if isinstance(arg, ast.Name):
                kernel_args.append(f"{arg.id}_ptr")
            elif isinstance(arg, ast.Constant):
                const_var = f"{kernel_name}_{count}_const_{len(const_args_map)}"
                const_args_map[arg.value] = const_var
                kernel_args.append(const_var)
            else:
                raise NotImplementedError(f"Unsupported arg type: {type(arg)}")

        kernel_grid = parse_grid_from_meta(kernel_name, self.output_path)

        self.codes.append(
            self.kernel_call_tpl.render(
                kernel_name=kernel_name,
                kernel_args=kernel_args,
                kernel_grid=kernel_grid,
                const_args_map=const_args_map,
                count=count,
            )
        )

        self.kernel_count.update({kernel_name: count + 1})

    def _handle_kernel_dynamic(self, stmt):
        """Handle kernel.run() in dynamic mode.

        In dynamic mode, kernel.run() args contain a mix of:
        - tensor args (Name nodes referencing buf/primals) → pass as _ptr
        - shape int args (Name nodes referencing s0, s1, ps0, etc. or xnumel vars)
        - constant args (Constant nodes) → declare as int64_t
        - grid= keyword with either grid(xnumel_var) or bmm_grid(s1, s0, s2, meta0)
        """
        call = stmt.value
        kernel_name = call.func.value.id
        kernel_args = []
        const_args_map = {}
        shape_args = []  # [(var_name, expr)] for shape int args
        count = self.kernel_count.get(kernel_name, 0)

        for arg in call.args:
            if isinstance(arg, ast.Name):
                name = arg.id
                if self._is_symbol_or_intermediate(name) or "_xnumel" in name:
                    # Shape int arg — pass directly as int64_t
                    var = f"{kernel_name}_{count}_shape_{len(shape_args)}"
                    shape_args.append((var, name))
                    kernel_args.append(var)
                else:
                    # Tensor arg
                    kernel_args.append(f"{name}_ptr")
            elif isinstance(arg, ast.Constant):
                const_var = f"{kernel_name}_{count}_const_{len(const_args_map)}"
                const_args_map[arg.value] = const_var
                kernel_args.append(const_var)
            else:
                raise NotImplementedError(f"Unsupported arg type in dynamic kernel: {ast.dump(arg)}")

        # Parse grid
        grid_x_expr, grid_y_expr, grid_z_expr = self._parse_dynamic_grid(call, kernel_name, count)

        self.codes.append(
            self.kernel_call_dynamic_tpl.render(
                kernel_name=kernel_name,
                kernel_args=kernel_args,
                grid_x_expr=grid_x_expr,
                grid_y_expr=grid_y_expr,
                grid_z_expr=grid_z_expr,
                shape_args=shape_args,
                const_args_map=const_args_map,
                count=count,
            )
        )

        self.kernel_count.update({kernel_name: count + 1})

    def _parse_dynamic_grid(self, call, kernel_name, count):
        """Parse the grid= keyword in a kernel.run() call for dynamic mode.

        Patterns:
        1. grid=grid(xnumel_var)  →  (xnumel + BLOCK - 1) / BLOCK, 1, 1
           where BLOCK is read from _meta.txt
        2. grid=bmm_grid(s1, s0, s2, meta0)  →  compute from args
        """
        grid_kw = None
        for kw in call.keywords:
            if kw.arg == "grid":
                grid_kw = kw.value
                break

        if grid_kw is None:
            raise NotImplementedError("Kernel call without 'grid' keyword.")

        # Case 1: grid=grid(xnumel_var) — pointwise kernel
        if isinstance(grid_kw, ast.Call) and isinstance(grid_kw.func, ast.Name) and grid_kw.func.id == "grid":
            numel_expr = _expr_to_cpp(grid_kw.args[0])
            # Read XBLOCK from meta.txt
            config = self._read_meta_blocks(kernel_name)
            xblock = config.get("XBLOCK", "1024")
            grid_x = f"({numel_expr} + {xblock} - 1) / {xblock}"
            return grid_x, "1", "1"

        # Case 2: grid=bmm_grid(batch, M, N, meta) — template/bmm kernel
        if (
            isinstance(grid_kw, ast.Call)
            and isinstance(grid_kw.func, ast.Attribute)
            and grid_kw.func.attr == "bmm_grid"
        ):
            # bmm_grid(batch, M, N, meta) where meta has BLOCK_M, BLOCK_N
            # grid = (ceil(M/BLOCK_M) * ceil(N/BLOCK_N), batch, 1)
            args = grid_kw.args
            batch_expr = _expr_to_cpp(args[0])
            m_expr = _expr_to_cpp(args[1])
            n_expr = _expr_to_cpp(args[2])
            # Get BLOCK_M, BLOCK_N from the meta dict (e.g. meta0, meta1)
            meta_name = args[3].id if isinstance(args[3], ast.Name) else None
            meta = self.meta_dicts.get(meta_name, {}) if meta_name else {}
            block_m = meta.get("BLOCK_M", 128)
            block_n = meta.get("BLOCK_N", 128)
            grid_x = f"(({m_expr} + {block_m} - 1) / {block_m}) * (({n_expr} + {block_n} - 1) / {block_n})"
            grid_y = batch_expr
            return grid_x, grid_y, "1"

        raise NotImplementedError(f"Unsupported grid pattern: {ast.dump(grid_kw)}")

    def _read_meta_blocks(self, kernel_name):
        """Read block constants from kernel _meta.txt file."""
        meta_path = os.path.join(self.output_path, f"{kernel_name}_meta.txt")
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        header = lines[0].split()
        values = lines[1].split()
        data = dict(zip(header, values))
        # Return a dict of block constants (XBLOCK, BLOCK_M, BLOCK_N, etc.)
        return {
            k: v for k, v in data.items() if k not in ("func_name", "shared", "num_warps", "XGRID", "YGRID", "ZGRID")
        }


def _parse_meta_dicts(source):
    """Parse top-level ``metaN = {...}`` dict assignments from the source.

    Returns:
        dict mapping meta variable name → dict of values, e.g.
        ``{"meta0": {"BLOCK_M": 128, "BLOCK_N": 128, ...}, ...}``
    """
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("meta")
            and isinstance(node.value, ast.Dict)
        ):
            name = node.targets[0].id
            d = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    d[k.value] = v.value
            result[name] = d
    return result


def _build_symbol_dim_map(symbol_bindings, tensor_input_names, source):
    """Build a mapping from shape symbols to (tensor_input_index, dim_index).

    In dynamic mode, ``assert_size_stride(primals_4, (s0, s1, s2), ...)``
    tells us that ``s0`` comes from ``primals_4.dim(0)``, ``s1`` from
    ``primals_4.dim(1)``, etc.

    Returns:
        dict mapping symbol_name → (tensor_input_index, dim_index)
        where tensor_input_index is the position among *tensor* inputs only.
    """
    tree = ast.parse(source)
    symbol_dim_map = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "call":
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "assert_size_stride"
            ):
                var_name = stmt.value.args[0].id
                if var_name not in tensor_input_names:
                    continue
                tensor_idx = tensor_input_names.index(var_name)
                shape_node = stmt.value.args[1]
                if isinstance(shape_node, ast.Tuple):
                    for dim_idx, elt in enumerate(shape_node.elts):
                        if isinstance(elt, ast.Name) and elt.id in symbol_bindings:
                            if elt.id not in symbol_dim_map:
                                symbol_dim_map[elt.id] = (tensor_idx, dim_idx)
    return symbol_dim_map


def render_kernel_file(model_path, op_name, output_path="./", dynamic=False):
    with open(model_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    triton_kernels_visitor = AsyncTritonAssignVisitor()
    triton_kernels_visitor.visit(tree)
    kernel_names = triton_kernels_visitor.kernels

    tensor_io_visitor = TensorIOVisitor()
    tensor_io_visitor.visit(tree)
    input_names = tensor_io_visitor.input_names
    output_names = tensor_io_visitor.output_names

    if dynamic:
        # Dynamic mode: shapes are symbolic, not concrete
        symbol_bindings = tensor_io_visitor.symbol_bindings
        int_arg_names = tensor_io_visitor.int_arg_names
        tensor_input_names = tensor_io_visitor.tensor_input_names

        # Filter output_names: exclude shape int symbols (s0, s1, ...) from outputs
        all_symbols = set(symbol_bindings.keys())
        tensor_output_names = [n for n in output_names if n not in all_symbols]

        # Build symbol → (tensor_input_index, dim_index) mapping
        symbol_dim_map = _build_symbol_dim_map(symbol_bindings, tensor_input_names, source)

        # Empty maps — not used in dynamic mode
        input_shapes_map = {}
        output_shapes_map = {}

        # Parse meta dicts (meta0, meta1, ...) for BLOCK_M/BLOCK_N values
        meta_dicts = _parse_meta_dicts(source)

        call_logic_visitor = CallLogicVisitor(
            input_shapes_map=input_shapes_map,
            output_shapes_map=output_shapes_map,
            output_names=tensor_output_names,
            output_path=output_path,
            dynamic=True,
            symbol_bindings=symbol_bindings,
            tensor_input_names=tensor_input_names,
            int_arg_names=int_arg_names,
            meta_dicts=meta_dicts,
        )
        call_logic_visitor.visit(tree)
        compute_codes = call_logic_visitor.codes

        env = Environment(
            loader=PackageLoader("mirai", "templates"),
            undefined=StrictUndefined,
        )
        tpl = env.get_template("kernel.cc.tpl")

        # For dynamic mode, input_args are only tensors, output_args have no fixed shape
        input_args = [TensorShape(name) for name in tensor_input_names]
        output_args = [TensorShape(name) for name in tensor_output_names]

        # Build symbol info for the template
        # symbol_info: list of {name, tensor_idx, dim_idx} for reading from input tensors
        symbol_info = []
        for sym, primal in sorted(symbol_bindings.items(), key=lambda x: x[0]):
            if sym in symbol_dim_map:
                tidx, didx = symbol_dim_map[sym]
                symbol_info.append(
                    {
                        "name": sym,
                        "tensor_name": tensor_input_names[tidx],
                        "tensor_idx": tidx,
                        "dim_idx": didx,
                    }
                )

        # Merge symbol info into _shapes.json (written by subprocess earlier)
        shapes_path = os.path.join(output_path, f"{op_name}_shapes.json")
        with open(shapes_path, "r") as f:
            shapes = json.load(f)
        shapes["dynamic"] = True
        shapes["symbol_bindings"] = symbol_bindings
        shapes["symbol_dim_map"] = {k: list(v) for k, v in symbol_dim_map.items()}
        shapes["int_arg_names"] = int_arg_names
        shapes["tensor_input_names"] = tensor_input_names
        shapes["tensor_output_names"] = tensor_output_names
        with open(shapes_path, "w") as f:
            json.dump(shapes, f, indent=2)

        result = tpl.render(
            op_name=op_name,
            input_args=input_args,
            output_args=output_args,
            kernel_names=kernel_names,
            compute_codes=compute_codes,
            dynamic=True,
            symbol_info=symbol_info,
        )
    else:
        # Static mode: original behavior
        shapes_path = os.path.join(output_path, f"{op_name}_shapes.json")
        with open(shapes_path, "r") as f:
            shapes = json.load(f)

        input_shapes = shapes["input_shapes"][0]
        output_shapes = shapes["output_shapes"]

        input_shapes_map = {name: shape for name, shape in zip(input_names, input_shapes)}
        output_shapes_map = {name: shape for name, shape in zip(output_names, output_shapes)}

        call_logic_visitor = CallLogicVisitor(
            input_shapes_map=input_shapes_map,
            output_shapes_map=output_shapes_map,
            output_names=output_names,
            output_path=output_path,
        )
        call_logic_visitor.visit(tree)
        compute_codes = call_logic_visitor.codes

        env = Environment(
            loader=PackageLoader("mirai", "templates"),
            undefined=StrictUndefined,
        )
        tpl = env.get_template("kernel.cc.tpl")

        input_args = [TensorShape(name, input_shape) for name, input_shape in zip(input_names, input_shapes)]
        output_args = [TensorShape(name, output_shape) for name, output_shape in zip(output_names, output_shapes)]

        result = tpl.render(
            op_name=op_name,
            input_args=input_args,
            output_args=output_args,
            kernel_names=kernel_names,
            compute_codes=compute_codes,
            dynamic=False,
        )

    with open(os.path.join(output_path, f"{op_name}.cc"), "w", encoding="utf-8") as f:
        f.write(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="model path of compiled module")
    parser.add_argument("model_name", type=str, help="model name to be set")
    parser.add_argument("--output_path", type=str, help="output path of generated code", default="./")

    args = parser.parse_args()
    model_path = args.model_path
    op_name = args.model_name
    output_path = args.output_path

    render_kernel_file(model_path, op_name, output_path)


if __name__ == "__main__":
    main()
