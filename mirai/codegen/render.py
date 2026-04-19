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
    def __init__(self):
        self.input_names = []
        self.output_names = []

    def visit_Module(self, node):
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "call":
                for stmt in child.body:
                    # input
                    if (
                        isinstance(stmt, ast.Assign)
                        and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
                        and isinstance(stmt.value, ast.Name)
                        and stmt.value.id == "args"
                    ):
                        self.input_names = [elt.id for elt in stmt.targets[0].elts if isinstance(elt, ast.Name)]
                    # output
                    elif isinstance(stmt, ast.Return):
                        self.output_names.extend(self._extract_return_vars(stmt.value))

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


class CallLogicVisitor(ast.NodeVisitor):
    def __init__(self, input_shapes_map, output_shapes_map, output_names, output_path="./"):

        self.input_shapes_map = input_shapes_map
        self.output_shapes_map = output_shapes_map
        self.output_names = output_names
        self.output_path = output_path

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

        self.kernel_count = {}

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

        # Variable assignment (rhs is a Name)
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

        raise NotImplementedError(f"Unsupported statement: {ast.dump(stmt)}")

    # Handle shape check
    def handle_shape_check(self, stmt):
        var_name = stmt.value.args[0].id
        shape = self.input_shapes_map[var_name]
        self.codes.append(self.check_shape_tpl.render(var_name=var_name, shape=shape))

    # Handle tensor allocation
    def handle_alloc(self, stmt):
        var_name = stmt.targets[0].id
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


def render_kernel_file(model_path, op_name, output_path="./"):
    with open(model_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    shapes_path = os.path.join(output_path, f"{op_name}_shapes.json")
    with open(shapes_path, "r") as f:
        shapes = json.load(f)

    input_shapes = shapes["input_shapes"][0]
    output_shapes = shapes["output_shapes"]

    triton_kernels_visitor = AsyncTritonAssignVisitor()
    triton_kernels_visitor.visit(tree)
    kernel_names = triton_kernels_visitor.kernels

    tensor_io_visitor = TensorIOVisitor()
    tensor_io_visitor.visit(tree)
    input_names = tensor_io_visitor.input_names
    output_names = tensor_io_visitor.output_names

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
