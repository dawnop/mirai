import ast
import os
import argparse
import textwrap

from ..log import logger

# ==============================================================================
# Injected Code Templates
# ==============================================================================

SAVE_PTX_SRC = textwrap.dedent(
    """
    def save_ptx(triton_kernel):
        metadata = triton_kernel.metadata
        const = triton_kernel.src.constants
        def dump_ptx(file_name, data):
            binary = isinstance(data, bytes)
            if not binary:
                data = str(data)
            mode = "wb" if binary else "w"
            with open(f"{file_name}.ptx", mode) as f:
                f.write(data)
        def dump_ptx_cfg(file_name, name, shared, num_warps, const):
            header = f"func_name shared num_warps"
            meta_str = f"{name} {shared} {num_warps}"
            for k, v in const.items():
                header += f" {k}"
                meta_str += f" {v}"
            with open(f"{file_name}_meta.txt", "w") as f:
                f.write(header)
                f.write("\\n")
                f.write(meta_str)
        
        # Handle different triton versions where asm might be different
        ptx_data = triton_kernel.asm["ptx"]
        dump_ptx(metadata.name, ptx_data)
        dump_ptx_cfg(metadata.name, metadata.name, metadata.shared, metadata.num_warps, const)
"""
)

RECORD_IO_SRC_TEMPLATE = textwrap.dedent(
    """
    def record_input_output():
        import functools
        import json
        import torch
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                def physical_shape(tensor):
                    shape = tensor.shape
                    stride = tensor.stride()
                    # Sort shape by stride descending
                    sorted_shape = [x for _, x in sorted(zip(stride, shape), reverse=True)]
                    return sorted_shape
                def extract_shape(x):
                    if isinstance(x, torch.Tensor):
                        if x.is_contiguous():
                            return tuple(x.shape)
                        else:
                            return tuple(physical_shape(x))
                    elif isinstance(x, (list, tuple)):
                        return [extract_shape(e) for e in x]
                    else:
                        return None
                
                input_shapes = extract_shape(args)
                output = func(*args, **kwargs)
                output_shapes = extract_shape(output)
                
                record = {
                    'input_shapes': input_shapes,
                    'output_shapes': output_shapes
                }
                # op_name will be replaced
                with open('{op_name}_shapes.json', 'w') as f:
                    f.write(json.dumps(record) + '\\n')
                
                return output
            return wrapper
        return decorator
"""
)

# ==============================================================================
# AST Transformers
# ==============================================================================


class ModuleInjector(ast.NodeTransformer):
    """
    Injects helper functions (save_ptx, record_input_output) at the top of the module
    and adds the decorator to the 'call' function.
    """

    def __init__(self, op_name):
        self.op_name = op_name
        # Prepare AST for save_ptx
        self.save_ptx_nodes = ast.parse(SAVE_PTX_SRC).body
        # Prepare AST for record_input_output
        record_src = RECORD_IO_SRC_TEMPLATE.replace("{op_name}", self.op_name)
        self.record_io_nodes = ast.parse(record_src).body

    def visit_Module(self, node):
        # 1. Insert helper functions at the beginning
        # We insert them before the 'call' function usually, or just at the top.
        # To be safe, let's put them after imports (heuristic: index 0 or later)
        insert_idx = 0
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                insert_idx = i + 1

        # Insert helper definitions
        node.body[insert_idx:insert_idx] = self.save_ptx_nodes + self.record_io_nodes

        # 2. Add decorator to 'call' function
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if node.name == "call":
            # Add @record_input_output() decorator
            decorator = ast.Call(func=ast.Name(id="record_input_output", ctx=ast.Load()), args=[], keywords=[])
            node.decorator_list.insert(0, decorator)
        return node


class KernelRunnerHook(ast.NodeTransformer):
    """
    Intercepts the kernel.run() call to:
    1. Capture the kernel object.
    2. Extract constants and grid.
    3. Call save_ptx.
    """

    def _create_hook_block(self, stmt, kernel_name_base):
        """
        Constructs the AST block to replace the original run() expression.
        """
        var_name = f"{kernel_name_base}_kernel"
        const_var = f"{kernel_name_base}_const"

        # 1. Assign kernel: var_name = stmt.value (the run call)
        assign_kernel = ast.Assign(targets=[ast.Name(id=var_name, ctx=ast.Store())], value=stmt.value)

        # 2. Assign constants: const_var = var_name.src.constants
        assign_const = ast.Assign(
            targets=[ast.Name(id=const_var, ctx=ast.Store())],
            value=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id=var_name, ctx=ast.Load()), attr="src", ctx=ast.Load()),
                attr="constants",
                ctx=ast.Load(),
            ),
        )

        # 3. Handle Grid
        # Find 'grid' keyword argument in the original call
        grid_call = None
        for kw in stmt.value.keywords:
            if kw.arg == "grid":
                grid_call = kw.value
                break

        if grid_call is None:
            raise NotImplementedError("Kernel call without 'grid' argument found.")

        # Construct the grid assignment logic
        # target: const_var['XGRID'], const_var['YGRID'], const_var['ZGRID']
        targets = ast.Tuple(
            elts=[
                ast.Subscript(
                    value=ast.Name(id=const_var, ctx=ast.Load()), slice=ast.Constant(value=dim), ctx=ast.Store()
                )
                for dim in ["XGRID", "YGRID", "ZGRID"]
            ],
            ctx=ast.Store(),
        )

        # value: logic to call the grid function
        if isinstance(grid_call.func, ast.Attribute):
            # Case 1: torch._inductor.kernel.bmm.bmm_grid(...)
            right_value = ast.Call(func=grid_call.func, args=grid_call.args, keywords=grid_call.keywords)
        else:
            # Case 2: grid(NUM_META)(meta) pattern
            # We need to call the grid function with the constants dictionary
            grid_func_call = ast.Call(func=grid_call.func, args=grid_call.args, keywords=grid_call.keywords)
            right_value = ast.Call(func=grid_func_call, args=[ast.Name(id=const_var, ctx=ast.Load())], keywords=[])

        assign_grid = ast.Assign(targets=[targets], value=right_value)

        # 4. Call save_ptx(var_name)
        call_save = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="save_ptx", ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg="triton_kernel", value=ast.Name(id=var_name, ctx=ast.Load()))],
            )
        )

        return [assign_kernel, assign_const, assign_grid, call_save]

    def visit_FunctionDef(self, node):
        # Only process the 'call' function
        if node.name == "call":
            self.generic_visit(node)
        return node

    def visit_With(self, node):
        # Check for "with torch.cuda._DeviceGuard(0):"
        # This is a heuristic to find the inner block where kernels are launched
        is_device_guard = False
        if len(node.items) == 1:
            expr = node.items[0].context_expr
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "_DeviceGuard":
                is_device_guard = True

        if is_device_guard:
            new_body = []
            for stmt in node.body:
                # Look for: kernel.run(...)
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "run"
                    and isinstance(stmt.value.func.value, ast.Name)
                ):

                    kernel_name_base = stmt.value.func.value.id
                    # Replace the single Expr with our block of assignments
                    injected_block = self._create_hook_block(stmt, kernel_name_base)
                    new_body.extend(injected_block)
                else:
                    new_body.append(stmt)
            node.body = new_body

        return node


_SIMPLE_MAIN = textwrap.dedent(
    """\
    if __name__ == '__main__':
        benchmark_compiled_module()
    """
)


class MainStripper(ast.NodeTransformer):
    """Strip ``benchmark_compiled_module`` down to a single ``call()`` invocation.

    The original body allocates random tensors, runs ``print_performance``
    (timing + stdout print) then returns.  We keep the tensor allocation but
    replace the ``return print_performance(...)`` with a plain ``fn()`` call,
    and swap the ``if __name__`` block to just invoke that function.

    This way the ``save_ptx`` / ``record_input_output`` hooks still fire, but
    nothing is printed.
    """

    def visit_FunctionDef(self, node):
        if node.name == "benchmark_compiled_module":
            # Replace ``return print_performance(fn, ...)`` with ``fn()``
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    # Replace with: fn()
                    new_body.append(
                        ast.Expr(value=ast.Call(func=ast.Name(id="fn", ctx=ast.Load()), args=[], keywords=[]))
                    )
                else:
                    new_body.append(stmt)
            node.body = new_body
        return node

    def visit_If(self, node):
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(isinstance(op, ast.Eq) for op in test.ops)
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
        ):
            new_node = ast.parse(_SIMPLE_MAIN).body[0]
            return ast.copy_location(new_node, node)
        return node


# ==============================================================================
# Main Logic
# ==============================================================================


def hack_triton_file(model_path, op_name, output_path="./"):
    logger.info("Patching file: %s -> %s.py", model_path, op_name)

    with open(model_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.error("Error parsing %s: %s", model_path, e)
        return

    # Debug: Dump original AST
    # with open(f"{op_name}.py.ast", "w") as f:
    #     f.write(ast.dump(tree, indent=2))

    # Apply Transformers
    # 1. Inject helper functions and decorators
    tree = ModuleInjector(op_name).visit(tree)

    # 2. Hook kernel execution to save PTX
    tree = KernelRunnerHook().visit(tree)

    # 3. Strip benchmark/print_performance — keep only the call() invocation
    tree = MainStripper().visit(tree)

    # Fix locations (line numbers) after modification
    ast.fix_missing_locations(tree)

    # Generate code
    result = ast.unparse(tree)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{op_name}.py")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)
    logger.info("Successfully generated %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Hack Triton generated code to extract PTX and shapes.")
    parser.add_argument("model_path", type=str, help="Path to the original output_code.py")
    parser.add_argument("model_name", type=str, help="Name of the operator (e.g., PffnFwd)")
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    hack_triton_file(args.model_path, args.model_name, args.output_path)


if __name__ == "__main__":
    main()
