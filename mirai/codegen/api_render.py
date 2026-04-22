import ast
import os
import argparse

from jinja2 import Environment, PackageLoader, StrictUndefined

from ..log import logger
from .render import TensorIOVisitor


def _parse_io(source_code):
    """Parse a forward or backward output_code.py to extract input/output variable names.

    Returns:
        (input_names, output_names, visitor) — lists of variable name strings + visitor.
    """
    tree = ast.parse(source_code)
    visitor = TensorIOVisitor()
    visitor.visit(tree)
    return visitor.input_names, visitor.output_names, visitor


def render_api_file(fwd_source, bwd_source, op_name, user_param_names, output_path="./", asset_base_prefix=""):
    """Generate a TF API wrapper file that maps user-friendly parameter names
    to internal variable names and connects forward outputs to backward inputs.

    Args:
        fwd_source: Source code string of the forward output_code.py.
        bwd_source: Source code string of the backward output_code.py.
        op_name: Base model name (e.g., "Pffn").
        user_param_names: List of user parameter names (e.g., ["inputs", "w_gate", ...]).
        output_path: Directory to write the generated API file.
        asset_base_prefix: Base prefix path for asset loading (e.g., "code/ops/pffn/").
    """
    fwd_op_name = f"{op_name}Fwd"
    bwd_op_name = f"{op_name}Bwd"

    # Parse forward IO
    fwd_input_names, fwd_output_names, fwd_visitor = _parse_io(fwd_source)

    # Parse backward IO
    bwd_input_names, bwd_output_names, bwd_visitor = _parse_io(bwd_source)

    # In dynamic mode, filter to tensor-only inputs/outputs
    # (the TF op only receives tensors, not shape ints)
    fwd_is_dynamic = bool(fwd_visitor.symbol_bindings)
    if fwd_is_dynamic:
        fwd_symbols = set(fwd_visitor.symbol_bindings.keys())
        fwd_input_names = fwd_visitor.tensor_input_names
        fwd_output_names = [n for n in fwd_output_names if n not in fwd_symbols]

    bwd_is_dynamic = bool(bwd_visitor.symbol_bindings)
    if bwd_is_dynamic:
        bwd_symbols = set(bwd_visitor.symbol_bindings.keys())
        bwd_input_names = bwd_visitor.tensor_input_names
        bwd_output_names = [n for n in bwd_output_names if n not in bwd_symbols]

    # === Build mappings ===

    # 1. User params -> forward input names (positional)
    #    user_param_names[i] -> fwd_input_names[i]
    fwd_param_mapping = list(zip(user_param_names, fwd_input_names))

    # 2. Forward output -> backward input (saved tensors connection)
    #    fwd_output_names[0] is the real result
    #    fwd_output_names[1:] are saved tensors for backward
    #    bwd_input_names[-1] is tangents (gradient input)
    #    bwd_input_names[:-1] are saved tensors from forward
    fwd_result_name = fwd_output_names[0]
    fwd_saved_tensors = fwd_output_names[1:]
    tangent_name = bwd_input_names[-1]
    bwd_saved_inputs = bwd_input_names[:-1]

    # bwd_param_mapping: [(bwd_input_name, fwd_output_var_name), ...]
    # Maps bwd_saved_inputs[i] -> fwd_saved_tensors[i] (positional)
    bwd_param_mapping = list(zip(bwd_saved_inputs, fwd_saved_tensors))

    # 3. Backward output names -> gradient names for user params
    #    bwd_output_names[i] corresponds to grad of user_param_names[i]
    grad_names = [f"d_{name}" for name in user_param_names]

    # Render template
    env = Environment(
        loader=PackageLoader("mirai", "templates"),
        undefined=StrictUndefined,
    )
    tpl = env.get_template("api.py.tpl")

    result = tpl.render(
        op_name=op_name,
        fwd_op_name=fwd_op_name,
        bwd_op_name=bwd_op_name,
        user_param_names=user_param_names,
        fwd_param_mapping=fwd_param_mapping,
        fwd_output_names=fwd_output_names,
        fwd_result_name=fwd_result_name,
        bwd_param_mapping=bwd_param_mapping,
        bwd_output_names=grad_names,
        tangent_name=tangent_name,
        asset_base_prefix=asset_base_prefix,
    )

    api_file_path = os.path.join(output_path, f"{op_name.lower()}_api.py")
    with open(api_file_path, "w", encoding="utf-8") as f:
        f.write(result)

    logger.info("[API] Generated %s", api_file_path)
    return api_file_path


def main():
    parser = argparse.ArgumentParser(description="Generate TF API wrapper from forward/backward output_code.py")
    parser.add_argument("--fwd-path", required=True, help="Path to forward output_code.py")
    parser.add_argument("--bwd-path", required=True, help="Path to backward output_code.py")
    parser.add_argument("--op-name", required=True, help="Base op name (e.g., Pffn)")
    parser.add_argument("--params", required=True, nargs="+", help="User parameter names in order")
    parser.add_argument("--output-path", default="./", help="Output directory")
    parser.add_argument("--asset-base-prefix", default="", help="Base prefix for asset loading path")

    args = parser.parse_args()

    with open(args.fwd_path, "r", encoding="utf-8") as f:
        fwd_source = f.read()
    with open(args.bwd_path, "r", encoding="utf-8") as f:
        bwd_source = f.read()

    render_api_file(
        fwd_source=fwd_source,
        bwd_source=bwd_source,
        op_name=args.op_name,
        user_param_names=args.params,
        output_path=args.output_path,
        asset_base_prefix=args.asset_base_prefix,
    )


if __name__ == "__main__":
    main()
