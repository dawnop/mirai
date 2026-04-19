import sys
import argparse
from pathlib import Path

from .codegen import render_build_file, render_api_file
from .discovery import find_ptxas
from .log import logger, setup_default_logging
from .pipeline import get_subprocess_env, process_kernel


def main():
    parser = argparse.ArgumentParser(description="MIRAI Operator Builder CLI")

    parser.add_argument("--model-name", required=True, help="Base model name (e.g., Pffn)")
    parser.add_argument("--fwd-path", required=True, help="Path to Forward output_code.py")
    parser.add_argument("--ptxas", default=None, help="Path to ptxas executable (auto-detected if not set)")
    parser.add_argument("--bwd-path", default=None, help="Path to Backward output_code.py (Optional)")
    parser.add_argument("--target", default="./generated", help="Target output directory (default: ./generated)")
    parser.add_argument("--version", default="tf32", help="Version tag (default: tf32)")
    parser.add_argument("--params", nargs="+", default=None, help="User parameter names in order (for API generation)")
    parser.add_argument("--asset-base-prefix", default="", help="Base prefix for asset loading path in generated API")

    args = parser.parse_args()

    setup_default_logging()

    # Auto-detect ptxas if not provided
    ptxas_path = args.ptxas or find_ptxas()
    env = get_subprocess_env(ptxas_path)
    target_path = Path(args.target).resolve()

    fwd_name = f"{args.model_name}Fwd"
    bwd_name = f"{args.model_name}Bwd"

    compiled_kernels = []

    # Forward (required)
    try:
        process_kernel(args.fwd_path, fwd_name, args.version, target_path, env)
        compiled_kernels.append(fwd_name)
    except Exception as e:
        logger.error("Forward pass failed: %s", e)
        sys.exit(1)

    # Backward (optional)
    if args.bwd_path:
        try:
            process_kernel(args.bwd_path, bwd_name, args.version, target_path, env)
            compiled_kernels.append(bwd_name)
        except Exception as e:
            logger.error("Backward pass failed: %s", e)
            sys.exit(1)
    else:
        logger.info("No backward path provided, skipping backward pass.")

    # Generate build.sh
    if compiled_kernels:
        logger.info("[Build] Rendering build script for: %s", compiled_kernels)
        render_build_file(compiled_kernels, str(target_path))
        logger.info("[Build] Artifacts generated in: %s", target_path)
    else:
        logger.warning("No kernels were compiled.")

    # Generate TF API wrapper (requires both fwd and bwd, plus --params)
    if args.bwd_path and args.params:
        logger.info("[API] Generating TF API wrapper...")
        with open(args.fwd_path, "r", encoding="utf-8") as f:
            fwd_source = f.read()
        with open(args.bwd_path, "r", encoding="utf-8") as f:
            bwd_source = f.read()

        render_api_file(
            fwd_source=fwd_source,
            bwd_source=bwd_source,
            op_name=args.model_name,
            user_param_names=args.params,
            output_path=str(target_path),
            asset_base_prefix=args.asset_base_prefix,
        )
    elif args.bwd_path and not args.params:
        logger.info("Skipped API generation (--params not provided).")
    else:
        logger.info("Skipped API generation (no backward path).")


if __name__ == "__main__":
    main()
