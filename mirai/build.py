import inspect
import os
import shutil
from pathlib import Path

import torch

from .discovery import find_output_codes, find_ptxas
from .codegen import render_build_file, render_api_file
from .log import logger, setup_default_logging
from .pipeline import get_subprocess_env, process_kernel


COMPILE_OPTIONS = {
    "max_autotune": True,
    "max_autotune_gemm_backends": "TRITON",
    "coordinate_descent_tuning": True,
    "allow_buffer_reuse": False,
    "inplace_buffers": False,
    "autotune_fallback_to_aten": False,
}


def _setup_env(output_dir, ptxas_path=None):
    """Configure environment for torch.compile debug output and TF32.

    Both env vars and runtime config are set because the env vars are only
    read once at ``import torch`` time, which has already happened.
    """
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(output_dir)
    dynamo_config.debug_dir_root = os.path.join(str(output_dir), "torch_compile_debug")
    inductor_config.trace.enabled = True

    if ptxas_path:
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _clear_inductor_cache():
    """Clear torchinductor cache so output_code.py is always regenerated."""
    import getpass
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / f"torchinductor_{getpass.getuser()}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Cleared inductor cache.")


def _compile_and_execute(func, sample_inputs, dynamic=False):
    """Stage 1: torch.compile the function and execute to trigger triton code generation."""
    logger.info("[Stage 1] Compiling with torch.compile (dynamic=%s) and executing...", dynamic)

    compiled = torch.compile(func, dynamic=dynamic, options=COMPILE_OPTIONS)

    # Forward pass
    output = compiled(*sample_inputs)

    # Backward pass to trigger backward kernel generation
    if isinstance(output, torch.Tensor) and output.requires_grad:
        loss = output.sum()
        loss.backward()
    elif isinstance(output, (tuple, list)):
        tensors = [t for t in output if isinstance(t, torch.Tensor) and t.requires_grad]
        if tensors:
            loss = sum(t.sum() for t in tensors)
            loss.backward()

    logger.info("[Stage 1] Done. Triton code generated.")


def build(func, sample_inputs, output_dir="./generated", version="tf32", ptxas=None, dynamic=False):
    """One-click build: from a @mirai.op decorated function to TF custom op artifacts.

    Args:
        func: Function decorated with @mirai.op(name="...").
        sample_inputs: List of sample tensors to drive torch.compile.
        output_dir: Directory for all generated artifacts.
        version: Version tag for PTX organization (default: "tf32").
        ptxas: Path to ptxas binary. If None, auto-detected.
        dynamic: If True, generate shape-generic TF ops that accept variable input shapes.
    """
    setup_default_logging()

    # Resolve op name
    op_name = getattr(func, "_mirai_op_name", None)
    if op_name is None:
        raise ValueError("Function must be decorated with @mirai.op(name='...'). Missing _mirai_op_name attribute.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ptxas
    ptxas_path = ptxas or find_ptxas()
    if ptxas_path:
        logger.info("Using ptxas: %s", ptxas_path)
    else:
        logger.warning("ptxas not found. PTX extraction may fail.")

    # === Stage 1: torch.compile + execute ===
    _setup_env(output_dir, ptxas_path)
    # Clear inductor cache to force regeneration of output_code.py
    # Without this, cached fwd won't produce new output_code.py on re-runs
    _clear_inductor_cache()
    _compile_and_execute(func, sample_inputs, dynamic=dynamic)

    # === Stage 2: Discover output_code.py files ===
    logger.info("[Stage 2] Searching for generated output_code.py files...")
    code_paths = find_output_codes(str(output_dir))

    fwd_path = code_paths.get("fwd")
    bwd_path = code_paths.get("bwd")

    if fwd_path:
        logger.info("  Forward:  %s", fwd_path)
    if bwd_path:
        logger.info("  Backward: %s", bwd_path)

    if not fwd_path:
        raise FileNotFoundError("Could not find forward output_code.py. torch.compile may have failed.")

    # === Stage 3: Extract PTX + Generate C++ ===
    env = get_subprocess_env(ptxas_path)
    compiled_kernels = []

    fwd_name = f"{op_name}Fwd"
    bwd_name = f"{op_name}Bwd"

    # Forward (required)
    process_kernel(fwd_path, fwd_name, version, output_dir, env, dynamic=dynamic)
    compiled_kernels.append(fwd_name)

    # Backward (optional)
    if bwd_path:
        process_kernel(bwd_path, bwd_name, version, output_dir, env, dynamic=dynamic)
        compiled_kernels.append(bwd_name)
    else:
        logger.info("No backward output_code.py found, skipping backward pass.")

    # === Stage 4: Generate build.sh ===
    logger.info("[Stage 4] Generating build.sh...")
    render_build_file(compiled_kernels, str(output_dir))

    # === Stage 5: Generate TF API wrapper ===
    if bwd_path:
        logger.info("[Stage 5] Generating TF API wrapper...")
        original_func = getattr(func, "_mirai_original_func", None)
        if original_func is not None:
            user_param_names = list(inspect.signature(original_func).parameters.keys())
        else:
            user_param_names = list(inspect.signature(func).parameters.keys())

        with open(fwd_path, "r", encoding="utf-8") as f:
            fwd_source = f.read()
        with open(bwd_path, "r", encoding="utf-8") as f:
            bwd_source = f.read()

        render_api_file(
            fwd_source=fwd_source,
            bwd_source=bwd_source,
            op_name=op_name,
            user_param_names=user_param_names,
            output_path=str(output_dir),
        )
    else:
        logger.info("[Stage 5] Skipped API generation (no backward pass).")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("Kernels: %s", ", ".join(compiled_kernels))
    logger.info("Artifacts:")
    for name in compiled_kernels:
        logger.info("  - %s.cc  (TF custom op C++ source)", name)
    logger.info("  - build.sh  (compilation script)")
    if bwd_path:
        logger.info("  - %s_api.py  (TF API wrapper)", op_name.lower())
    logger.info("  - %s/  (PTX files and metadata)", version)
    logger.info("")
    logger.info("Next step: cd %s && bash build.sh", output_dir)
