"""Shared pipeline utilities used by both build.py (programmatic API) and main.py (CLI)."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

from .codegen import hack_triton_file, render_kernel_file
from .log import logger


def get_subprocess_env(ptxas_path=None):
    """Get environment dict for subprocess calls."""
    env = os.environ.copy()
    if ptxas_path:
        env["TRITON_PTXAS_PATH"] = ptxas_path
    return env


def process_kernel(model_path, kernel_name, version, output_root, env, dynamic=False):
    """Process a single kernel: patch triton code, extract PTX, generate C++.

    Args:
        model_path: Path to the original output_code.py.
        kernel_name: Name for the kernel (e.g., "PffnFwd").
        version: Version tag for PTX organization (e.g., "tf32").
        output_root: Root output directory.
        env: Environment dict for subprocess calls.
        dynamic: If True, generate shape-generic TF ops.
    """
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Patch the triton generated code
    logger.info("[Stage 3] Patching %s...", kernel_name)
    hack_triton_file(str(model_path), kernel_name, str(output_root))

    # 2. Execute patched script to extract PTX and shapes
    kernel_script = output_root / f"{kernel_name}.py"
    if not kernel_script.exists():
        raise FileNotFoundError(f"Patched script not found: {kernel_script}")

    logger.info("[Stage 3] Executing %s.py to extract PTX...", kernel_name)
    # Disable torch compile debug in the subprocess — we only need it in Stage 1.
    # Without this the subprocess creates a stray torch_compile_debug/ in cwd.
    sub_env = dict(env)
    sub_env.pop("TORCH_COMPILE_DEBUG", None)
    sub_env.pop("TORCH_COMPILE_DEBUG_DIR", None)
    result = subprocess.run(
        [sys.executable, kernel_script.name],
        env=sub_env,
        check=True,
        cwd=str(output_root),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.debug("subprocess stdout:\n%s", result.stdout.rstrip())
    if result.stderr:
        logger.debug("subprocess stderr:\n%s", result.stderr.rstrip())

    # 3. Render C++ TF op (use original output_code.py, not the patched one)
    logger.info("[Stage 3] Rendering %s.cc...", kernel_name)
    render_kernel_file(str(model_path), kernel_name, str(output_root), dynamic=dynamic)

    # 4. Move PTX and meta files to version directory
    dest_dir = output_root / version / kernel_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["triton*.ptx", "triton*_meta.txt"]:
        for file_path in output_root.glob(pattern):
            try:
                shutil.move(str(file_path), str(dest_dir / file_path.name))
                logger.info("  -> Moved %s to %s", file_path.name, dest_dir)
            except shutil.Error as e:
                logger.warning("  -> Failed to move %s: %s", file_path.name, e)
