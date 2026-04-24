"""Shared pipeline utilities used by both build.py (programmatic API) and main.py (CLI)."""

import os
import re
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


def ptxas_cuda_version(ptxas_path):
    """Return the CUDA version string (e.g. ``"12.2"``) from a ptxas binary.

    Returns None if the version cannot be determined.
    """
    try:
        out = subprocess.check_output([ptxas_path, "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"release (\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def cuda_version_to_ptx_isa(cuda_version):
    """Convert a CUDA version string to the expected PTX ISA version string.

    >>> cuda_version_to_ptx_isa("12.2")
    '8.2'
    >>> cuda_version_to_ptx_isa("11.7")
    '7.7'
    """
    major, minor = map(int, cuda_version.split("."))
    return f"{major - 4}.{minor}"


def read_ptx_isa_version(ptx_path):
    """Read the ``.version X.Y`` directive from a PTX file.

    Returns the version string (e.g. ``"8.2"``) or None.
    """
    with open(ptx_path, "r") as f:
        for line in f:
            m = re.match(r"\.version\s+(\S+)", line.strip())
            if m:
                return m.group(1)
    return None


def verify_ptx_version(ptxas_path, ptx_dir):
    """Check that generated PTX files match the expected ISA version for *ptxas_path*.

    Raises ``RuntimeError`` on mismatch.  Returns the verified version string
    on success, or None if verification was skipped (no ptx files / cannot
    determine version).
    """
    cuda_ver = ptxas_cuda_version(ptxas_path)
    if cuda_ver is None:
        return None

    expected = cuda_version_to_ptx_isa(cuda_ver)

    for ptx_file in Path(ptx_dir).glob("triton*.ptx"):
        actual = read_ptx_isa_version(str(ptx_file))
        if actual is None:
            continue
        if actual != expected:
            raise RuntimeError(
                f"PTX ISA version mismatch: generated .version {actual}, "
                f"but ptxas {ptxas_path} expects .version {expected}. "
                f"TRITON_PTXAS_PATH may not have taken effect — "
                f"ensure mirai.build() is called with an absolute ptxas path."
            )
        return expected

    return None


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

    # 3. Verify PTX ISA version matches the ptxas we configured
    _ptxas = env.get("TRITON_PTXAS_PATH")
    if _ptxas:
        verified = verify_ptx_version(_ptxas, str(output_root))
        if verified:
            logger.info("[Stage 3] PTX ISA version: %s (matches ptxas)", verified)

    # 4. Render C++ TF op (use original output_code.py, not the patched one)
    logger.info("[Stage 3] Rendering %s.cc...", kernel_name)
    render_kernel_file(str(model_path), kernel_name, str(output_root), dynamic=dynamic)

    # 5. Move PTX and meta files to version directory
    dest_dir = output_root / version / kernel_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["triton*.ptx", "triton*_meta.txt"]:
        for file_path in output_root.glob(pattern):
            try:
                shutil.move(str(file_path), str(dest_dir / file_path.name))
                logger.info("  -> Moved %s to %s", file_path.name, dest_dir)
            except shutil.Error as e:
                logger.warning("  -> Failed to move %s: %s", file_path.name, e)
