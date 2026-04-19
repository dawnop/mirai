import os
import shutil
from pathlib import Path

from .log import logger


def find_output_codes(debug_dir):
    """Find forward/backward output_code.py under the latest run directory.

    Only searches the most recent run_* directory to avoid picking up
    stale files from previous runs.

    Returns:
        dict with keys 'fwd' and optionally 'bwd', values are Path objects.
    """
    debug_dir = Path(debug_dir)

    # Locate all run_* directories, pick the latest by mtime
    tc_debug = debug_dir / "torch_compile_debug"
    if not tc_debug.exists():
        raise FileNotFoundError(f"No torch_compile_debug directory found under {debug_dir}. Did torch.compile run?")

    run_dirs = sorted(tc_debug.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {tc_debug}.")

    latest_run = run_dirs[0]
    logger.info("  Using run dir: %s", latest_run.name)

    # Search only within the latest run
    candidates = sorted(latest_run.rglob("**/output_code.py"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            f"No output_code.py found under {latest_run}. "
            "Inductor cache may have been hit — try clearing /tmp/torchinductor_* and re-running."
        )

    result = {}

    for path in candidates:
        path_str = str(path).lower()
        if "backward" in path_str or "bwd" in path_str:
            if "bwd" not in result:
                result["bwd"] = path
        else:
            if "fwd" not in result:
                result["fwd"] = path

    # Fallback: if we only found one and couldn't classify, treat as forward
    if "fwd" not in result and candidates:
        result["fwd"] = candidates[0]

    return result


def find_ptxas():
    """Find ptxas binary by priority: TRITON_PTXAS_PATH env > PATH > common CUDA locations."""
    # 1. Environment variable
    env_path = os.environ.get("TRITON_PTXAS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. PATH lookup
    path_result = shutil.which("ptxas")
    if path_result:
        return path_result

    # 3. Common CUDA locations
    common_paths = [
        "/usr/local/cuda/bin/ptxas",
        "/usr/local/cuda-12/bin/ptxas",
        "/usr/local/cuda-11/bin/ptxas",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    return None
