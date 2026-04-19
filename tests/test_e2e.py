"""End-to-end pipeline test.

Runs through the full mirai build pipeline stage by stage, saving
intermediate artifacts to a temporary directory for inspection.

Usage:
    # Normal: temp dir auto-cleaned on success
    pytest tests/test_e2e.py -v

    # Debug: keep artifacts for inspection
    pytest tests/test_e2e.py -v --keep-artifacts

    # Show artifact path even on success
    pytest tests/test_e2e.py -v -s
"""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import mirai
from mirai.codegen import hack_triton_file, render_kernel_file, render_build_file, render_api_file
from mirai.discovery import find_output_codes, find_ptxas
from mirai.pipeline import get_subprocess_env


@pytest.fixture(scope="module")
def artifacts_dir(request):
    keep = request.config.getoption("--keep-artifacts", default=False)
    tmp = tempfile.mkdtemp(prefix="mirai_e2e_")
    print(f"\n[E2E] Artifacts dir: {tmp}")
    yield Path(tmp)
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"\n[E2E] Keeping artifacts at: {tmp}")


@pytest.fixture(scope="module")
def sample_data():
    """Create sample inputs matching the PFFN model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs, tokens, dim, inner = 5000, 32, 512, 1536

    torch.manual_seed(42)
    inputs = torch.randn(bs, tokens, dim, device=device, requires_grad=True)
    w_gate = torch.randn(tokens, dim, inner, device=device, requires_grad=True) * 0.01
    b_gate = torch.zeros(tokens, inner, device=device, requires_grad=True)
    w_up = torch.randn(tokens, dim, inner, device=device, requires_grad=True) * 0.01
    b_up = torch.zeros(tokens, inner, device=device, requires_grad=True)
    w_down = torch.randn(tokens, inner, dim, device=device, requires_grad=True) * 0.01
    b_down = torch.zeros(tokens, dim, device=device, requires_grad=True)

    return [inputs, w_gate, b_gate, w_up, b_up, w_down, b_down]


@mirai.op(name="Pffn")
def pffn_func(inputs, w_gate, b_gate, w_up, b_up, w_down, b_down):
    inputs_t = inputs.transpose(0, 1).contiguous()
    gates = torch.bmm(inputs_t, w_gate) + b_gate.unsqueeze(1)
    gates = torch.nn.functional.silu(gates)
    vals = torch.bmm(inputs_t, w_up) + b_up.unsqueeze(1)
    outputs = torch.bmm(gates * vals, w_down) + b_down.unsqueeze(1)
    return outputs.transpose(0, 1).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestE2EPipeline:
    """Full pipeline test, one method per stage."""

    def test_stage1_torch_compile(self, artifacts_dir, sample_data):
        """Stage 1: torch.compile generates output_code.py."""
        from mirai.build import _setup_env, _clear_inductor_cache, _compile_and_execute, COMPILE_OPTIONS

        ptxas_path = find_ptxas()
        _setup_env(artifacts_dir, ptxas_path)
        _clear_inductor_cache()
        _compile_and_execute(pffn_func, sample_data)

        # Verify torch_compile_debug was created
        tc_debug = artifacts_dir / "torch_compile_debug"
        assert tc_debug.exists(), "torch_compile_debug not created"

        # Verify output_code.py exists
        code_paths = find_output_codes(str(artifacts_dir))
        assert "fwd" in code_paths, "Forward output_code.py not found"
        assert "bwd" in code_paths, "Backward output_code.py not found"

        # Save paths for next stages
        (artifacts_dir / "fwd_path.txt").write_text(str(code_paths["fwd"]))
        (artifacts_dir / "bwd_path.txt").write_text(str(code_paths["bwd"]))

    def test_stage2_patch_and_extract_ptx(self, artifacts_dir):
        """Stage 3: Patch output_code.py and extract PTX."""
        fwd_path = (artifacts_dir / "fwd_path.txt").read_text().strip()
        assert Path(fwd_path).exists(), "Run test_stage1 first"

        # Patch
        hack_triton_file(fwd_path, "PffnFwd", str(artifacts_dir))
        patched = artifacts_dir / "PffnFwd.py"
        assert patched.exists(), "Patched PffnFwd.py not created"

        # Verify patched code is valid Python
        ast.parse(patched.read_text())

        # Execute to extract PTX
        env = get_subprocess_env(find_ptxas())
        sub_env = dict(env)
        sub_env.pop("TORCH_COMPILE_DEBUG", None)
        sub_env.pop("TORCH_COMPILE_DEBUG_DIR", None)

        result = subprocess.run(
            [sys.executable, "PffnFwd.py"],
            env=sub_env,
            cwd=str(artifacts_dir),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"PTX extraction failed:\n{result.stderr}"

        # Verify PTX files were generated
        ptx_files = list(artifacts_dir.glob("*.ptx"))
        assert len(ptx_files) > 0, "No PTX files generated"

        meta_files = list(artifacts_dir.glob("*_meta.txt"))
        assert len(meta_files) > 0, "No meta files generated"

        # Verify shapes JSON
        shapes_file = artifacts_dir / "PffnFwd_shapes.json"
        assert shapes_file.exists(), "Shapes file not created"
        shapes = json.loads(shapes_file.read_text())
        assert "input_shapes" in shapes
        assert "output_shapes" in shapes

    def test_stage3_render_cc(self, artifacts_dir):
        """Stage 3 continued: Render C++ TF op from AST."""
        fwd_path = (artifacts_dir / "fwd_path.txt").read_text().strip()
        shapes_file = artifacts_dir / "PffnFwd_shapes.json"
        assert shapes_file.exists(), "Run test_stage2 first"

        render_kernel_file(fwd_path, "PffnFwd", str(artifacts_dir))

        cc_file = artifacts_dir / "PffnFwd.cc"
        assert cc_file.exists(), "PffnFwd.cc not generated"

        content = cc_file.read_text()
        assert "REGISTER_OP" in content
        assert "cuLaunchKernel" in content
        assert "PffnFwd" in content

    def test_stage4_build_sh(self, artifacts_dir):
        """Stage 4: Generate build.sh."""
        render_build_file(["PffnFwd"], str(artifacts_dir))

        build_sh = artifacts_dir / "build.sh"
        assert build_sh.exists()
        content = build_sh.read_text()
        assert "PffnFwd.cc" in content
        assert "PffnFwd.so" in content

    def test_stage5_api_wrapper(self, artifacts_dir):
        """Stage 5: Generate TF API wrapper."""
        fwd_path = (artifacts_dir / "fwd_path.txt").read_text().strip()
        bwd_path = (artifacts_dir / "bwd_path.txt").read_text().strip()

        with open(fwd_path) as f:
            fwd_source = f.read()
        with open(bwd_path) as f:
            bwd_source = f.read()

        result = render_api_file(
            fwd_source=fwd_source,
            bwd_source=bwd_source,
            op_name="Pffn",
            user_param_names=["inputs", "w_gate", "b_gate", "w_up", "b_up", "w_down", "b_down"],
            output_path=str(artifacts_dir),
        )

        api_file = Path(result)
        assert api_file.exists()
        content = api_file.read_text()
        # Valid Python
        ast.parse(content)
        assert "init_pffn" in content
        assert "custom_gradient" in content
