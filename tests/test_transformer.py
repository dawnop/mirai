"""Tests for mirai.codegen.transformer — AST transformers."""

import ast
from pathlib import Path

import pytest

from mirai.codegen.transformer import ModuleInjector, KernelRunnerHook, MainStripper

FIXTURES = Path(__file__).parent / "fixtures"
FWD_CODE = FIXTURES / "fwd_output_code.py"


@pytest.fixture
def fwd_source():
    return FWD_CODE.read_text()


@pytest.fixture
def fwd_tree(fwd_source):
    return ast.parse(fwd_source)


class TestModuleInjector:
    def test_injects_save_ptx(self, fwd_tree):
        tree = ModuleInjector("TestOp").visit(fwd_tree)
        source = ast.unparse(tree)
        assert "def save_ptx(" in source

    def test_injects_record_input_output(self, fwd_tree):
        tree = ModuleInjector("TestOp").visit(fwd_tree)
        source = ast.unparse(tree)
        assert "def record_input_output(" in source

    def test_adds_decorator_to_call(self, fwd_tree):
        tree = ModuleInjector("TestOp").visit(fwd_tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "call":
                decorator_names = [ast.unparse(d) for d in node.decorator_list]
                assert any("record_input_output" in d for d in decorator_names)
                return
        pytest.fail("No 'call' function found")


class TestKernelRunnerHook:
    def test_intercepts_kernel_run(self, fwd_tree):
        tree = ModuleInjector("TestOp").visit(fwd_tree)
        tree = KernelRunnerHook().visit(tree)
        ast.fix_missing_locations(tree)
        source = ast.unparse(tree)
        assert "save_ptx(" in source
        # Original kernel.run() should be captured as assignment
        assert "_kernel = " in source

    def test_extracts_grid_constants(self, fwd_tree):
        tree = ModuleInjector("TestOp").visit(fwd_tree)
        tree = KernelRunnerHook().visit(tree)
        ast.fix_missing_locations(tree)
        source = ast.unparse(tree)
        assert "XGRID" in source
        assert "YGRID" in source
        assert "ZGRID" in source


class TestMainStripper:
    def test_strips_benchmark_and_main(self, fwd_tree):
        tree = MainStripper().visit(fwd_tree)
        source = ast.unparse(tree)
        # print_performance call removed, compiled_module_main removed
        assert "return print_performance(" not in source
        assert "compiled_module_main" not in source
        # benchmark_compiled_module still exists (body rewritten)
        assert "benchmark_compiled_module" in source
