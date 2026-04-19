"""Tests for mirai.codegen.render — AST visitors."""

import ast
from pathlib import Path

import pytest

from mirai.codegen.render import AsyncTritonAssignVisitor, TensorIOVisitor

FIXTURES = Path(__file__).parent / "fixtures"
FWD_CODE = FIXTURES / "fwd_output_code.py"


@pytest.fixture
def fwd_tree():
    return ast.parse(FWD_CODE.read_text())


class TestAsyncTritonAssignVisitor:
    def test_finds_kernel_names(self, fwd_tree):
        visitor = AsyncTritonAssignVisitor()
        visitor.visit(fwd_tree)
        assert len(visitor.kernels) > 0
        for name in visitor.kernels:
            assert name.startswith("triton_")


class TestTensorIOVisitor:
    def test_finds_inputs(self, fwd_tree):
        visitor = TensorIOVisitor()
        visitor.visit(fwd_tree)
        assert len(visitor.input_names) > 0
        assert any("primals" in n for n in visitor.input_names)

    def test_finds_outputs(self, fwd_tree):
        visitor = TensorIOVisitor()
        visitor.visit(fwd_tree)
        assert len(visitor.output_names) > 0
