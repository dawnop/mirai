"""Tests for mirai.decorator — @mirai.op decorator."""

import torch
import mirai


class TestMiraiOp:
    def test_attaches_op_name(self):
        @mirai.op(name="Foo")
        def my_func(x):
            return x

        assert my_func._mirai_op_name == "Foo"

    def test_preserves_original_func(self):
        def my_func(x):
            return x

        wrapped = mirai.op(name="Bar")(my_func)
        assert wrapped._mirai_original_func is my_func

    def test_contiguous_output(self):
        @mirai.op(name="Contig")
        def make_non_contig(x):
            return x.t()  # transpose makes it non-contiguous

        x = torch.randn(3, 4)
        out = make_non_contig(x)
        assert out.is_contiguous()

    def test_module_alias(self):
        @mirai.module
        def my_func(x):
            return x + 1

        x = torch.randn(3)
        out = my_func(x)
        assert torch.allclose(out, x + 1)
