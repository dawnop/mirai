"""
Static-shape PFFN example — generates TF ops with fixed input shapes.

The generated kernels have all shapes baked in as compile-time constants,
which allows maximum kernel optimization but requires recompilation if
input shapes change. For variable batch sizes, see pffn_dynamic.py.

Generated artifacts (in ``./generated/``):
  PffnFwd.cc / PffnBwd.cc   — TF custom-op C++ sources
  pffn_api.py                — TF Python wrapper with tf.custom_gradient
  build.sh                   — one-step compilation script
  tf32/                      — PTX files and metadata

Usage:
  python examples/pffn.py --ptxas /usr/local/cuda/bin/ptxas
  cd generated && bash build.sh
"""

import torch
import torch.nn.functional as F
import mirai


@mirai.op(name="Pffn")
def pffn(inputs, w_gate, b_gate, w_up, b_up, w_down, b_down):
    """
    Per-Token SwiGLU (Swish-Gated Linear Unit).

    inputs: [batch, num_tokens, dim]
    w_gate / w_up:   [num_tokens, dim, inner_dim]
    b_gate / b_up:   [num_tokens, inner_dim]
    w_down:          [num_tokens, inner_dim, output_dim]
    b_down:          [num_tokens, output_dim]
    """
    inputs_t = inputs.transpose(0, 1).contiguous()  # TF side cannot represent non-contiguous tensors

    gates = torch.bmm(inputs_t, w_gate) + b_gate.unsqueeze(1)
    gates = F.silu(gates)

    vals = torch.bmm(inputs_t, w_up) + b_up.unsqueeze(1)

    outputs = gates * vals
    outputs = torch.bmm(outputs, w_down) + b_down.unsqueeze(1)

    outputs = outputs.transpose(0, 1)
    return outputs


def _make_sample_inputs(bs, num_tokens, dim, inner_dim, output_dim, device):
    """Create random sample inputs for tracing."""
    inputs = torch.randn(bs, num_tokens, dim, device=device, requires_grad=True)
    w_gate = torch.randn(num_tokens, dim, inner_dim, device=device, requires_grad=True)
    b_gate = torch.randn(num_tokens, inner_dim, device=device, requires_grad=True)
    w_up = torch.randn(num_tokens, dim, inner_dim, device=device, requires_grad=True)
    b_up = torch.randn(num_tokens, inner_dim, device=device, requires_grad=True)
    w_down = torch.randn(num_tokens, inner_dim, output_dim, device=device, requires_grad=True)
    b_down = torch.randn(num_tokens, output_dim, device=device, requires_grad=True)
    return [inputs, w_gate, b_gate, w_up, b_up, w_down, b_down]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate static-shape PFFN TF custom ops via Mirai")
    parser.add_argument("--ptxas", type=str, default=None, help="Path to ptxas binary")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    bs, num_tokens, dim = 5000, 32, 512
    inner_dim = dim * 3
    output_dim = 512

    sample_inputs = _make_sample_inputs(bs, num_tokens, dim, inner_dim, output_dim, device)

    # One-click build: generates all artifacts in ./generated/
    mirai.build(pffn, sample_inputs=sample_inputs, ptxas=args.ptxas)

    # Next step: cd generated && bash build.sh


if __name__ == "__main__":
    main()
