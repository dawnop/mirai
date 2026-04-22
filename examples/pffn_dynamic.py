"""
Dynamic-shape PFFN example — generates TF ops that accept variable batch sizes.

Key differences from the static example (pffn.py):
  - ``dynamic=True`` tells torch.compile to trace with symbolic shapes, so the
    generated Triton kernels use runtime-computed grid sizes and xnumel values
    instead of hard-coded constants.
  - The C++ TF op reads shape symbols (s0, s1, ...) from ``input.dim_size()``
    at kernel launch time, so a *single* compiled op handles any batch size.
  - Extra ``*_shapes.json`` files are emitted alongside each ``.cc`` to record
    the symbol-to-(tensor, dim) mapping.

Generated artifacts (in ``./generated/``):
  PffnFwd.cc / PffnBwd.cc   — TF custom-op C++ sources (shape-generic)
  PffnFwd_shapes.json / ...  — symbol dimension maps
  pffn_api.py                — TF Python wrapper with tf.custom_gradient
  build.sh                   — one-step compilation script
  tf32/                      — PTX files and metadata

TF-side usage:
  # The same op binary works for any batch size:
  out = pffn_op(inputs_2000, ...)   # bs=2000
  out = pffn_op(inputs_8000, ...)   # bs=8000
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
    """Create simple random inputs for tracing (no custom init needed)."""
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

    parser = argparse.ArgumentParser(description="Generate dynamic-shape PFFN TF custom ops via Mirai")
    parser.add_argument("--ptxas", type=str, default=None, help="Path to ptxas binary")
    parser.add_argument("--bs", type=int, default=5000, help="Sample batch size for tracing (default: 5000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    num_tokens, dim = 32, 512
    inner_dim = dim * 3
    output_dim = 512

    sample_inputs = _make_sample_inputs(args.bs, num_tokens, dim, inner_dim, output_dim, device)

    mirai.build(
        pffn,
        sample_inputs=sample_inputs,
        output_dir="./generated",
        ptxas=args.ptxas,
        dynamic=True,
    )


if __name__ == "__main__":
    main()
