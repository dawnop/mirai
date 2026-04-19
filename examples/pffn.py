import torch
import torch.nn.functional as F
import torch.nn.init as init
import mirai


@mirai.op(name="Pffn")
def pffn(inputs, w_gate, b_gate, w_up, b_up, w_down, b_down):
    """
    Per-Token SwiGLU pure function implementation.
    inputs: [batch, num_tokens, dim]
    """
    # [batch, tokens, dim] -> [tokens, batch, dim]
    inputs_t = inputs.transpose(0, 1).contiguous()

    # Gate branch
    gates = torch.bmm(inputs_t, w_gate)
    gates = gates + b_gate.unsqueeze(1)
    gates = F.silu(gates)  # Swish

    # Up branch
    vals = torch.bmm(inputs_t, w_up)
    vals = vals + b_up.unsqueeze(1)

    # Element-wise mult (SwiGLU)
    outputs = gates * vals

    # Down branch
    outputs = torch.bmm(outputs, w_down)
    outputs = outputs + b_down.unsqueeze(1)

    # [tokens, batch, out] -> [batch, tokens, out]
    outputs = outputs.transpose(0, 1).contiguous()

    return outputs


# -----------------------------------------------------------------------------
# Weight initialization
# -----------------------------------------------------------------------------
def get_data_with_custom_init(bs, num_tokens, dim, inner_dim, output_dim, device):
    std_gate_up = (2.0 / dim) ** 0.5
    std_down = (1.0 / inner_dim) ** 0.5 * 0.1

    print(f"Init Config:")
    print(f"  > Gate/Up Std : {std_gate_up:.6f}")
    print(f"  > Down Std    : {std_down:.6f}")

    def create_param(shape, std=None, is_bias=False):
        t = torch.empty(shape, device=device)
        with torch.no_grad():
            if is_bias:
                init.zeros_(t)
            else:
                init.normal_(t, mean=0.0, std=std)
        t.requires_grad_(True)
        return t

    inputs = torch.randn(bs, num_tokens, dim, device=device, requires_grad=True)
    w_gate = create_param((num_tokens, dim, inner_dim), std=std_gate_up)
    b_gate = create_param((num_tokens, inner_dim), is_bias=True)
    w_up = create_param((num_tokens, dim, inner_dim), std=std_gate_up)
    b_up = create_param((num_tokens, inner_dim), is_bias=True)
    w_down = create_param((num_tokens, inner_dim, output_dim), std=std_down)
    b_down = create_param((num_tokens, output_dim), is_bias=True)

    return [inputs, w_gate, b_gate, w_up, b_up, w_down, b_down]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", type=str, default=None, help="Path to ptxas binary")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(42)

    # Config
    bs, num_tokens, dim = 5000, 32, 512
    expansion_ratio = 3
    inner_dim = dim * expansion_ratio
    output_dim = 512

    print(f"Config: BS={bs}, Tokens={num_tokens}, Dim={dim}, Inner={inner_dim}")

    # Prepare sample inputs
    sample_inputs = get_data_with_custom_init(bs, num_tokens, dim, inner_dim, output_dim, device)

    # One-click build: generates all artifacts in ./generated/
    mirai.build(pffn, sample_inputs=sample_inputs, ptxas=args.ptxas)

    # Next step: cd generated && bash build.sh


if __name__ == "__main__":
    main()
