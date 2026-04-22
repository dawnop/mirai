<p align="center">
  <img src="assets/logo.svg" alt="Mirai" width="320">
</p>
<p align="center">
  A compiler that converts PyTorch models into deployable TensorFlow custom ops backed by Triton PTX kernels.
</p>

Mirai takes a decorated PyTorch function, runs `torch.compile` to generate optimized Triton kernels, extracts the PTX, and wraps everything into a TensorFlow custom op — with a single function call.

## Quick Start

```python
import torch
import mirai

@mirai.op(name="Pffn")
def pffn(inputs, w_gate, b_gate, w_up, b_up, w_down, b_down):
    inputs_t = inputs.transpose(0, 1).contiguous()  # TF side cannot represent non-contiguous tensors
    gates = torch.bmm(inputs_t, w_gate) + b_gate.unsqueeze(1)
    gates = torch.nn.functional.silu(gates)
    vals = torch.bmm(inputs_t, w_up) + b_up.unsqueeze(1)
    outputs = torch.bmm(gates * vals, w_down) + b_down.unsqueeze(1)
    return outputs.transpose(0, 1)

# One call: torch.compile → PTX extraction → C++ codegen → build script
mirai.build(pffn, sample_inputs=[inputs, w_gate, b_gate, w_up, b_up, w_down, b_down])
```

Output in `./generated/`:
```
generated/
├── PffnFwd.cc          # TF custom op C++ source (forward)
├── PffnBwd.cc          # TF custom op C++ source (backward)
├── build.sh            # g++ compilation script
├── pffn_api.py         # TF Python API wrapper
└── tf32/               # PTX kernels and metadata
    ├── PffnFwd/
    └── PffnBwd/
```

Build the op:
```bash
cd generated && bash build.sh
```

### Dynamic Shapes

Use `dynamic=True` to generate shape-generic ops that accept variable batch sizes at runtime:

```python
mirai.build(pffn, sample_inputs=sample_inputs, dynamic=True)
```

A single compiled op then handles any batch size — no recompilation needed:

```python
# TF side: same op binary, different batch sizes
out = pffn_op(inputs_2000, ...)   # bs=2000
out = pffn_op(inputs_8000, ...)   # bs=8000
```

See [`examples/pffn_dynamic.py`](examples/pffn_dynamic.py) for a complete example.

## Requirements

The pipeline spans two environments (typically on different machines):

| Stage | Dependencies |
|---|---|
| Code generation (`mirai.build`) | PyTorch 2.x, CUDA|
| Op compilation (`build.sh`) | TensorFlow 1.x, CUDA|

Run `mirai.build()` in the PyTorch environment, copy `generated/` to the TF environment, then `bash build.sh`.

### Install

```bash
pip install mirai-compiler
```

## How It Works

Mirai bridges two frameworks at the PTX level — it leverages PyTorch's compiler stack to produce hardware-optimized GPU kernels, then wraps them as native TensorFlow ops.

```
                        PyTorch environment                          TF environment
               ┌─────────────────────────────────────────┐    ┌──────────────────────┐
 @mirai.op  →  │ torch.compile  →  Triton  →  PTX/CUBIN  │ →  │  C++ TF custom op    │
   function    │    (max_autotune, fwd + bwd)            │    │  (.so, Python API)   │
               └─────────────────────────────────────────┘    └──────────────────────┘
```

**Under the hood:**

1. **Trace & Optimize** — `torch.compile` with `max_autotune` explores kernel configurations across the search space. Both forward and backward graphs are traced and optimized independently.

2. **Intercept & Extract** — Mirai rewrites the inductor-generated Python via AST transformers, injecting hooks that capture PTX binaries and tensor metadata at each kernel launch site. The patched code runs in an isolated subprocess.

3. **Codegen** — Each captured kernel is rendered into a self-contained C++ TF op via Jinja2 templates, complete with shape inference, PTX loading, and CUDA launch logic. A Python API wrapper with `tf.custom_gradient` connects forward and backward ops seamlessly.

## Project Structure

```
mirai/
├── build.py           # mirai.build() entry point
├── decorator.py       # @mirai.op decorator
├── pipeline.py        # Per-kernel: AST patch → PTX extraction → C++ render
├── codegen/           # AST transformers, C++ / build.sh / API renderers
└── templates/         # Jinja2 templates (kernel.cc, build.sh, api.py)
examples/
├── pffn.py            # Static shape example
└── pffn_dynamic.py    # Dynamic shape example
```

## Development

```bash
pip install -e ".[dev]"
black --check mirai/ tests/ examples/
pytest tests/ -v
```
