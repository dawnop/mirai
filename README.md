# Mirai

A compiler that converts PyTorch models into deployable TensorFlow custom ops backed by Triton PTX kernels.

Mirai takes a decorated PyTorch function, runs `torch.compile` to generate optimized Triton kernels, extracts the PTX, and wraps everything into a TensorFlow custom op — with a single function call.

## Quick Start

```python
import torch
import mirai

@mirai.op(name="Pffn")
def pffn(inputs, w_gate, b_gate, w_up, b_up, w_down, b_down):
    inputs_t = inputs.transpose(0, 1).contiguous()
    gates = torch.bmm(inputs_t, w_gate) + b_gate.unsqueeze(1)
    gates = torch.nn.functional.silu(gates)
    vals = torch.bmm(inputs_t, w_up) + b_up.unsqueeze(1)
    outputs = torch.bmm(gates * vals, w_down) + b_down.unsqueeze(1)
    return outputs.transpose(0, 1).contiguous()

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

## Requirements

The pipeline spans two environments (typically on different machines):

| Stage | Environment | Dependencies |
|---|---|---|
| PTX generation (Stage 1–3) | PyTorch 2.x + Triton + CUDA GPU with `ptxas` | `torch>=2.0`, `jinja2>=3.0`, Python >= 3.8 |
| TF op compilation (build.sh) | TensorFlow + CUDA toolkit + g++ | `tensorflow`, CUDA headers |

`mirai.build()` / `python -m mirai` runs in the PyTorch environment, producing `generated/` artifacts. Copy them to the TF environment and run `bash build.sh` to compile into `.so`.

### Install

```bash
pip install -e .
```

## How It Works

```
@mirai.op decorated function
        │
        ▼
  torch.compile + execute        ← Stage 1: generate Triton kernels
        │
        ▼
  discover output_code.py        ← Stage 2: locate generated code
        │
        ▼
  AST patch → run → save PTX     ← Stage 3: extract PTX and shapes
        │
        ▼
  render C++ TF op               ← Stage 3: Jinja2 template → .cc
        │
        ▼
  generate build.sh              ← Stage 4
        │
        ▼
  generate TF API wrapper        ← Stage 5 (if backward exists)
```

**Stage 1** uses `torch.compile` with `max_autotune` to find optimal kernel configurations. AUTOTUNE progress is shown during compilation.

**Stage 3** patches inductor-generated code via AST transformers to intercept kernel launches and dump PTX binaries. The patched script runs in a subprocess with compile debug disabled.

**Stage 5** generates a Python API wrapper that connects forward outputs to backward inputs, mapping user parameter names to internal tensor names.

## CLI

For pre-existing `output_code.py` files (skips Stage 1):

```bash
python -m mirai \
  --model-name Pffn \
  --fwd-path /path/to/fwd_output_code.py \
  --bwd-path /path/to/bwd_output_code.py \
  --target ./generated \
  --version tf32
```

## Logging

All output uses the `mirai` logger with `[MIRAI INFO]` prefix:

```python
from mirai.log import setup_default_logging
import logging

setup_default_logging(logging.DEBUG)   # show everything including torch internals
setup_default_logging(logging.INFO)    # default: MIRAI logs + AUTOTUNE tables
setup_default_logging(logging.WARNING) # quiet mode
```

## Project Structure

```
mirai/
├── __init__.py        # Public API: op, module, build
├── log.py             # Unified logging
├── decorator.py       # @mirai.op — contiguous IO wrapper
├── build.py           # mirai.build() entry point
├── __main__.py        # CLI entry point (python -m mirai)
├── discovery.py       # Auto-discover output_code.py
├── pipeline.py        # Per-kernel processing pipeline
├── codegen/
│   ├── transformer.py # AST transformers (ModuleInjector, KernelRunnerHook, MainStripper)
│   ├── render.py      # AST → C++ TF op via Jinja2
│   ├── builder.py     # Render build.sh
│   └── api_render.py  # Render TF API wrapper
└── templates/
    ├── kernel.cc.tpl  # C++ TF custom op template
    ├── build.sh.tpl   # Compilation script template
    └── api.py.tpl     # Python API wrapper template
examples/
└── pffn.py            # PFFN SwiGLU end-to-end example
```

## Development

```bash
uv sync --extra dev
uv run black --check .
uv run pytest tests/ --ignore=tests/test_e2e.py   # unit tests (no GPU)
uv run pytest tests/test_e2e.py                    # e2e tests (GPU required)
```
