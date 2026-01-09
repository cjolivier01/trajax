# AGENTS.md (Trajax repo)

This file provides operational guidance for automated coding agents and human contributors working in this repository.

## Scope

These instructions apply to the entire repository rooted at this `AGENTS.md`.

## Repo Overview

- `trajax/`: JAX reference implementation of Trajax (original backend).
- `trajax/torch/`: GPU-first PyTorch port of the core APIs (no NumPy/SciPy/JAX in the Torch compute path).
- `tests/`: unit tests + JAX↔Torch parity tests.
- `benchmarks/`: micro-benchmarks comparing JAX vs Torch on GPU.

Primary goals of the Torch backend:

- Keep *all* compute on CUDA tensors (no `.cpu()`, no NumPy/SciPy, no JAX calls).
- Match the JAX API and numerics closely enough for parity tests.
- Provide a path compatible with CUDA Graph capture (fixed shapes; avoid hidden allocations).

## Environment and Installation

The project is a Python package; editable installs are recommended for development.

- Base install: `pip install -e .`
- Test extras (constraint solver dependencies): `pip install -e '.[sqp]'`
- Torch extras (Torch backend deps): `pip install -e '.[torch]'`

Notes:

- Benchmarks require *both* PyTorch CUDA and JAX CUDA to be available.
- `tests/conftest.py` forces JAX to CPU by default to keep pytest fast and avoid long GPU compilation; this does not affect benchmarks.

## Coding Conventions

- Follow existing style in the touched file; the codebase largely follows Google-style Python (2-space indents are common).
- Keep diffs focused; do not refactor unrelated code.
- Avoid adding new dependencies unless required; if you do, update `setup.py` extras and/or `requirements_test.txt` accordingly.

## Torch Backend Rules (Important)

- The Torch backend lives under `trajax/torch/` and must not depend on JAX/NumPy/SciPy for the compute path.
- Assume CUDA tensors:
  - Torch functions validate CUDA by default and will raise if given CPU tensors.
  - You can override for debugging with `TRAJAX_TORCH_ALLOW_CPU=1`, but do not rely on it in core logic.
- Avoid host/device sync:
  - Do not call `.item()` or `print(tensor)` inside hot paths.
  - Do not use Python control flow dependent on tensor values unless you intentionally want a sync.
- CUDA Graph friendliness:
  - Prefer in-place variants where provided, and avoid per-call allocations in capture-critical paths.
  - Torch TVLQR exposes `trajax.torch.tvlqr.tvlqr_inplace` and `trajax.torch.tvlqr.rollout_inplace` for writing into preallocated buffers.

## Tests

Run the full test suite:

- `pytest -q`

Key test files:

- `tests/torch_parity_test.py`: parity checks between JAX and Torch for key routines.

If you add/modify Torch functionality that should match JAX, add or extend parity tests rather than only adding Torch-only tests.

## Benchmarks (JAX vs Torch on GPU)

The benchmark driver is:

- `benchmarks/compare_backends.py`

It times JAX (`jax.jit`) and Torch (CUDA events) on GPU and prints:

- input element count (where applicable)
- ms/iter for JAX and Torch
- JAX/Torch speedup ratio

Examples:

- `python benchmarks/compare_backends.py tvlqr_solve --T 100 --n 20 --m 30`
- `python benchmarks/compare_backends.py tvlqr_rollout --T 100 --n 20 --m 30`
- `python benchmarks/compare_backends.py ilqr_linear --T 200 --n 32 --m 32 --maxiter 25`
- `python benchmarks/compare_backends.py constrained_ilqr_linear --T 200 --n 64 --m 96 --maxiter-al 1 --maxiter-ilqr 3`

### Large “big” preset

For reproducible large runs, most commands accept:

- `--preset big`

Current `big` preset choices (see `benchmarks/compare_backends.py`):

- `tvlqr_solve` / `tvlqr_rollout`: `T=625, n=80, m=120` (≈100× more input elements vs defaults)
- `ilqr_linear`: `T=625, n=80, m=120` with reduced iterations (`maxiter<=10`, `warmup<=1`, `iters<=3`)
- `constrained_ilqr_linear`: `T=200, n=64, m=96` with reduced iterations (`maxiter-al<=1`, `maxiter-ilqr<=3`)

Notes:

- `--torch-compile` is intentionally disabled for iLQR / constrained iLQR benchmarks (nested `torch.func` transforms can be unsupported/slow depending on PyTorch version).
- Benchmark results are highly sensitive to dtype and TF32 settings; for `float32`, the benchmark sets `torch.set_float32_matmul_precision("high")`.

## When Changing Numerical Kernels (Checklist)

Before finishing a change that affects math kernels or solver behavior:

- Update JAX and Torch backends consistently (if the change is intended to affect both).
- Add/extend `tests/torch_parity_test.py` to cover the behavior.
- Run `pytest -q`.
- Run the relevant GPU benchmark command(s) in `benchmarks/compare_backends.py` to confirm performance didn’t regress.

## Performance Pitfalls (Torch)

If Torch is unexpectedly slow compared to JAX:

- Watch for Python overhead in solver loops (e.g., per-time-step Python control flow).
- Heavy use of `torch.func.jacrev/hessian` can be expensive; prefer batched/fused formulations when possible.
- Avoid creating many small tensors; prefer vectorized operations over per-step ops.
- Ensure tensors are on CUDA and contiguous where possible (`.contiguous()` only when needed).

If you introduce CUDA extensions or custom kernels:

- Prefer minimal surface area and clear build instructions.
- Keep portability in mind; document build steps and add a small correctness test.
