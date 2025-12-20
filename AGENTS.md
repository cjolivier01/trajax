# AGENTS

This repository now contains both the original JAX-based Trajax code and a PyTorch implementation under `trajax/torch`.

Guidance for automated agents working on this repo:

- Prefer JAX paths for existing APIs; PyTorch equivalents live in `trajax/torch`.
- Keep tensor operations device-resident: avoid `.cpu()`/NumPy round-trips in the PyTorch code unless explicitly testing cross-framework equivalence.
- When adding tests, ensure they run via `PYTHONPATH=.` so the local package is importable without installation.
- Avoid destructive git actions (`reset --hard`, etc.); the working tree may contain user changes.
- Default to float32 in new PyTorch tests to keep functorch happy; widen precision only if necessary.
