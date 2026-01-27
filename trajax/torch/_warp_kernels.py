"""Warp helpers for optional fused CUDA kernels.

This module is intentionally optional: callers must handle the case where Warp
(`warp-lang`) is not installed.
"""

from __future__ import annotations

import torch

try:
  import warp as wp  # type: ignore[import-not-found]
  import warp.torch as wpt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
  wp = None
  wpt = None


def is_available() -> bool:
  return wp is not None and wpt is not None


_initialized: bool = False


def _init_warp():
  global _initialized
  if _initialized:
    return
  assert wp is not None
  # Avoid noisy startup prints in library/test contexts.
  wp.config.quiet = True
  wp.init()
  _initialized = True


if wp is not None:

  @wp.kernel
  def _box_ineq_active(
      T: int,
      m: int,
      U: wp.array(dtype=wp.float32, ndim=2),
      umax: wp.array(dtype=wp.float32, ndim=1),
      dual_ineq: wp.array(dtype=wp.float32, ndim=2),
      out_ineq: wp.array(dtype=wp.float32, ndim=2),
      out_active: wp.array(dtype=wp.bool, ndim=2),
  ):
    idx = wp.tid()
    two_m = 2 * m
    t = idx // two_m
    j = idx - t * two_m

    u = 0.0
    if t < T:
      if j < m:
        u = U[t, j]
        ineq = u - umax[j]
      else:
        u = U[t, j - m]
        ineq = -u - umax[j - m]
    else:
      # t == T: padded control is 0.
      if j < m:
        ineq = -umax[j]
      else:
        ineq = -umax[j - m]

    out_ineq[t, j] = ineq

    d = dual_ineq[t, j]
    inactive = (wp.abs(d) <= 0.0) and (ineq < 0.0)
    out_active[t, j] = not inactive


def box_ineq_active_inplace(
    U: torch.Tensor,
    umax: torch.Tensor,
    dual_ineq: torch.Tensor,
    out_ineq: torch.Tensor,
    out_active: torch.Tensor,
) -> None:
  """Fused kernel for box constraint inequality + active-set mask.

  Fills:
    out_ineq[t] = [u[t]-umax, -u[t]-umax] for t<T, and [-umax, -umax] for t==T
    out_active[t] = ~((dual_ineq[t] == 0) & (out_ineq[t] < 0))

  All tensors must be CUDA and float32 (out_active is bool).
  """
  if not is_available():
    raise RuntimeError("Warp is not available (install `warp-lang`).")
  _init_warp()
  assert wp is not None and wpt is not None

  if U.device.type != "cuda":
    raise ValueError("Warp kernels require CUDA tensors.")
  if U.dtype != torch.float32:
    raise ValueError("Warp kernels currently require float32 tensors.")
  if umax.dtype != torch.float32 or dual_ineq.dtype != torch.float32 or out_ineq.dtype != torch.float32:
    raise ValueError("Warp kernels currently require float32 tensors.")
  if out_active.dtype != torch.bool:
    raise ValueError("`out_active` must be a bool tensor.")

  T = int(U.shape[0])
  m = int(U.shape[1])
  if umax.shape != (m,):
    raise ValueError(f"`umax` must have shape ({m},), got {tuple(umax.shape)}")
  if dual_ineq.shape != (T + 1, 2 * m):
    raise ValueError(
        f"`dual_ineq` must have shape ({T+1}, {2*m}), got {tuple(dual_ineq.shape)}")
  if out_ineq.shape != (T + 1, 2 * m):
    raise ValueError(
        f"`out_ineq` must have shape ({T+1}, {2*m}), got {tuple(out_ineq.shape)}")
  if out_active.shape != (T + 1, 2 * m):
    raise ValueError(
        f"`out_active` must have shape ({T+1}, {2*m}), got {tuple(out_active.shape)}")

  device = f"cuda:{U.device.index}"
  wp.launch(
      _box_ineq_active,
      dim=(T + 1) * 2 * m,
      inputs=[
          T,
          m,
          wpt.from_torch(U.contiguous(), return_ctype=True),
          wpt.from_torch(umax.contiguous(), return_ctype=True),
          wpt.from_torch(dual_ineq.contiguous(), return_ctype=True),
          wpt.from_torch(out_ineq, return_ctype=True),
          wpt.from_torch(out_active, return_ctype=True),
      ],
      device=device,
  )
