"""PyTorch solver for discrete-time finite-horizon time-varying LQR (GPU-first).

This module provides:
  - a robust eager-mode path using `torch.linalg.lstsq` (matches JAX behavior),
  - a compile/control-flow friendly path using `torch._higher_order_ops.scan`
    plus `torch.linalg.solve` (static output shapes).
"""

from __future__ import annotations

from typing import Callable, Literal, Tuple

import torch

from torch import _higher_order_ops as _ho

_Solver = Literal["lstsq", "solve"]


def rollout(
    K: torch.Tensor,
    k: torch.Tensor,
    x0: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    use_scan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Rolls-out time-varying linear policy `u[t] = K[t] @ x[t] + k[t]`.

  Shapes:
    K: (T, m, n), k: (T, m), x0: (n,)
    A: (T, n, n), B: (T, n, m), c: (T, n)
  """
  if K.device != x0.device:
    raise ValueError("All inputs must be on the same device.")
  T = K.shape[0]
  A = A[:T]
  B = B[:T]
  c = c[:T]

  n = x0.shape[0]
  m = k.shape[1]
  if use_scan:
    timesteps = torch.arange(T, device=x0.device, dtype=torch.int64)

    def step(x, xs_t):
      K_t, k_t, A_t, B_t, c_t, t = xs_t
      u = K_t @ x + k_t
      x_next = A_t @ x + B_t @ u + c_t
      # Avoid aliasing between carry and outputs.
      y = torch.cat([x_next, u], dim=0) + 0
      return x_next, y

    xs = (K, k, A, B, c, timesteps)
    _, y = _ho.scan(step, x0, xs)
    X = torch.cat([x0.unsqueeze(0), y[:, :n]], dim=0)
    U = y[:, n:].contiguous()
    return X, U

  X = torch.empty((T + 1, n), device=x0.device, dtype=x0.dtype)
  U = torch.empty((T, m), device=x0.device, dtype=x0.dtype)
  X[0] = x0
  for t in range(T):
    u = K[t] @ X[t] + k[t]
    X[t + 1] = A[t] @ X[t] + B[t] @ u + c[t]
    U[t] = u
  return X, U


def lqr_step(
    P: torch.Tensor,
    p: torch.Tensor,
    Q: torch.Tensor,
    q: torch.Tensor,
    R: torch.Tensor,
    r: torch.Tensor,
    M: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    delta: float = 1e-8,
    solver: _Solver = "lstsq",
    I_m: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Single LQR Riccati step (backward recursion)."""
  sym = lambda X: 0.5 * (X + X.transpose(-1, -2))

  AtP = A.transpose(-1, -2) @ P
  AtPA = sym(AtP @ A)
  BtP = B.transpose(-1, -2) @ P
  BtPA = BtP @ A

  H = BtPA + M.transpose(-1, -2)
  h = B.transpose(-1, -2) @ p + BtP @ c + r

  G = sym(R + BtP @ B)
  rhs = -torch.cat([H, h.unsqueeze(-1)], dim=-1)  # (m, n+1)
  if solver == "lstsq":
    # Match JAX behavior (`np.linalg.lstsq(..., rcond=delta)`), which is robust
    # even when `G` is poorly conditioned. Note: `torch.compile` cannot compile
    # `lstsq` due to dynamic output shapes.
    K_k = torch.linalg.lstsq(G, rhs, rcond=delta).solution
  elif solver == "solve":
    if I_m is None:
      I_m = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    G_damped = G + float(delta) * I_m
    K_k = torch.linalg.solve(G_damped, rhs)
  else:
    raise ValueError(f"Unknown solver: {solver}")
  # Avoid aliasing between K and k views under higher-order ops capture.
  K = K_k[..., :-1].contiguous()
  k = K_k[..., -1].contiguous()

  H_GK = H + G @ K
  P_new = sym(Q + AtPA + H_GK.transpose(-1, -2) @ K + K.transpose(-1, -2) @ H)
  p_new = q + A.transpose(-1, -2) @ p + AtP @ c + H_GK.transpose(
      -1, -2) @ k + K.transpose(-1, -2) @ h
  return P_new, p_new, K, k


def tvlqr_inplace(
    Q: torch.Tensor,
    q: torch.Tensor,
    R: torch.Tensor,
    r: torch.Tensor,
    M: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    K_out: torch.Tensor,
    k_out: torch.Tensor,
    P_out: torch.Tensor,
    p_out: torch.Tensor,
    delta: float = 1e-8,
    I_m: torch.Tensor | None = None,
    solver: _Solver = "lstsq",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """In-place TVLQR variant for CUDA graph capture.

  All tensors must be on the same CUDA device with fixed shapes. For best CUDA
  graph capture compatibility, pass a preallocated `I_m = eye(m)` tensor.
  """
  T = Q.shape[0] - 1
  R = R[:T]
  r = r[:T]
  M = M[:T]
  A = A[:T]
  B = B[:T]
  c = c[:T]
  m = R.shape[1]

  if I_m is None:
    I_m = torch.eye(m, device=Q.device, dtype=Q.dtype)

  def sym(X):
    return 0.5 * (X + X.transpose(-1, -2))

  P_out[T].copy_(Q[T])
  p_out[T].copy_(q[T])

  for t in range(T - 1, -1, -1):
    P_next = P_out[t + 1]
    p_next = p_out[t + 1]

    AtP = A[t].transpose(-1, -2) @ P_next
    AtPA = sym(AtP @ A[t])
    BtP = B[t].transpose(-1, -2) @ P_next
    BtPA = BtP @ A[t]

    H = BtPA + M[t].transpose(-1, -2)
    h = B[t].transpose(-1, -2) @ p_next + BtP @ c[t] + r[t]

    G = sym(R[t] + BtP @ B[t])
    rhs = -torch.cat([H, h.unsqueeze(-1)], dim=-1)
    if solver == "lstsq":
      K_k = torch.linalg.lstsq(G, rhs, rcond=delta).solution
    elif solver == "solve":
      K_k = torch.linalg.solve(G + float(delta) * I_m, rhs)
    else:
      raise ValueError(f"Unknown solver: {solver}")
    K_t = K_k[..., :-1].contiguous()
    k_t = K_k[..., -1].contiguous()

    H_GK = H + G @ K_t
    P_t = sym(Q[t] + AtPA + H_GK.transpose(-1, -2) @ K_t +
              K_t.transpose(-1, -2) @ H)
    p_t = q[t] + A[t].transpose(-1, -2) @ p_next + AtP @ c[t] + H_GK.transpose(
        -1, -2) @ k_t + K_t.transpose(-1, -2) @ h

    P_out[t].copy_(P_t)
    p_out[t].copy_(p_t)
    K_out[t].copy_(K_t)
    k_out[t].copy_(k_t)

  return K_out, k_out, P_out, p_out


def rollout_inplace(
    K: torch.Tensor,
    k: torch.Tensor,
    x0: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    X_out: torch.Tensor,
    U_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """In-place rollout for CUDA graph capture."""
  T = K.shape[0]
  A = A[:T]
  B = B[:T]
  c = c[:T]

  X_out[0].copy_(x0)
  for t in range(T):
    u = K[t] @ X_out[t] + k[t]
    X_out[t + 1].copy_(A[t] @ X_out[t] + B[t] @ u + c[t])
    U_out[t].copy_(u)
  return X_out, U_out


def tvlqr(
    Q: torch.Tensor,
    q: torch.Tensor,
    R: torch.Tensor,
    r: torch.Tensor,
    M: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    delta: float = 1e-8,
    solver: _Solver = "lstsq",
    use_scan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Discrete-time finite-horizon time-varying LQR.

  Shapes:
    Q: (T+1, n, n), q: (T+1, n)
    R: (T, m, m), r: (T, m), M: (T, n, m)
    A: (T, n, n), B: (T, n, m), c: (T, n)
  """
  T = Q.shape[0] - 1
  R = R[:T]
  r = r[:T]
  M = M[:T]
  A = A[:T]
  B = B[:T]
  c = c[:T]
  n = Q.shape[1]
  m = R.shape[1]
  if not use_scan:
    P = torch.empty((T + 1, n, n), device=Q.device, dtype=Q.dtype)
    p = torch.empty((T + 1, n), device=Q.device, dtype=Q.dtype)
    K = torch.empty((T, m, n), device=Q.device, dtype=Q.dtype)
    k = torch.empty((T, m), device=Q.device, dtype=Q.dtype)

    P[T] = Q[T]
    p[T] = q[T]
    I_m = torch.eye(m, device=Q.device, dtype=Q.dtype) if solver == "solve" else None
    for t in range(T - 1, -1, -1):
      P_t, p_t, K_t, k_t = lqr_step(P[t + 1],
                                   p[t + 1],
                                   Q[t],
                                   q[t],
                                   R[t],
                                   r[t],
                                   M[t],
                                   A[t],
                                   B[t],
                                   c[t],
                                   delta=delta,
                                   solver=solver,
                                   I_m=I_m)
      P[t] = P_t
      p[t] = p_t
      K[t] = K_t
      k[t] = k_t

    return K, k, P, p

  if solver != "solve":
    raise ValueError("`use_scan=True` requires `solver='solve'` (static shapes).")

  I_m = torch.eye(m, device=Q.device, dtype=Q.dtype)

  def combine(carry, xs_t):
    P_next, p_next = carry
    Q_t, q_t, R_t, r_t, M_t, A_t, B_t, c_t = xs_t
    P_t, p_t, K_t, k_t = lqr_step(P_next,
                                 p_next,
                                 Q_t,
                                 q_t,
                                 R_t,
                                 r_t,
                                 M_t,
                                 A_t,
                                 B_t,
                                 c_t,
                                 delta=delta,
                                 solver="solve",
                                 I_m=I_m)
    # Avoid HOP aliasing: outputs must not alias carry.
    out = (P_t + 0, p_t + 0, K_t, k_t)
    return (P_t, p_t), out

  # Avoid input aliasing: `scan` capture forbids aliasing between inputs.
  init = (Q[T].clone(), q[T].clone())
  Q_stage = Q[:T].contiguous()
  q_stage = q[:T].contiguous()
  xs = (Q_stage, q_stage, R.contiguous(), r.contiguous(), M.contiguous(),
        A.contiguous(), B.contiguous(), c.contiguous())
  (_, _), (P_seq, p_seq, K_seq, k_seq) = _ho.scan(combine, init, xs, reverse=True)

  P = torch.cat([P_seq, Q[T].unsqueeze(0)], dim=0)
  p = torch.cat([p_seq, q[T].unsqueeze(0)], dim=0)
  return K_seq, k_seq, P, p


def ctvlqr(
    projector: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                            torch.Tensor]],
    Q: torch.Tensor,
    q: torch.Tensor,
    R: torch.Tensor,
    r: torch.Tensor,
    M: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    x0: torch.Tensor,
    rho: float = 1.0,
    maxiter: int = 100,
    delta: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Constrained TVLQR via ADMM (GPU-first)."""
  T = R.shape[0]
  n = Q.shape[1]
  m = R.shape[1]

  X = torch.zeros((T + 1, n), device=x0.device, dtype=x0.dtype)
  U = torch.zeros((T, m), device=x0.device, dtype=x0.dtype)
  VX = torch.zeros_like(X)
  VU = torch.zeros_like(U)
  ZX = torch.zeros_like(X)
  ZU = torch.zeros_like(U)
  I_m = torch.eye(m, device=x0.device, dtype=x0.dtype).expand(T, m, m)
  I_n = torch.eye(n, device=x0.device, dtype=x0.dtype).expand(T + 1, n, n)

  for _ in range(maxiter):
    K, k, _, _ = tvlqr(
        Q + rho * I_n,
        q - (ZX - VX),
        R + rho * I_m,
        r - (ZU - VU),
        M,
        A,
        B,
        c,
        delta=delta,
    )
    X, U = rollout(K, k, x0, A, B, c)
    ZX, ZU = projector(X + VX, U + VU)
    VX = VX + X - ZX
    VU = VU + U - ZU

  return X, U, K, k
