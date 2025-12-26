"""PyTorch solver for discrete-time finite-horizon time-varying LQR."""

import torch


def rollout(K, k, x0, A, B, c):
  """Rolls out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
  T, m, n = K.shape
  device, dtype = x0.device, x0.dtype
  X = torch.zeros((T + 1, n), device=device, dtype=dtype)
  U = torch.zeros((T, m), device=device, dtype=dtype)
  X[0] = x0
  for t in range(T):
    u = torch.matmul(K[t], X[t]) + k[t]
    x = torch.matmul(A[t], X[t]) + torch.matmul(B[t], u) + c[t]
    X[t + 1] = x
    U[t] = u
  return X, U


def _symmetrize(x):
  return 0.5 * (x + x.transpose(-1, -2))


def lqr_step(P, p, Q, q, R, r, M, A, B, c, delta=1e-8):
  """Single LQR step."""
  AtP = torch.matmul(A.transpose(-1, -2), P)
  AtPA = _symmetrize(torch.matmul(AtP, A))
  BtP = torch.matmul(B.transpose(-1, -2), P)
  BtPA = torch.matmul(BtP, A)

  H = BtPA + M.transpose(-1, -2)
  h = torch.matmul(B.transpose(-1, -2), p) + torch.matmul(BtP, c) + r

  G = _symmetrize(R + torch.matmul(BtP, B))

  stacked = -torch.cat((H, h.reshape(h.shape[0], 1)), dim=1)
  sol = torch.linalg.lstsq(G + delta * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype).to(torch.float32), stacked.to(torch.float32)).solution
  sol = sol.to(G.dtype)
  K = sol[:, :-1]
  k = sol[:, -1]

  H_GK = H + torch.matmul(G, K)
  P_new = _symmetrize(Q + AtPA + torch.matmul(H_GK.transpose(-1, -2), K) +
                      torch.matmul(K.transpose(-1, -2), H))
  p_new = q + torch.matmul(A.transpose(-1, -2), p) + torch.matmul(AtP, c)
  p_new = p_new + torch.matmul(H_GK.transpose(-1, -2), k) + torch.matmul(K.transpose(-1, -2), h)

  return P_new, p_new, K, k


def tvlqr(Q, q, R, r, M, A, B, c):
  """Discrete-time finite-horizon time-varying LQR."""
  T = Q.shape[0] - 1
  m = R.shape[1]
  n = Q.shape[1]
  device, dtype = Q.device, Q.dtype

  P = torch.zeros((T + 1, n, n), device=device, dtype=dtype)
  p = torch.zeros((T + 1, n), device=device, dtype=dtype)
  K = torch.zeros((T, m, n), device=device, dtype=dtype)
  k = torch.zeros((T, m), device=device, dtype=dtype)

  P[-1] = Q[T]
  p[-1] = q[T]

  for tt in range(T - 1, -1, -1):
    P_t, p_t, K_t, k_t = lqr_step(P[tt + 1], p[tt + 1], Q[tt], q[tt], R[tt],
                                  r[tt], M[tt], A[tt], B[tt], c[tt])
    P[tt] = P_t
    p[tt] = p_t
    K[tt] = K_t
    k[tt] = k_t

  return K, k, P, p


def ctvlqr(projector, Q, q, R, r, M, A, B, c, x0, rho=1.0, maxiter=100):
  """Constrained discrete-time finite-horizon time-varying LQR."""
  T, m, _ = R.shape
  n = Q.shape[1]
  device, dtype = Q.device, Q.dtype

  X = torch.zeros((T + 1, n), device=device, dtype=dtype)
  U = torch.zeros((T, m), device=device, dtype=dtype)
  VX = torch.zeros_like(X)
  VU = torch.zeros_like(U)
  ZX = torch.zeros_like(X)
  ZU = torch.zeros_like(U)
  K = torch.zeros((T, m, n), device=device, dtype=dtype)
  k = torch.zeros((T, m), device=device, dtype=dtype)
  Im = torch.eye(m, device=device, dtype=dtype).expand(T, m, m)
  In = torch.eye(n, device=device, dtype=dtype).expand(T + 1, n, n)

  for _ in range(maxiter):
    K, k, _, _ = tvlqr(Q + rho * In, q - (ZX - VX), R + rho * Im,
                       r - (ZU - VU), M, A, B, c)
    X, U = rollout(K, k, x0, A, B, c)
    ZX, ZU = projector(X + VX, U + VU)
    VX = VX + X - ZX
    VU = VU + U - ZU

  return X, U, K, k
