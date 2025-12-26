# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
"""PyTorch Solver for Discrete-time Finite Horizon Time-varying LQR.

Solve:

 min_{x0, x1, ...xT, u0, u1...u_{T-1}}

      sum_{t=0}^{T-1} [0.5* x(t)^T Q(t) x(t) + q(t)^T x(t) +
                      0.5* u(t)^T R(t) u(t) + r(t)^T u(t) +
                            +
                      x(t)^T M(t) u(t)]
                            +
                      0.5*x(T)^T Q(T) x(t) + q(T)^T x(T)

      subject to Linear Dynamics:
                      x(t+1) = A(t) x(t) + B(t) u(t) + c(t)

"""
import torch


def rollout(K, k, x0, A, B, c):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

    T, m, n = K.shape
    device = K.device
    dtype = K.dtype

    X = torch.zeros((T + 1, n), device=device, dtype=dtype)
    U = torch.zeros((T, m), device=device, dtype=dtype)
    X[0] = x0

    for t in range(T):
        u = torch.matmul(K[t], X[t]) + k[t]
        x = torch.matmul(A[t], X[t]) + torch.matmul(B[t], u) + c[t]
        X[t + 1] = x
        U[t] = u

    return X, U


def lqr_step(P, p, Q, q, R, r, M, A, B, c, delta=1e-8):
    """Single LQR Step.

    Args:
        P: [n, n] tensor.
        p: [n] tensor.
        Q: [n, n] tensor.
        q: [n] tensor.
        R: [m, m] tensor.
        r: [m] tensor.
        M: [n, m] tensor.
        A: [n, n] tensor.
        B: [n, m] tensor.
        c: [n] tensor.
        delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

    Returns:
        P, p: updated matrices encoding quadratic value function.
        K, k: state feedback gain and affine term.
    """
    symmetrize = lambda x: (x + x.T) / 2

    AtP = torch.matmul(A.T, P)
    AtPA = symmetrize(torch.matmul(AtP, A))
    BtP = torch.matmul(B.T, P)
    BtPA = torch.matmul(BtP, A)

    H = BtPA + M.T
    h = torch.matmul(B.T, p) + torch.matmul(BtP, c) + r

    G = symmetrize(R + torch.matmul(BtP, B))

    # Get damped (Levenberg-Marquardt) inverse using lstsq
    K_k = torch.linalg.lstsq(
        G, -torch.hstack((H, h.reshape(-1, 1))), rcond=delta
    ).solution
    K = K_k[:, :-1]
    k = K_k[:, -1]

    H_GK = H + torch.matmul(G, K)
    P = symmetrize(Q + AtPA + torch.matmul(H_GK.T, K) + torch.matmul(K.T, H))
    p = q + torch.matmul(A.T, p) + torch.matmul(AtP, c) + torch.matmul(
        H_GK.T, k) + torch.matmul(K.T, h)

    return P, p, K, k


def tvlqr(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    Note - for vectorization convenience, the leading dimension of R, r, M, A, B,
    C can be (T + 1) but the last row will be ignored.

    Args:
        Q: [T+1, n, n] tensor.
        q: [T+1, n] tensor.
        R: [T, m, m] tensor.
        r: [T, m] tensor.
        M: [T, n, m] tensor.
        A: [T, n, n] tensor.
        B: [T, n, m] tensor.
        c: [T, n] tensor.

    Returns:
        K: [T, m, n] Gains
        k: [T, m] Affine terms (u_t = torch.matmul(K[t], x_t) + k[t])
        P: [T+1, n, n] tensor encoding initial value function.
        p: [T+1, n] tensor encoding initial value function.
    """

    T = Q.shape[0] - 1
    m = R.shape[1]
    n = Q.shape[1]
    device = Q.device
    dtype = Q.dtype

    P = torch.zeros((T+1, n, n), device=device, dtype=dtype)
    p = torch.zeros((T+1, n), device=device, dtype=dtype)
    K = torch.zeros((T, m, n), device=device, dtype=dtype)
    k = torch.zeros((T, m), device=device, dtype=dtype)

    P[-1] = Q[T]
    p[-1] = q[T]

    for tt in range(T):
        t = T - 1 - tt
        P_t, p_t, K_t, k_t = lqr_step(P[t+1], p[t+1], Q[t], q[t], R[t], r[t], M[t],
                                       A[t], B[t], c[t])
        K[t] = K_t
        k[t] = k_t
        P[t] = P_t
        p[t] = p_t

    return K, k, P, p


def ctvlqr(projector, Q, q, R, r, M, A, B, c, x0, rho=1.0, maxiter=100):
    """Constrained Discrete-time Finite Horizon Time-varying LQR.

    Note - for vectorization convenience, the leading dimension of R, r, M, A, B,
    C can be (T + 1) but the last row will be ignored.

    Args:
        projector: X1, U1 = projector(X, U) projects X, U to the constraint set.
        Q: [T+1, n, n] tensor.
        q: [T+1, n] tensor.
        R: [T, m, m] tensor.
        r: [T, m] tensor.
        M: [T, n, m] tensor.
        A: [T, n, n] tensor.
        B: [T, n, m] tensor.
        c: [T, n] tensor.
        x0: [n] initial condition.
        rho: ADMM rho parameter.
        maxiter: maximum iterations.

    Returns:
        X: [T+1, n] state trajectory.
        U: [T, m] control sequence.
        K: [T, m, n] Gains
        k: [T, m] Affine terms (u_t = torch.matmul(K[t], x_t) + k[t])

    Note: this implementation is ADMM-based and follows from section 5.2 of
          Distributed Optimization and Statistical Learning via the
          Alternating Direction Method of Multipliers, Boyd et.al. 2010.
          https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    """

    T, m, _ = R.shape
    n = Q.shape[1]
    device = Q.device
    dtype = Q.dtype

    X = torch.zeros((T+1, n), device=device, dtype=dtype)
    U = torch.zeros((T, m), device=device, dtype=dtype)
    VX = torch.zeros((T+1, n), device=device, dtype=dtype)
    VU = torch.zeros((T, m), device=device, dtype=dtype)
    ZX = torch.zeros((T+1, n), device=device, dtype=dtype)
    ZU = torch.zeros((T, m), device=device, dtype=dtype)
    K = torch.zeros((T, m, n), device=device, dtype=dtype)
    k = torch.zeros((T, m), device=device, dtype=dtype)
    Im = torch.eye(m, device=device, dtype=dtype).unsqueeze(0).expand(T, -1, -1)
    In = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(T+1, -1, -1)

    for _ in range(maxiter):
        K, k, _, _ = tvlqr(Q + rho*In,
                           q - (ZX - VX),
                           R + rho*Im,
                           r - (ZU - VU),
                           M, A, B, c)
        X, U = rollout(K, k, x0, A, B, c)
        ZX, ZU = projector(X + VX, U + VU)
        VX = VX + X - ZX
        VU = VU + U - ZU

    return X, U, K, k
