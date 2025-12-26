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
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t].

    vmap-compatible: avoids in-place operations.
    """
    T, m, n = K.shape

    # Build trajectories as lists, then stack (vmap-compatible)
    X_list = [x0]
    U_list = []
    x_current = x0

    for t in range(T):
        u = torch.matmul(K[t], x_current) + k[t]
        x_next = torch.matmul(A[t], x_current) + torch.matmul(B[t], u) + c[t]
        U_list.append(u)
        X_list.append(x_next)
        x_current = x_next

    return torch.stack(X_list), torch.stack(U_list)


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

    # CUDA-graphable: use explicit inverse (torch.inverse is graph-compatible)
    # Add sufficient damping for numerical stability
    min_damping = max(delta, 1e-6)
    G_damped = G + min_damping * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)

    # Solve: G_damped @ K_k = -[H, h] using explicit inverse
    # torch.inverse is CUDA-graphable unlike solve/cholesky
    G_inv = torch.inverse(G_damped)
    rhs = -torch.hstack((H, h.reshape(-1, 1)))
    K_k = torch.matmul(G_inv, rhs)

    K = K_k[:, :-1]
    k = K_k[:, -1]

    H_GK = H + torch.matmul(G, K)
    P = symmetrize(Q + AtPA + torch.matmul(H_GK.T, K) + torch.matmul(K.T, H))
    p = q + torch.matmul(A.T, p) + torch.matmul(AtP, c) + torch.matmul(
        H_GK.T, k) + torch.matmul(K.T, h)

    return P, p, K, k


def tvlqr(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    vmap-compatible: avoids in-place operations.

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

    # Build backward pass as lists, then stack (vmap-compatible)
    P_list = [Q[T]]
    p_list = [q[T]]
    K_list = []
    k_list = []

    P_next = Q[T]
    p_next = q[T]

    for tt in range(T):
        t = T - 1 - tt
        P_t, p_t, K_t, k_t = lqr_step(P_next, p_next, Q[t], q[t], R[t], r[t], M[t],
                                       A[t], B[t], c[t])
        K_list.append(K_t)
        k_list.append(k_t)
        P_list.append(P_t)
        p_list.append(p_t)
        P_next = P_t
        p_next = p_t

    # Reverse lists (built backward, need forward order)
    K = torch.stack(list(reversed(K_list)))
    k = torch.stack(list(reversed(k_list)))
    P = torch.stack(list(reversed(P_list)))
    p = torch.stack(list(reversed(p_list)))

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
