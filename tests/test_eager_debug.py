"""Test early stopping with debug output."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')


# Inline a minimal ilqr with debug
def ilqr_debug(cost, dynamics, x0, U, maxiter=50, grad_norm_threshold=1e-4):
    """Minimal iLQR with debug output."""
    from trajax_torch.optimizers import (
        _rollout, quadratize, linearize, pad, evaluate,
        adjoint, tvlqr, line_search_ddp
    )

    T, m = U.shape
    n = x0.shape[0]
    device = U.device
    dtype = U.dtype

    quadratizer = quadratize(cost)
    dynamics_jacobians = linearize(dynamics)
    cost_gradients = linearize(cost)

    X = _rollout(dynamics, U, x0)
    timesteps = torch.arange(X.shape[0], device=device)
    obj = torch.sum(evaluate(cost, X, pad(U)))

    def get_lqr_params(X, U):
        Q, R, M = quadratizer(X, pad(U), timesteps)
        q, r = cost_gradients(X, pad(U), timesteps)
        A, B = dynamics_jacobians(X, pad(U), timesteps)
        return (Q, q, R, r, M, A, B)

    c = torch.zeros((T, n), device=device, dtype=dtype)
    lqr = get_lqr_params(X, U)
    _, q, _, r, _, A, B = lqr
    gradient, adjoints, _ = adjoint(A, B, q, r)
    grad_norm_initial = torch.linalg.norm(gradient)

    print(f"Initial gradient norm: {grad_norm_initial.item():.6f}")
    print(f"Threshold: {grad_norm_threshold:.6f}")
    print()

    alpha_0 = 1.0
    alpha_min = 0.00005

    for iteration in range(maxiter):
        Q, q, R, r, M, A, B = lqr

        K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)
        X_new, U_new, obj_new, alpha = line_search_ddp(
            cost, dynamics, X, U, K, k, obj,
            (), (), alpha_0, alpha_min
        )

        lqr = get_lqr_params(X_new, U_new)
        _, q_new, _, r_new, _, A_new, B_new = lqr
        gradient, adjoints, _ = adjoint(A_new, B_new, q_new, r_new)

        U_step = torch.linalg.norm(U_new - U)
        obj_step = torch.abs(obj_new - obj)

        X, U, obj = X_new, U_new, obj_new

        grad_norm = torch.linalg.norm(gradient)

        print(f"Iter {iteration+1}: grad_norm={grad_norm.item():.6f}, obj={obj.item():.6f}, alpha={alpha:.6f}")

        # Check if we should stop
        if grad_norm <= grad_norm_threshold:
            print(f"  → Stopping: grad_norm {grad_norm.item():.6f} <= threshold {grad_norm_threshold:.6f}")
            return U, iteration + 1

        if alpha <= alpha_min:
            print(f"  → Stopping: alpha {alpha:.6f} <= alpha_min {alpha_min:.6f}")
            return U, iteration + 1

    print(f"  → Reached maxiter={maxiter}")
    return U, maxiter


class DoubleIntegrator:
    def __init__(self, dt=0.1):
        self.dt = dt

    def dynamics(self, x, u, t):
        pos_next = x[0] + self.dt * x[1]
        vel_next = x[1] + self.dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(self, x, u, t):
        return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = DoubleIntegrator()
    T = 15

    x0 = torch.tensor([0.5, 0.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, 1), device=device, dtype=torch.float32)

    U, iters = ilqr_debug(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        maxiter=50,
        grad_norm_threshold=1e-4
    )

    print(f"\nFinal iterations: {iters}")
