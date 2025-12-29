"""Detailed debug of stopping criteria."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch.optimizers import (
    _rollout, quadratize, linearize, pad, evaluate,
    adjoint, tvlqr, line_search_ddp
)


class SimpleSystem:
    def __init__(self):
        self.dt = 0.05

    def dynamics(self, x, u, t):
        return x + self.dt * u

    def cost(self, x, u, t):
        return torch.sum(x**2) + 0.01 * torch.sum(u**2)


def ilqr_debug_detailed(cost, dynamics, x0, U, maxiter=100, grad_norm_threshold=1e-3):
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

    alpha_0 = 1.0
    alpha_min = 0.00005
    obj_step_threshold = 0.0
    inputs_step_threshold = 0.0

    print(f"grad_norm_threshold = {grad_norm_threshold}")
    print(f"alpha_min = {alpha_min}")
    print()

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

        # Check stopping criteria
        grad_norm = torch.linalg.norm(gradient)
        has_nan = torch.isnan(gradient).any()
        effective_grad_norm = torch.where(has_nan,
                                          torch.tensor(float('inf'), device=device, dtype=dtype),
                                          grad_norm)

        # Convert tensor bools to Python bools
        still_improving_obj = (obj_step > obj_step_threshold * (torch.abs(obj) + 1.0)).item()
        still_moving_U = (U_step > inputs_step_threshold * (torch.linalg.norm(U) + 1.0)).item()
        still_progressing = still_improving_obj and still_moving_U
        has_potential = (effective_grad_norm > grad_norm_threshold).item() and still_progressing

        if iteration < 5 or iteration % 10 == 9:
            print(f"Iter {iteration+1}:")
            print(f"  grad_norm = {grad_norm.item():.6f}, effective = {effective_grad_norm.item():.6f}")
            print(f"  alpha = {alpha:.6f}")
            print(f"  still_improving_obj = {still_improving_obj}")
            print(f"  still_moving_U = {still_moving_U}")
            print(f"  still_progressing = {still_progressing}")
            print(f"  has_potential = {has_potential}")
            print(f"  should_continue = {has_potential and alpha > alpha_min}")
            print()

        # Early exit if converged
        if not (has_potential and alpha > alpha_min):
            print(f"STOPPING at iteration {iteration+1}")
            print(f"  has_potential = {has_potential}")
            print(f"  alpha > alpha_min = {alpha > alpha_min}")
            return U, iteration + 1

    print(f"Reached maxiter={maxiter}")
    return U, maxiter


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = SimpleSystem()
    T = 10
    n = 2
    m = 2

    x0 = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, m), device=device, dtype=torch.float32)

    U, iters = ilqr_debug_detailed(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        maxiter=100,
        grad_norm_threshold=1e-3
    )

    print(f"\nFinal iterations: {iters}")
