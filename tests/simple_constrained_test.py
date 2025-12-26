"""Simple test for constrained iLQR."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Simple double integrator
    dt = 0.1

    def dynamics(x, u, t):
        """x = [pos, vel]"""
        pos_next = x[0] + dt * x[1]
        vel_next = x[1] + dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(x, u, t):
        """Reach target position 1.0."""
        return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

    # Inequality constraint: limit control to [-5, 5]
    def inequality_constraint(x, u, t):
        return torch.tensor([
            u[0] - 5.0,   # u <= 5
            -u[0] - 5.0   # u >= -5
        ], device=device, dtype=torch.float64)

    T = 10
    x0 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    U_init = torch.zeros((T, 1), device=device, dtype=torch.float64)

    print("Running constrained iLQR...")
    result = torch_optimizers.constrained_ilqr(
        cost,
        dynamics,
        x0,
        U_init,
        inequality_constraint=inequality_constraint,
        maxiter_al=3,
        maxiter_ilqr=10,
        constraints_threshold=1e-2,
        make_psd=False
    )

    X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints, \
        max_violation, obj, gradient, iter_ilqr, iter_al = result

    print(f"Success!")
    print(f"  Final position: {X[-1, 0].item():.4f}")
    print(f"  Max control: {torch.max(torch.abs(U)).item():.4f}")
    print(f"  Max violation: {max_violation.item():.6f}")
    print(f"  iLQR iters: {iter_ilqr}, AL iters: {iter_al}")
    print(f"  All on GPU: {X.device.type == device.split(':')[0]}")


if __name__ == '__main__':
    main()
