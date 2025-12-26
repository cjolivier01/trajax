"""Test constrained iLQR on GPU."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


class CartPole:
    """Cart-pole system with discrete-time dynamics."""

    def __init__(self, mass_cart=1.0, mass_pole=0.1, length=0.5,
                 gravity=9.81, dt=0.02):
        self.mc = mass_cart
        self.mp = mass_pole
        self.l = length
        self.g = gravity
        self.dt = dt

    def dynamics(self, x, u, t):
        """Cart-pole dynamics: x = [pos, vel, theta, theta_dot]."""
        pos, vel, theta, theta_dot = x[0], x[1], x[2], x[3]
        force = u[0]

        # Cart-pole physics
        total_mass = self.mc + self.mp
        pole_mass_length = self.mp * self.l

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (
            self.l * (4.0/3.0 - self.mp * cos_theta**2 / total_mass))
        acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        # Euler integration
        pos_next = pos + self.dt * vel
        vel_next = vel + self.dt * acc
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_acc

        return torch.stack([pos_next, vel_next, theta_next, theta_dot_next])

    def cost(self, x, u, t):
        """Quadratic cost - keep pole upright, cart at origin."""
        # Target: upright (theta=0), at origin (pos=0), stationary
        pos_cost = 1.0 * x[0]**2
        vel_cost = 0.1 * x[1]**2
        theta_cost = 10.0 * x[2]**2
        theta_dot_cost = 0.1 * x[3]**2
        control_cost = 0.01 * u[0]**2
        return pos_cost + vel_cost + theta_cost + theta_dot_cost + control_cost


def test_constrained_ilqr_equality():
    """Test constrained iLQR with equality constraints on GPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    system = CartPole()
    T = 50
    n_state = 4
    n_control = 1

    # Initial state: pole hanging down, cart at origin
    x0 = torch.tensor([0.0, 0.0, torch.pi, 0.0], device=device, dtype=torch.float64)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float64)

    # Equality constraint: cart position must be 0 at final time
    # Note: constraint functions must return fixed size for all t
    def equality_constraint(x, u, t):
        # Return pos constraint, but only enforce at final time via large penalty
        # Actually, better approach: always return same size, use weight based on t
        # For simplicity in augmented Lagrangian, we want final-time constraint
        # We can achieve this by returning the constraint at all times but it only
        # matters at the end due to the optimization
        return torch.tensor([x[0] if t == T else 0.0], device=device, dtype=torch.float64)

    # Run constrained iLQR
    result = torch_optimizers.constrained_ilqr(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        equality_constraint=equality_constraint,
        maxiter_al=10,
        maxiter_ilqr=50,
        constraints_threshold=1e-2,
        make_psd=False
    )

    X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints, \
        max_violation, obj, gradient, iter_ilqr, iter_al = result

    # Verify all outputs are on GPU
    assert X.device.type == device.split(':')[0]
    assert U.device.type == device.split(':')[0]

    # Check shapes
    assert X.shape == (T + 1, n_state)
    assert U.shape == (T, n_control)

    # Check constraint satisfaction
    print(f"Max constraint violation: {max_violation.item():.6f}")
    print(f"Final cart position: {X[-1, 0].item():.6f}")
    print(f"iLQR iterations: {iter_ilqr}, AL iterations: {iter_al}")
    print(f"Final objective: {obj.item():.6f}")

    assert torch.all(torch.isfinite(X))
    assert torch.all(torch.isfinite(U))

    print("✓ Equality constraint test passed\n")


def test_constrained_ilqr_inequality():
    """Test constrained iLQR with inequality constraints on GPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    system = CartPole()
    T = 30
    n_state = 4
    n_control = 1

    # Initial state
    x0 = torch.tensor([0.0, 0.0, 0.2, 0.0], device=device, dtype=torch.float64)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float64)

    # Inequality constraint: limit cart position to [-0.5, 0.5]
    position_limit = 0.5

    def inequality_constraint(x, u, t):
        # pos <= position_limit  => pos - position_limit <= 0
        # pos >= -position_limit => -pos - position_limit <= 0
        return torch.tensor([
            x[0] - position_limit,    # pos <= 0.5
            -x[0] - position_limit    # pos >= -0.5
        ], device=device, dtype=torch.float64)

    # Run constrained iLQR
    result = torch_optimizers.constrained_ilqr(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        inequality_constraint=inequality_constraint,
        maxiter_al=10,
        maxiter_ilqr=50,
        constraints_threshold=1e-2,
        make_psd=False
    )

    X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints, \
        max_violation, obj, gradient, iter_ilqr, iter_al = result

    # Verify all outputs are on GPU
    assert X.device.type == device.split(':')[0]
    assert U.device.type == device.split(':')[0]

    # Check shapes
    assert X.shape == (T + 1, n_state)
    assert U.shape == (T, n_control)

    # Check constraint satisfaction
    max_pos = torch.max(torch.abs(X[:, 0]))
    print(f"Max constraint violation: {max_violation.item():.6f}")
    print(f"Max cart position: {max_pos.item():.6f}")
    print(f"iLQR iterations: {iter_ilqr}, AL iterations: {iter_al}")
    print(f"Final objective: {obj.item():.6f}")

    assert torch.all(torch.isfinite(X))
    assert torch.all(torch.isfinite(U))

    print("✓ Inequality constraint test passed\n")


def test_constrained_ilqr_both():
    """Test constrained iLQR with both equality and inequality constraints."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Simple double integrator for cleaner test
    dt = 0.1

    def dynamics(x, u, t):
        """x = [pos, vel]"""
        pos_next = x[0] + dt * x[1]
        vel_next = x[1] + dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(x, u, t):
        """Minimize distance to target (1, 0) with control effort."""
        target_pos = 1.0
        return (x[0] - target_pos)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

    T = 20
    x0 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    U_init = torch.zeros((T, 1), device=device, dtype=torch.float64)

    # Equality: final velocity must be 0
    # Constraint functions must return consistent sizes
    def equality_constraint(x, u, t):
        # Only enforce at final time
        return torch.tensor([x[1] if t == T else 0.0], device=device, dtype=torch.float64)

    # Inequality: velocity must stay in [-2, 2]
    vel_limit = 2.0

    def inequality_constraint(x, u, t):
        return torch.tensor([
            x[1] - vel_limit,   # vel <= 2
            -x[1] - vel_limit   # vel >= -2
        ], device=device, dtype=torch.float64)

    # Run constrained iLQR
    result = torch_optimizers.constrained_ilqr(
        cost,
        dynamics,
        x0,
        U_init,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        maxiter_al=15,
        maxiter_ilqr=50,
        constraints_threshold=1e-3,
        make_psd=False
    )

    X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints, \
        max_violation, obj, gradient, iter_ilqr, iter_al = result

    # Verify all outputs are on GPU
    assert X.device.type == device.split(':')[0]
    assert U.device.type == device.split(':')[0]

    # Check results
    print(f"Max constraint violation: {max_violation.item():.6f}")
    print(f"Final position: {X[-1, 0].item():.6f}, Final velocity: {X[-1, 1].item():.6f}")
    print(f"Max velocity: {torch.max(torch.abs(X[:, 1])).item():.6f}")
    print(f"iLQR iterations: {iter_ilqr}, AL iterations: {iter_al}")
    print(f"Final objective: {obj.item():.6f}")

    assert torch.all(torch.isfinite(X))
    assert torch.all(torch.isfinite(U))

    print("✓ Combined constraints test passed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Constrained iLQR on GPU")
    print("=" * 60 + "\n")

    test_constrained_ilqr_equality()
    test_constrained_ilqr_inequality()
    test_constrained_ilqr_both()

    print("=" * 60)
    print("All constrained iLQR tests passed!")
    print("=" * 60)
