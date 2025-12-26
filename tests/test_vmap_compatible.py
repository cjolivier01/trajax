"""Test that PyTorch trajax optimizers are vmap-compatible."""

import torch
from torch.func import vmap
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


def test_ilqr_vmap():
    """Test that iLQR works with vmap."""
    print("Testing iLQR with vmap...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = 0.1

    def dynamics(x, u, t):
        """Double integrator: x = [pos, vel]"""
        pos_next = x[0] + dt * x[1]
        vel_next = x[1] + dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(x, u, t):
        """Quadratic cost."""
        return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

    T = 10
    batch_size = 5

    # Batch of initial states
    x0_batch = torch.randn(batch_size, 2, device=device, dtype=torch.float32)
    # Batch of initial controls
    U_batch = torch.zeros(batch_size, T, 1, device=device, dtype=torch.float32)

    def solve_single(x0, U):
        """Solve single problem."""
        result = torch_optimizers.ilqr(
            cost,
            dynamics,
            x0,
            U,
            maxiter=5,
            make_psd=False
        )
        return result[1]  # Return U

    # Use vmap to batch over initial conditions
    print(f"  Solving {batch_size} problems with vmap...")
    U_batch_out = vmap(solve_single)(x0_batch, U_batch)

    print(f"  Output shape: {U_batch_out.shape}")
    assert U_batch_out.shape == (batch_size, T, 1)
    assert torch.all(torch.isfinite(U_batch_out))

    print("  ✓ iLQR vmap test passed\n")


def test_constrained_ilqr_vmap():
    """Test that constrained iLQR works with vmap."""
    print("Testing constrained iLQR with vmap...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.tensor(0.1, device=device, dtype=torch.float32)

    def dynamics(x, u, t):
        """Double integrator."""
        pos_next = x[0] + dt * x[1]
        vel_next = x[1] + dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(x, u, t):
        """Quadratic cost."""
        return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

    def inequality_constraint(x, u, t):
        """Limit control to [-3, 3]."""
        return torch.stack([
            u[0] - torch.tensor(3.0, device=u.device, dtype=u.dtype),   # u <= 3
            -u[0] - torch.tensor(3.0, device=u.device, dtype=u.dtype)   # u >= -3
        ])

    T = 10
    batch_size = 3

    # Batch of initial states
    x0_batch = torch.randn(batch_size, 2, device=device, dtype=torch.float32) * 0.5
    # Batch of initial controls
    U_batch = torch.zeros(batch_size, T, 1, device=device, dtype=torch.float32)

    def solve_single(x0, U):
        """Solve single constrained problem."""
        result = torch_optimizers.constrained_ilqr(
            cost,
            dynamics,
            x0,
            U,
            inequality_constraint=inequality_constraint,
            maxiter_al=2,
            maxiter_ilqr=5,
            constraints_threshold=1e-2,
            make_psd=False
        )
        return result[1]  # Return U

    # Use vmap to batch over initial conditions
    print(f"  Solving {batch_size} constrained problems with vmap...")
    U_batch_out = vmap(solve_single)(x0_batch, U_batch)

    print(f"  Output shape: {U_batch_out.shape}")
    assert U_batch_out.shape == (batch_size, T, 1)
    assert torch.all(torch.isfinite(U_batch_out))

    print("  ✓ Constrained iLQR vmap test passed\n")


def test_rollout_vmap():
    """Test that rollout is vmap-compatible."""
    print("Testing rollout with vmap...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = 0.1

    def dynamics(x, u, t):
        pos_next = x[0] + dt * x[1]
        vel_next = x[1] + dt * u[0]
        return torch.stack([pos_next, vel_next])

    T = 15
    batch_size = 10

    # Batch of initial states
    x0_batch = torch.randn(batch_size, 2, device=device, dtype=torch.float32)
    # Batch of control sequences
    U_batch = torch.randn(batch_size, T, 1, device=device, dtype=torch.float32) * 0.1

    def rollout_single(x0, U):
        return torch_optimizers.rollout(dynamics, U, x0)

    print(f"  Rolling out {batch_size} trajectories with vmap...")
    X_batch = vmap(rollout_single)(x0_batch, U_batch)

    print(f"  Output shape: {X_batch.shape}")
    assert X_batch.shape == (batch_size, T + 1, 2)
    assert torch.all(torch.isfinite(X_batch))

    print("  ✓ Rollout vmap test passed\n")


if __name__ == '__main__':
    print("=" * 70)
    print("vmap Compatibility Tests for PyTorch Trajax")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print()

    try:
        test_rollout_vmap()
        test_ilqr_vmap()
        test_constrained_ilqr_vmap()

        print("=" * 70)
        print("✓ All vmap tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
