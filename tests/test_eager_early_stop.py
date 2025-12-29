"""Test early stopping without torch.compile."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


class DoubleIntegrator:
    """Simple double integrator for testing."""

    def __init__(self, dt=0.1):
        self.dt = dt

    def dynamics(self, x, u, t):
        """x = [position, velocity], returns next state."""
        pos_next = x[0] + self.dt * x[1]
        vel_next = x[1] + self.dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(self, x, u, t):
        """Quadratic cost - reach position 1.0."""
        return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2


def test_ilqr_early_stop():
    """Test that early stopping works in eager mode."""
    print("Testing iLQR with early stopping (eager mode)...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = DoubleIntegrator()
    T = 15
    n_control = 1

    x0 = torch.tensor([0.5, 0.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    result = torch_optimizers.ilqr(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        maxiter=50,  # Max iterations
        grad_norm_threshold=1e-4,  # Will stop early!
        make_psd=False
    )

    U = result[1]
    iterations = result[6]

    print(f"  Iterations: {iterations}")
    print(f"  Final control norm: {torch.linalg.norm(U).item():.4f}")

    # Verify early stopping worked (should stop before maxiter=50)
    if iterations < 50:
        print(f"  ✓ Early stopping works! Stopped at {iterations}/50 iterations")
    else:
        print(f"  ✗ Early stopping did not work - ran full {iterations} iterations")

    return iterations < 50


if __name__ == '__main__':
    try:
        success = test_ilqr_early_stop()
        if success:
            print("\n✓ Test passed!")
        else:
            print("\n✗ Test failed!")
            exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
