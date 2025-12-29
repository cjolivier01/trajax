"""Test early stopping with a problem that actually converges."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


class SimpleSystem:
    """Very simple linear system that should converge quickly."""

    def __init__(self):
        self.dt = 0.05

    def dynamics(self, x, u, t):
        """Simple dynamics."""
        return x + self.dt * u

    def cost(self, x, u, t):
        """Quadratic cost - minimize state and control."""
        return torch.sum(x**2) + 0.01 * torch.sum(u**2)


def test_early_stop():
    """Test that early stopping works."""
    print("Testing early stopping with a simple converging problem...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = SimpleSystem()
    T = 10
    n = 2  # 2D state
    m = 2  # 2D control

    # Start with a state that needs correction
    x0 = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, m), device=device, dtype=torch.float32)

    print(f"Initial state: {x0}")
    print(f"Running with maxiter=100, grad_norm_threshold=1e-3\n")

    result = torch_optimizers.ilqr(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        maxiter=100,
        grad_norm_threshold=1e-3,  # Looser threshold
        make_psd=False
    )

    U = result[1]
    obj = result[2]
    iterations = result[6]

    print(f"Converged in {iterations} iterations")
    print(f"Final objective: {obj.item():.6f}")
    print(f"Final control norm: {torch.linalg.norm(U).item():.6f}")

    if iterations < 100:
        print(f"\n✓ Early stopping worked! Stopped at {iterations}/100 iterations")
        return True
    else:
        print(f"\n✗ Early stopping did not work")
        return False


if __name__ == '__main__':
    try:
        success = test_early_stop()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
