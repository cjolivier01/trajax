"""Test early stopping with a loose threshold that will actually trigger."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


class SimpleSystem:
    def __init__(self):
        self.dt = 0.05

    def dynamics(self, x, u, t):
        return x + self.dt * u

    def cost(self, x, u, t):
        return torch.sum(x**2) + 0.01 * torch.sum(u**2)


def test_early_stop():
    """Test that early stopping works with a loose threshold."""
    print("Testing early stopping with loose threshold...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = SimpleSystem()
    T = 10
    n = 2
    m = 2

    x0 = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, m), device=device, dtype=torch.float32)

    # Use a very loose threshold that the optimizer will reach
    print(f"Running with maxiter=100, grad_norm_threshold=2.6 (loose!)\n")

    result = torch_optimizers.ilqr(
        system.cost,
        system.dynamics,
        x0,
        U_init,
        maxiter=100,
        grad_norm_threshold=2.6,  # Very loose - will trigger around iteration 40
        make_psd=False
    )

    iterations = result[6]

    print(f"\nConverged in {iterations} iterations")

    if iterations < 100:
        print(f"✓ Early stopping WORKS! Stopped at {iterations}/100 iterations")
        print(f"  (With a threshold of 2.6, it stopped when gradient norm dropped below that value)")
        return True
    else:
        print(f"✗ Early stopping did not work")
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
