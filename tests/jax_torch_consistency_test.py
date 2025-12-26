"""Tests for JAX-PyTorch consistency.

This module contains tests to verify that the PyTorch port of trajax
produces consistent results with the original JAX implementation.
"""

import numpy as np
import torch
import jax.numpy as jnp
import jax

# Enable 64-bit precision for JAX
try:
    from jax.config import config
    config.update('jax_enable_x64', True)
except:
    jax.config.update('jax_enable_x64', True)

# Import JAX versions
from trajax import optimizers as jax_optimizers
from trajax.integrators import euler as jax_euler
from trajax.integrators import rk4 as jax_rk4

# Import PyTorch versions
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers
from trajax_torch.integrators import euler as torch_euler
from trajax_torch.integrators import rk4 as torch_rk4


def numpy_to_torch(arr, device='cpu'):
    """Convert numpy array to torch tensor."""
    return torch.tensor(arr, dtype=torch.float64, device=device)


def jax_to_torch(arr, device='cpu'):
    """Convert JAX array to torch tensor."""
    return numpy_to_torch(np.array(arr), device=device)


def torch_to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


class ConsistencyTest:
    """Test JAX-PyTorch consistency."""

    def __init__(self, device='cpu', rtol=1e-5, atol=1e-7):
        self.device = device
        self.rtol = rtol
        self.atol = atol
        self.passed = 0
        self.failed = 0

    def assert_close(self, jax_result, torch_result, test_name):
        """Assert that JAX and PyTorch results are close."""
        jax_np = np.array(jax_result)
        torch_np = torch_to_numpy(torch_result)

        try:
            np.testing.assert_allclose(jax_np, torch_np, rtol=self.rtol, atol=self.atol)
            print(f"✓ {test_name} PASSED")
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"✗ {test_name} FAILED")
            print(f"  JAX shape: {jax_np.shape}, PyTorch shape: {torch_np.shape}")
            print(f"  Max difference: {np.max(np.abs(jax_np - torch_np))}")
            print(f"  Error: {str(e)[:200]}")
            self.failed += 1
            return False

    def test_pendulum_dynamics(self):
        """Test pendulum dynamics consistency."""
        print("\n=== Testing Pendulum Dynamics ===")

        def pendulum(state, action, t):
            theta, theta_dot = state[0], state[1]
            m, l, g = 1.0, 1.0, 9.81
            return jnp.array([
                theta_dot,
                (jnp.squeeze(action) - m * g * l * jnp.sin(theta)) / (m * l * l)
            ])

        def pendulum_torch(state, action, t):
            theta, theta_dot = state[0], state[1]
            m, l, g = 1.0, 1.0, 9.81
            return torch.tensor([
                theta_dot,
                (torch.squeeze(action) - m * g * l * torch.sin(theta)) / (m * l * l)
            ], device=state.device, dtype=state.dtype)

        # Test raw dynamics
        state_jax = jnp.array([0.5, 0.1])
        action_jax = jnp.array([0.5])
        result_jax = pendulum(state_jax, action_jax, 0)

        state_torch = numpy_to_torch(np.array([0.5, 0.1]), self.device)
        action_torch = numpy_to_torch(np.array([0.5]), self.device)
        result_torch = pendulum_torch(state_torch, action_torch, 0)

        self.assert_close(result_jax, result_torch, "Pendulum dynamics")

        # Test Euler integration
        dynamics_jax = jax_euler(pendulum, dt=0.01)
        dynamics_torch = torch_euler(pendulum_torch, dt=0.01)

        result_jax = dynamics_jax(state_jax, action_jax, 0)
        result_torch = dynamics_torch(state_torch, action_torch, 0)

        self.assert_close(result_jax, result_torch, "Euler integration")

        # Test RK4 integration
        dynamics_jax = jax_rk4(pendulum, dt=0.01)
        dynamics_torch = torch_rk4(pendulum_torch, dt=0.01)

        result_jax = dynamics_jax(state_jax, action_jax, 0)
        result_torch = dynamics_torch(state_torch, action_torch, 0)

        self.assert_close(result_jax, result_torch, "RK4 integration")

    def test_rollout(self):
        """Test rollout consistency."""
        print("\n=== Testing Rollout ===")

        # Simple linear dynamics
        def dynamics_jax(x, u, t):
            return x + 0.1 * u

        def dynamics_torch(x, u, t):
            return x + 0.1 * u

        T = 10
        U_jax = jnp.ones((T, 1))
        x0_jax = jnp.array([0.0])

        U_torch = torch.ones((T, 1), device=self.device, dtype=torch.float64)
        x0_torch = torch.tensor([0.0], device=self.device, dtype=torch.float64)

        X_jax = jax_optimizers.rollout(dynamics_jax, U_jax, x0_jax)
        X_torch = torch_optimizers.rollout(dynamics_torch, U_torch, x0_torch)

        self.assert_close(X_jax, X_torch, "Rollout")

    def test_objective(self):
        """Test objective function consistency."""
        print("\n=== Testing Objective Function ===")

        # Quadratic cost
        def cost_jax(x, u, t):
            return 0.5 * jnp.sum(x**2) + 0.5 * jnp.sum(u**2)

        def cost_torch(x, u, t):
            return 0.5 * torch.sum(x**2) + 0.5 * torch.sum(u**2)

        def dynamics_jax(x, u, t):
            return x + 0.1 * u

        def dynamics_torch(x, u, t):
            return x + 0.1 * u

        T = 10
        U_jax = jnp.ones((T, 1))
        x0_jax = jnp.array([0.0])

        U_torch = torch.ones((T, 1), device=self.device, dtype=torch.float64)
        x0_torch = torch.tensor([0.0], device=self.device, dtype=torch.float64)

        obj_jax = jax_optimizers.objective(cost_jax, dynamics_jax, U_jax, x0_jax)
        obj_torch = torch_optimizers.objective(cost_torch, dynamics_torch, U_torch, x0_torch)

        self.assert_close(obj_jax, obj_torch, "Objective function")

    def test_ilqr_simple(self):
        """Test iLQR on simple problem."""
        print("\n=== Testing iLQR on Simple LQR Problem ===")

        # Simple LQR problem
        def cost_jax(x, u, t):
            Q = 1.0
            R = 0.1
            return 0.5 * Q * jnp.sum(x**2) + 0.5 * R * jnp.sum(u**2)

        def cost_torch(x, u, t):
            Q = 1.0
            R = 0.1
            return 0.5 * Q * torch.sum(x**2) + 0.5 * R * torch.sum(u**2)

        def dynamics_jax(x, u, t):
            A = 1.1
            B = 0.5
            return A * x + B * u

        def dynamics_torch(x, u, t):
            A = 1.1
            B = 0.5
            return A * x + B * u

        T = 20
        x0_jax = jnp.array([1.0])
        U_jax = jnp.zeros((T, 1))

        x0_torch = torch.tensor([1.0], device=self.device, dtype=torch.float64)
        U_torch = torch.zeros((T, 1), device=self.device, dtype=torch.float64)

        # Run iLQR
        X_jax, U_jax, obj_jax, grad_jax, _, _, iter_jax = jax_optimizers.ilqr(
            cost_jax, dynamics_jax, x0_jax, U_jax, maxiter=50, make_psd=False
        )

        X_torch, U_torch, obj_torch, grad_torch, _, _, iter_torch = torch_optimizers.ilqr(
            cost_torch, dynamics_torch, x0_torch, U_torch, maxiter=50, make_psd=False
        )

        print(f"  JAX iterations: {iter_jax}, PyTorch iterations: {iter_torch}")
        print(f"  JAX objective: {obj_jax:.6f}, PyTorch objective: {obj_torch:.6f}")

        self.assert_close(X_jax, X_torch, "iLQR - State trajectory")
        self.assert_close(U_jax, U_torch, "iLQR - Control sequence")
        self.assert_close(obj_jax, obj_torch, "iLQR - Objective value")

    def test_ilqr_pendulum(self):
        """Test iLQR on pendulum swingup."""
        print("\n=== Testing iLQR on Pendulum Swingup ===")

        horizon = 50

        # JAX pendulum
        def pendulum_jax(state, action, t):
            theta, theta_dot = state
            m, l, g = 1.0, 1.0, 9.81
            return jnp.array([
                theta_dot,
                (jnp.squeeze(action) - m * g * l * jnp.sin(theta)) / (m * l * l)
            ])

        # PyTorch pendulum
        def pendulum_torch(state, action, t):
            theta, theta_dot = state[0], state[1]
            m, l, g = 1.0, 1.0, 9.81
            return torch.tensor([
                theta_dot,
                (torch.squeeze(action) - m * g * l * torch.sin(theta)) / (m * l * l)
            ], device=state.device, dtype=state.dtype)

        # Wrap angle
        def angle_wrap_jax(theta):
            return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

        def angle_wrap_torch(theta):
            return (theta + np.pi) % (2 * np.pi) - np.pi

        # JAX cost
        def cost_jax(state, action, t):
            final_weight, stage_weight, action_weight = 100.0, 10.0, 1.0
            theta, theta_dot = state
            theta_err = angle_wrap_jax(theta - jnp.pi)
            state_cost = stage_weight * (theta_err**2 + theta_dot**2)
            action_cost = action_weight * jnp.squeeze(action)**2
            return jnp.where(t == horizon, final_weight * state_cost,
                             state_cost + action_cost)

        # PyTorch cost
        def cost_torch(state, action, t):
            final_weight, stage_weight, action_weight = 100.0, 10.0, 1.0
            theta, theta_dot = state[0], state[1]
            # Simplified angle wrap that maintains gradients
            pi_tensor = torch.tensor(np.pi, dtype=theta.dtype, device=theta.device)
            two_pi = 2.0 * pi_tensor
            theta_wrapped = theta - pi_tensor
            theta_wrapped = theta_wrapped - two_pi * torch.floor(theta_wrapped / two_pi + 0.5)
            theta_err = theta_wrapped
            state_cost = stage_weight * (theta_err**2 + theta_dot**2)
            action_cost = action_weight * torch.squeeze(action)**2
            # Use torch.where to maintain gradients
            total_cost = torch.where(
                torch.tensor(t == horizon, dtype=torch.bool, device=theta.device),
                final_weight * state_cost,
                state_cost + action_cost
            )
            return total_cost

        dynamics_jax = jax_rk4(pendulum_jax, dt=0.01)
        dynamics_torch = torch_rk4(pendulum_torch, dt=0.01)

        x0_jax = jnp.array([-0.9, 0.0])
        U0_jax = jnp.zeros((horizon, 1))

        x0_torch = torch.tensor([-0.9, 0.0], device=self.device, dtype=torch.float64)
        U0_torch = torch.zeros((horizon, 1), device=self.device, dtype=torch.float64)

        # Run iLQR
        print("  Running JAX iLQR...")
        X_jax, U_jax, obj_jax, grad_jax, _, _, iter_jax = jax_optimizers.ilqr(
            cost_jax, dynamics_jax, x0_jax, U0_jax, maxiter=100, make_psd=False
        )

        print("  Running PyTorch iLQR...")
        X_torch, U_torch, obj_torch, grad_torch, _, _, iter_torch = torch_optimizers.ilqr(
            cost_torch, dynamics_torch, x0_torch, U0_torch, maxiter=100, make_psd=False
        )

        print(f"  JAX iterations: {iter_jax}, PyTorch iterations: {iter_torch}")
        print(f"  JAX objective: {float(obj_jax):.6f}, PyTorch objective: {float(obj_torch):.6f}")
        print(f"  JAX final state: {np.array(X_jax[-1])}")
        print(f"  PyTorch final state: {torch_to_numpy(X_torch[-1])}")

        # For pendulum, we allow slightly more tolerance due to numerical differences
        # in trigonometric functions
        original_rtol = self.rtol
        original_atol = self.atol
        self.rtol = 1e-3
        self.atol = 1e-5

        self.assert_close(X_jax, X_torch, "iLQR Pendulum - State trajectory")
        self.assert_close(U_jax, U_torch, "iLQR Pendulum - Control sequence")

        # Check that both achieved similar objective (within 1%)
        obj_diff = abs(float(obj_jax) - float(obj_torch))
        obj_relative = obj_diff / (abs(float(obj_jax)) + 1e-8)
        if obj_relative < 0.01:
            print(f"✓ iLQR Pendulum - Objective consistency PASSED (relative diff: {obj_relative:.6f})")
            self.passed += 1
        else:
            print(f"✗ iLQR Pendulum - Objective consistency FAILED (relative diff: {obj_relative:.6f})")
            self.failed += 1

        # Restore tolerances
        self.rtol = original_rtol
        self.atol = original_atol

    def run_all_tests(self):
        """Run all consistency tests."""
        print(f"\n{'='*60}")
        print(f"Running JAX-PyTorch Consistency Tests on {self.device}")
        print(f"{'='*60}")

        self.test_pendulum_dynamics()
        self.test_rollout()
        self.test_objective()
        self.test_ilqr_simple()
        self.test_ilqr_pendulum()

        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed} passed, {self.failed} failed")
        print(f"{'='*60}\n")

        return self.failed == 0


if __name__ == '__main__':
    # Test on CPU
    tester_cpu = ConsistencyTest(device='cpu', rtol=1e-5, atol=1e-7)
    success_cpu = tester_cpu.run_all_tests()

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU is available, running tests on GPU...")
        print("="*60)
        tester_gpu = ConsistencyTest(device='cuda', rtol=1e-5, atol=1e-7)
        success_gpu = tester_gpu.run_all_tests()
        success = success_cpu and success_gpu
    else:
        print("\nGPU not available, skipping GPU tests")
        success = success_cpu

    if success:
        print("\n🎉 All consistency tests PASSED!")
        exit(0)
    else:
        print("\n❌ Some consistency tests FAILED")
        exit(1)
