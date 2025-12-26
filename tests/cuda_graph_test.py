"""Tests for CUDA graph compatibility of PyTorch trajax implementation.

This module verifies that the PyTorch trajectory optimization algorithms
can be captured in CUDA graphs for maximum performance.
"""

import torch
import pytest
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers
from trajax_torch.integrators import euler


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA not available")


class SimplePendulum:
    """Simple pendulum dynamics for testing (discrete-time)."""

    def __init__(self, mass=1.0, length=1.0, damping=0.1, gravity=9.81, dt=0.05):
        self.mass = mass
        self.length = length
        self.damping = damping
        self.gravity = gravity
        self.dt = dt

    def dynamics(self, x, u, t):
        """Pendulum dynamics: x = [theta, theta_dot], returns next state."""
        theta, theta_dot = x[0], x[1]

        # Equation of motion for pendulum
        theta_ddot = (-self.gravity / self.length * torch.sin(theta)
                     - self.damping * theta_dot
                     + u[0] / (self.mass * self.length**2))

        # Euler integration for discrete-time dynamics
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_ddot

        return torch.stack([theta_next, theta_dot_next])

    def cost(self, x, u, t):
        """Quadratic cost on state and control."""
        # Target: upright position (theta=0, theta_dot=0)
        state_cost = x[0]**2 + 0.1 * x[1]**2
        control_cost = 0.01 * u[0]**2
        return state_cost + control_cost


class DoubleIntegrator:
    """Simple double integrator for testing (discrete-time)."""

    def __init__(self, dt=0.1):
        self.dt = dt

    def dynamics(self, x, u, t):
        """Double integrator: x = [position, velocity], returns next state."""
        # Discrete-time double integrator with Euler integration
        pos_next = x[0] + self.dt * x[1]
        vel_next = x[1] + self.dt * u[0]
        return torch.stack([pos_next, vel_next])

    def cost(self, x, u, t):
        """Quadratic cost."""
        # Target: origin
        state_cost = x[0]**2 + x[1]**2
        control_cost = 0.1 * u[0]**2
        return state_cost + control_cost


def test_ilqr_cuda_graph_simple():
    """Test that iLQR can be captured in a CUDA graph with simple problem."""
    device = 'cuda'

    # Setup problem
    system = DoubleIntegrator()
    T = 20
    n_state = 2
    n_control = 1

    # Initial state and controls
    x0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    # iLQR parameters
    max_iter = 5

    # Create a wrapped function for graph capture
    def ilqr_step(x0, U):
        """Single iLQR optimization step."""
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=max_iter,
            make_psd=False  # NOTE: make_psd uses eigh which is not CUDA-graphable
        )
        return result[1]  # Return optimized controls (U is second return value)

    # Warmup run (not in graph)
    warmup_result = ilqr_step(x0, U_init)

    # Capture CUDA graph
    g = torch.cuda.CUDAGraph()

    # Use static inputs for graph
    static_x0 = x0.clone()
    static_U = U_init.clone()

    # Warmup for graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    # Capture the graph
    with torch.cuda.graph(g):
        static_output = ilqr_step(static_x0, static_U)

    # Replay the graph
    g.replay()

    # Verify output is reasonable
    assert static_output.shape == (T, n_control)
    assert torch.all(torch.isfinite(static_output))

    print("✓ iLQR CUDA graph test passed for double integrator")


def test_ilqr_cuda_graph_pendulum():
    """Test that iLQR can be captured in a CUDA graph with pendulum."""
    device = 'cuda'

    # Setup pendulum problem
    system = SimplePendulum()
    T = 30
    n_state = 2
    n_control = 1

    # Initial state: hanging down, need to swing up
    x0 = torch.tensor([torch.pi, 0.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    # iLQR parameters
    max_iter = 10

    def ilqr_step(x0, U):
        """Single iLQR optimization step."""
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=max_iter,
            make_psd=False  # NOTE: make_psd uses eigh which is not CUDA-graphable
        )
        return result[1], result[0]  # controls and states (U, X)

    # Warmup
    warmup_U, warmup_X = ilqr_step(x0, U_init)

    # Static tensors for graph
    static_x0 = x0.clone()
    static_U = U_init.clone()

    # Create CUDA graph
    g = torch.cuda.CUDAGraph()

    # Warmup for graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    with torch.cuda.graph(g):
        static_U_out, static_X_out = ilqr_step(static_x0, static_U)

    # Replay
    g.replay()

    # Verify
    assert static_U_out.shape == (T, n_control)
    assert static_X_out.shape == (T + 1, n_state)
    assert torch.all(torch.isfinite(static_U_out))
    assert torch.all(torch.isfinite(static_X_out))

    print("✓ iLQR CUDA graph test passed for pendulum")


def test_cem_cuda_graph():
    """Test that CEM can be captured in a CUDA graph."""
    device = 'cuda'

    # Setup simple problem
    system = DoubleIntegrator()
    T = 15
    n_control = 1

    # Initial state
    x0 = torch.tensor([1.0, 0.5], device=device, dtype=torch.float32)
    init_controls = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    # Control bounds
    control_low = -10.0
    control_high = 10.0

    # CEM hyperparameters
    hyperparams = {
        'sampling_smoothing': 0.,
        'evolution_smoothing': 0.1,
        'elite_portion': 0.1,
        'max_iter': 5,
        'num_samples': 100
    }

    def cem_step(x0, init_controls):
        """Single CEM optimization."""
        result = torch_optimizers.cem(
            system.cost,
            system.dynamics,
            x0,
            init_controls,
            control_low,
            control_high,
            random_seed=42,
            hyperparams=hyperparams
        )
        return result[1]  # Return optimized controls (U is second return value)

    # Warmup
    warmup_result = cem_step(x0, init_controls)

    # Static inputs
    static_x0 = x0.clone()
    static_init_controls = init_controls.clone()

    # Create CUDA graph
    g = torch.cuda.CUDAGraph()

    # Warmup for graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            cem_step(static_x0, static_init_controls)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    with torch.cuda.graph(g):
        static_output = cem_step(static_x0, static_init_controls)

    # Replay
    g.replay()

    # Verify
    assert static_output.shape == (T, n_control)
    assert torch.all(torch.isfinite(static_output))

    print("✓ CEM CUDA graph test passed")


def test_cuda_graph_multiple_replays():
    """Test that CUDA graph can be replayed multiple times correctly."""
    device = 'cuda'

    system = DoubleIntegrator()
    T = 10
    n_control = 1

    x0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    U_init = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    def ilqr_step(x0, U):
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=5,
            make_psd=False
        )
        return result[1]  # Return U

    # Static tensors
    static_x0 = x0.clone()
    static_U = U_init.clone()

    # Capture graph
    g = torch.cuda.CUDAGraph()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g):
        static_output = ilqr_step(static_x0, static_U)

    # Replay multiple times
    results = []
    for _ in range(5):
        g.replay()
        results.append(static_output.clone())

    # All replays should give same result (deterministic)
    for i in range(1, len(results)):
        assert torch.allclose(results[0], results[i], rtol=1e-5, atol=1e-7)

    print("✓ Multiple replay test passed")


def test_cuda_graph_different_initial_conditions():
    """Test updating static inputs between graph replays."""
    device = 'cuda'

    system = DoubleIntegrator()
    T = 10
    n_control = 1

    # Static tensors that we'll update
    static_x0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    static_U = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    def ilqr_step(x0, U):
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=5,
            make_psd=False
        )
        return result[1]  # Return U

    # Capture graph
    g = torch.cuda.CUDAGraph()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g):
        static_output = ilqr_step(static_x0, static_U)

    # Test with different initial conditions
    test_x0s = [
        torch.tensor([1.0, 0.0], device=device, dtype=torch.float32),
        torch.tensor([2.0, 1.0], device=device, dtype=torch.float32),
        torch.tensor([-1.0, -0.5], device=device, dtype=torch.float32),
    ]

    for test_x0 in test_x0s:
        # Update static input
        static_x0.copy_(test_x0)

        # Replay graph
        g.replay()

        # Verify output is valid
        assert static_output.shape == (T, n_control)
        assert torch.all(torch.isfinite(static_output))

    print("✓ Different initial conditions test passed")


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Running CUDA graph tests...")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")

        test_ilqr_cuda_graph_simple()
        test_ilqr_cuda_graph_pendulum()
        test_cem_cuda_graph()
        test_cuda_graph_multiple_replays()
        test_cuda_graph_different_initial_conditions()

        print("\n✓ All CUDA graph tests passed!")
    else:
        print("CUDA not available, skipping tests")
