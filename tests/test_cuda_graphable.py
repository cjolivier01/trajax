"""Test that PyTorch trajax optimizers are CUDA-graphable."""

import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers


class DoubleIntegrator:
    """Simple double integrator for testing (discrete-time)."""

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


def test_ilqr_cuda_graph():
    """Test that iLQR can be captured in a CUDA graph."""
    print("Testing iLQR CUDA graph capture...")

    device = 'cuda'
    system = DoubleIntegrator()
    T = 15
    n_control = 1

    # Static tensors for graph
    static_x0 = torch.tensor([0.5, 0.0], device=device, dtype=torch.float32)
    static_U = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    def ilqr_step(x0, U):
        """Single iLQR optimization."""
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=5,
            make_psd=False  # make_psd=True uses eigh which is not graphable
        )
        return result[1]  # Return U

    # Warmup runs (not in graph)
    print("  Running warmup...")
    for _ in range(3):
        _ = ilqr_step(static_x0, static_U)

    # Create CUDA graph
    print("  Capturing graph...")
    g = torch.cuda.CUDAGraph()

    # Warmup in side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    # Capture the graph
    with torch.cuda.graph(g):
        static_output = ilqr_step(static_x0, static_U)

    print("  Graph captured successfully!")

    # Replay the graph
    print("  Replaying graph...")
    g.replay()
    torch.cuda.synchronize()

    # Verify output
    assert static_output.shape == (T, n_control)
    assert torch.all(torch.isfinite(static_output))

    print("  ✓ iLQR CUDA graph test passed\n")
    return True


def test_constrained_ilqr_cuda_graph():
    """Test that constrained iLQR can be captured in a CUDA graph."""
    print("Testing constrained iLQR CUDA graph capture...")

    device = 'cuda'
    system = DoubleIntegrator()
    T = 10
    n_control = 1

    # Static tensors
    static_x0 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    static_U = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    # Inequality constraint: limit control to [-3, 3]
    def inequality_constraint(x, u, t):
        return torch.tensor([
            u[0] - 3.0,   # u <= 3
            -u[0] - 3.0   # u >= -3
        ], device=device, dtype=torch.float32)

    def constrained_ilqr_step(x0, U):
        """Single constrained iLQR optimization."""
        result = torch_optimizers.constrained_ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            inequality_constraint=inequality_constraint,
            maxiter_al=3,
            maxiter_ilqr=5,
            constraints_threshold=1e-2,
            make_psd=False
        )
        return result[1]  # Return U

    # Warmup runs
    print("  Running warmup...")
    for _ in range(3):
        _ = constrained_ilqr_step(static_x0, static_U)

    # Create CUDA graph
    print("  Capturing graph...")
    g = torch.cuda.CUDAGraph()

    # Warmup in side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            constrained_ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    # Capture the graph
    with torch.cuda.graph(g):
        static_output = constrained_ilqr_step(static_x0, static_U)

    print("  Graph captured successfully!")

    # Replay the graph multiple times
    print("  Replaying graph 10 times...")
    for i in range(10):
        g.replay()
    torch.cuda.synchronize()

    # Verify output
    assert static_output.shape == (T, n_control)
    assert torch.all(torch.isfinite(static_output))

    print("  ✓ Constrained iLQR CUDA graph test passed\n")
    return True


def test_cuda_graph_with_different_inputs():
    """Test updating inputs between graph replays."""
    print("Testing CUDA graph with different inputs...")

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
        return result[1]

    # Capture graph
    print("  Capturing graph...")
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
    print("  Testing with different initial conditions...")
    test_x0s = [
        torch.tensor([0.5, 0.0], device=device, dtype=torch.float32),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float32),
        torch.tensor([-0.5, -0.3], device=device, dtype=torch.float32),
    ]

    results = []
    for i, test_x0 in enumerate(test_x0s):
        # Update static input
        static_x0.copy_(test_x0)

        # Replay graph
        g.replay()
        torch.cuda.synchronize()

        # Store result
        results.append(static_output.clone())

        # Verify output is valid
        assert static_output.shape == (T, n_control)
        assert torch.all(torch.isfinite(static_output))
        print(f"    Test {i+1}: x0={test_x0.tolist()} -> max_u={torch.max(torch.abs(static_output)).item():.3f}")

    # Verify we got different outputs for different inputs
    assert not torch.allclose(results[0], results[1])
    assert not torch.allclose(results[1], results[2])

    print("  ✓ Different inputs test passed\n")
    return True


def benchmark_cuda_graph_speedup():
    """Benchmark the speedup from CUDA graphs."""
    print("Benchmarking CUDA graph speedup...")

    device = 'cuda'
    system = DoubleIntegrator()
    T = 20
    n_control = 1

    static_x0 = torch.tensor([0.5, 0.0], device=device, dtype=torch.float32)
    static_U = torch.zeros((T, n_control), device=device, dtype=torch.float32)

    def ilqr_step(x0, U):
        result = torch_optimizers.ilqr(
            system.cost,
            system.dynamics,
            x0,
            U,
            maxiter=10,
            make_psd=False
        )
        return result[1]

    # Warmup
    for _ in range(10):
        _ = ilqr_step(static_x0, static_U)
    torch.cuda.synchronize()

    # Benchmark without CUDA graph
    print("  Benchmarking without graph...")
    n_iters = 100

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        _ = ilqr_step(static_x0, static_U)
    end_event.record()
    torch.cuda.synchronize()

    time_no_graph = start_event.elapsed_time(end_event) / n_iters
    print(f"    Time per iteration: {time_no_graph:.3f} ms")

    # Capture graph
    print("  Capturing graph...")
    g = torch.cuda.CUDAGraph()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ilqr_step(static_x0, static_U)
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g):
        static_output = ilqr_step(static_x0, static_U)

    # Benchmark with CUDA graph
    print("  Benchmarking with graph...")
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_iters):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    time_with_graph = start_event.elapsed_time(end_event) / n_iters
    print(f"    Time per iteration: {time_with_graph:.3f} ms")

    speedup = time_no_graph / time_with_graph
    print(f"  Speedup: {speedup:.2f}x\n")

    return True


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    print("=" * 70)
    print("CUDA Graph Tests for PyTorch Trajax")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()

    try:
        test_ilqr_cuda_graph()
        test_constrained_ilqr_cuda_graph()
        test_cuda_graph_with_different_inputs()
        benchmark_cuda_graph_speedup()

        print("=" * 70)
        print("✓ All CUDA graph tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
