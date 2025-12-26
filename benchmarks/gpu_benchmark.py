"""GPU-only performance benchmarks comparing JAX and PyTorch implementations."""

import time
import numpy as np
import torch
import jax.numpy as jnp
import jax

# Enable 64-bit precision for JAX
try:
    jax.config.update('jax_enable_x64', True)
except:
    pass

# Import JAX versions
from trajax import optimizers as jax_optimizers
from trajax.integrators import rk4 as jax_rk4

# Import PyTorch versions
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers
from trajax_torch.integrators import rk4 as torch_rk4


def time_function(fn, *args, warmup=2, runs=5, **kwargs):
    """Time a function with warmup."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)

    # Sync GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Actual timing
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'result': result
    }


def benchmark_lqr(horizon, state_dim, control_dim):
    """Benchmark LQR problem."""
    print(f"\nLQR: T={horizon}, n={state_dim}, m={control_dim}")
    print("-" * 60)

    # JAX setup (runs on GPU by default if available)
    def cost_jax(x, u, t):
        Q = 1.0
        R = 0.1
        return 0.5 * Q * jnp.sum(x**2) + 0.5 * R * jnp.sum(u**2)

    def dynamics_jax(x, u, t):
        A = jnp.eye(state_dim) * 1.05
        B = jnp.ones((state_dim, control_dim)) * 0.1
        return jnp.matmul(A, x) + jnp.matmul(B, u)

    x0_jax = jnp.ones(state_dim)
    U0_jax = jnp.zeros((horizon, control_dim))

    # PyTorch setup (GPU)
    def cost_torch(x, u, t):
        Q = 1.0
        R = 0.1
        return 0.5 * Q * torch.sum(x**2) + 0.5 * R * torch.sum(u**2)

    def dynamics_torch(x, u, t):
        A = torch.eye(state_dim, device='cuda', dtype=torch.float64) * 1.05
        B = torch.ones((state_dim, control_dim), device='cuda', dtype=torch.float64) * 0.1
        return torch.matmul(A, x) + torch.matmul(B, u)

    x0_torch = torch.ones(state_dim, device='cuda', dtype=torch.float64)
    U0_torch = torch.zeros((horizon, control_dim), device='cuda', dtype=torch.float64)

    # Benchmark JAX
    print("  JAX iLQR...", end=' ', flush=True)
    jax_result = time_function(
        jax_optimizers.ilqr,
        cost_jax, dynamics_jax, x0_jax, U0_jax,
        maxiter=50, make_psd=False,
        warmup=2, runs=5
    )
    _, _, obj_jax, _, _, _, iters_jax = jax_result['result']
    print(f"{jax_result['mean']*1000:.2f}ms (iters={iters_jax}, obj={obj_jax:.4f})")

    # Benchmark PyTorch
    print("  PyTorch iLQR...", end=' ', flush=True)
    torch_result = time_function(
        torch_optimizers.ilqr,
        cost_torch, dynamics_torch, x0_torch, U0_torch,
        maxiter=50, make_psd=False,
        warmup=2, runs=5
    )
    _, _, obj_torch, _, _, _, iters_torch = torch_result['result']
    print(f"{torch_result['mean']*1000:.2f}ms (iters={iters_torch}, obj={obj_torch:.4f})")

    speedup = jax_result['mean'] / torch_result['mean']
    winner = "PyTorch" if speedup > 1 else "JAX"
    print(f"  → Speedup: {abs(speedup):.2f}x ({winner} faster)")

    return {
        'name': f'LQR_T{horizon}_n{state_dim}_m{control_dim}',
        'jax_time': jax_result['mean'],
        'torch_time': torch_result['mean'],
        'speedup': speedup
    }


def benchmark_pendulum(horizon):
    """Benchmark pendulum swingup."""
    print(f"\nPendulum: T={horizon}")
    print("-" * 60)

    # JAX pendulum
    def pendulum_jax(state, action, t):
        theta, theta_dot = state
        m, l, g = 1.0, 1.0, 9.81
        return jnp.array([
            theta_dot,
            (jnp.squeeze(action) - m * g * l * jnp.sin(theta)) / (m * l * l)
        ])

    def cost_jax(state, action, t):
        Q, R = 10.0, 1.0
        theta, theta_dot = state
        goal_theta = jnp.pi
        theta_err = theta - goal_theta
        return 0.5 * Q * (theta_err**2 + theta_dot**2) + 0.5 * R * jnp.squeeze(action)**2

    dynamics_jax = jax_rk4(pendulum_jax, dt=0.01)
    x0_jax = jnp.array([0.0, 0.0])
    U0_jax = jnp.zeros((horizon, 1))

    # PyTorch pendulum
    def pendulum_torch(state, action, t):
        theta, theta_dot = state[0], state[1]
        m, l, g = 1.0, 1.0, 9.81
        return torch.tensor([
            theta_dot,
            (torch.squeeze(action) - m * g * l * torch.sin(theta)) / (m * l * l)
        ], device='cuda', dtype=torch.float64)

    def cost_torch(state, action, t):
        Q, R = 10.0, 1.0
        theta, theta_dot = state[0], state[1]
        goal_theta = torch.tensor(np.pi, dtype=torch.float64, device='cuda')
        theta_err = theta - goal_theta
        return 0.5 * Q * (theta_err**2 + theta_dot**2) + 0.5 * R * torch.squeeze(action)**2

    dynamics_torch = torch_rk4(pendulum_torch, dt=0.01)
    x0_torch = torch.tensor([0.0, 0.0], device='cuda', dtype=torch.float64)
    U0_torch = torch.zeros((horizon, 1), device='cuda', dtype=torch.float64)

    # Benchmark JAX
    print("  JAX iLQR...", end=' ', flush=True)
    jax_result = time_function(
        jax_optimizers.ilqr,
        cost_jax, dynamics_jax, x0_jax, U0_jax,
        maxiter=50, make_psd=False,
        warmup=2, runs=5
    )
    _, _, obj_jax, _, _, _, iters_jax = jax_result['result']
    print(f"{jax_result['mean']*1000:.2f}ms (iters={iters_jax}, obj={obj_jax:.4f})")

    # Benchmark PyTorch
    print("  PyTorch iLQR...", end=' ', flush=True)
    torch_result = time_function(
        torch_optimizers.ilqr,
        cost_torch, dynamics_torch, x0_torch, U0_torch,
        maxiter=50, make_psd=False,
        warmup=2, runs=5
    )
    _, _, obj_torch, _, _, _, iters_torch = torch_result['result']
    print(f"{torch_result['mean']*1000:.2f}ms (iters={iters_torch}, obj={obj_torch:.4f})")

    speedup = jax_result['mean'] / torch_result['mean']
    winner = "PyTorch" if speedup > 1 else "JAX"
    print(f"  → Speedup: {abs(speedup):.2f}x ({winner} faster)")

    return {
        'name': f'Pendulum_T{horizon}',
        'jax_time': jax_result['mean'],
        'torch_time': torch_result['mean'],
        'speedup': speedup
    }


def benchmark_rollout(horizon, state_dim):
    """Benchmark rollout."""
    print(f"\nRollout: T={horizon}, n={state_dim}")
    print("-" * 60)

    # JAX
    def dynamics_jax(x, u, t):
        return 1.05 * x + 0.1 * u

    U_jax = jnp.ones((horizon, state_dim))
    x0_jax = jnp.zeros(state_dim)

    # PyTorch
    def dynamics_torch(x, u, t):
        return 1.05 * x + 0.1 * u

    U_torch = torch.ones((horizon, state_dim), device='cuda', dtype=torch.float64)
    x0_torch = torch.zeros(state_dim, device='cuda', dtype=torch.float64)

    # Benchmark JAX
    print("  JAX rollout...", end=' ', flush=True)
    jax_result = time_function(jax_optimizers.rollout, dynamics_jax, U_jax, x0_jax, warmup=3, runs=10)
    print(f"{jax_result['mean']*1000:.2f}ms")

    # Benchmark PyTorch
    print("  PyTorch rollout...", end=' ', flush=True)
    torch_result = time_function(torch_optimizers.rollout, dynamics_torch, U_torch, x0_torch, warmup=3, runs=10)
    print(f"{torch_result['mean']*1000:.2f}ms")

    speedup = jax_result['mean'] / torch_result['mean']
    winner = "PyTorch" if speedup > 1 else "JAX"
    print(f"  → Speedup: {abs(speedup):.2f}x ({winner} faster)")

    return {
        'name': f'Rollout_T{horizon}_n{state_dim}',
        'jax_time': jax_result['mean'],
        'torch_time': torch_result['mean'],
        'speedup': speedup
    }


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: GPU not available!")
        exit(1)

    print("\n" + "="*80)
    print("GPU Performance Benchmarks: JAX (CPU) vs PyTorch (GPU)")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Note: JAX running on CPU due to GPU solver issues")
    print("="*80)

    results = []

    # Small problems
    print("\n### SMALL PROBLEMS ###")
    results.append(benchmark_lqr(20, 2, 1))
    results.append(benchmark_lqr(50, 4, 2))
    results.append(benchmark_pendulum(50))

    # Medium problems
    print("\n### MEDIUM PROBLEMS ###")
    results.append(benchmark_lqr(100, 8, 4))
    results.append(benchmark_lqr(200, 10, 5))
    results.append(benchmark_pendulum(100))
    results.append(benchmark_pendulum(200))

    # Large problems
    print("\n### LARGE PROBLEMS ###")
    results.append(benchmark_lqr(500, 16, 8))
    results.append(benchmark_rollout(1000, 20))
    results.append(benchmark_rollout(2000, 50))

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<35} {'JAX (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<15}")
    print("-"*80)

    for r in results:
        jax_ms = r['jax_time'] * 1000
        torch_ms = r['torch_time'] * 1000
        speedup = r['speedup']
        if speedup > 1:
            speedup_str = f"{speedup:.2f}x (PT)"
        else:
            speedup_str = f"{1/speedup:.2f}x (JAX)"
        print(f"{r['name']:<35} {jax_ms:<15.2f} {torch_ms:<15.2f} {speedup_str:<15}")

    print("="*80)

    # Statistics
    speedups = [r['speedup'] for r in results]
    avg_speedup = np.mean(speedups)
    torch_wins = sum(1 for s in speedups if s > 1)
    jax_wins = len(speedups) - torch_wins

    print(f"\nStatistics:")
    print(f"  PyTorch faster: {torch_wins}/{len(results)} benchmarks")
    print(f"  JAX faster: {jax_wins}/{len(results)} benchmarks")
    print(f"  Average speedup: {avg_speedup:.2f}x ({'PyTorch' if avg_speedup > 1 else 'JAX'} on average)")

    print("\n✓ GPU Benchmarks complete!\n")
