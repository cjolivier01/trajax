"""Performance benchmarks comparing JAX and PyTorch implementations."""

import time
import numpy as np
import torch
import jax.numpy as jnp
import jax
from functools import partial

# Enable 64-bit precision for JAX
try:
    from jax.config import config
    config.update('jax_enable_x64', True)
except:
    jax.config.update('jax_enable_x64', True)

# Import JAX versions
from trajax import optimizers as jax_optimizers
from trajax.integrators import euler as jax_euler, rk4 as jax_rk4

# Import PyTorch versions
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers
from trajax_torch.integrators import euler as torch_euler, rk4 as torch_rk4


class BenchmarkRunner:
    """Run performance benchmarks."""

    def __init__(self, device='cpu', warmup=3, runs=10):
        self.device = device
        self.warmup = warmup
        self.runs = runs
        self.results = []

    def time_function(self, fn, *args, **kwargs):
        """Time a function with warmup."""
        # Warmup
        for _ in range(self.warmup):
            result = fn(*args, **kwargs)

        # Actual timing
        times = []
        for _ in range(self.runs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'result': result
        }

    def benchmark_lqr(self, horizon=50, state_dim=4, control_dim=2):
        """Benchmark LQR problem."""
        print(f"\n{'='*60}")
        print(f"LQR Benchmark: T={horizon}, n={state_dim}, m={control_dim}")
        print(f"{'='*60}")

        # JAX setup
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

        # PyTorch setup
        def cost_torch(x, u, t):
            Q = 1.0
            R = 0.1
            return 0.5 * Q * torch.sum(x**2) + 0.5 * R * torch.sum(u**2)

        def dynamics_torch(x, u, t):
            A = torch.eye(state_dim, device=self.device, dtype=torch.float64) * 1.05
            B = torch.ones((state_dim, control_dim), device=self.device, dtype=torch.float64) * 0.1
            return torch.matmul(A, x) + torch.matmul(B, u)

        x0_torch = torch.ones(state_dim, device=self.device, dtype=torch.float64)
        U0_torch = torch.zeros((horizon, control_dim), device=self.device, dtype=torch.float64)

        # Benchmark JAX iLQR
        print("\nJAX iLQR:")
        jax_result = self.time_function(
            jax_optimizers.ilqr,
            cost_jax, dynamics_jax, x0_jax, U0_jax,
            maxiter=50, make_psd=False
        )
        X_jax, U_jax, obj_jax, _, _, _, iters_jax = jax_result['result']
        print(f"  Time: {jax_result['mean']*1000:.2f} ± {jax_result['std']*1000:.2f} ms")
        print(f"  Iterations: {iters_jax}, Objective: {obj_jax:.6f}")

        # Benchmark PyTorch iLQR
        print("\nPyTorch iLQR:")
        torch_result = self.time_function(
            torch_optimizers.ilqr,
            cost_torch, dynamics_torch, x0_torch, U0_torch,
            maxiter=50, make_psd=False
        )
        X_torch, U_torch, obj_torch, _, _, _, iters_torch = torch_result['result']
        print(f"  Time: {torch_result['mean']*1000:.2f} ± {torch_result['std']*1000:.2f} ms")
        print(f"  Iterations: {iters_torch}, Objective: {obj_torch:.6f}")

        # Speedup
        speedup = jax_result['mean'] / torch_result['mean']
        print(f"\nSpeedup: {speedup:.2f}x ({'PyTorch faster' if speedup > 1 else 'JAX faster'})")

        self.results.append({
            'name': f'LQR_T{horizon}_n{state_dim}_m{control_dim}',
            'jax_time': jax_result['mean'],
            'torch_time': torch_result['mean'],
            'speedup': speedup,
            'jax_iters': iters_jax,
            'torch_iters': iters_torch
        })

    def benchmark_pendulum(self, horizon=100):
        """Benchmark pendulum swingup."""
        print(f"\n{'='*60}")
        print(f"Pendulum Swingup: T={horizon}")
        print(f"{'='*60}")

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
            ], device=state.device, dtype=state.dtype)

        def cost_torch(state, action, t):
            Q, R = 10.0, 1.0
            theta, theta_dot = state[0], state[1]
            goal_theta = torch.tensor(np.pi, dtype=state.dtype, device=state.device)
            theta_err = theta - goal_theta
            return 0.5 * Q * (theta_err**2 + theta_dot**2) + 0.5 * R * torch.squeeze(action)**2

        dynamics_torch = torch_rk4(pendulum_torch, dt=0.01)
        x0_torch = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        U0_torch = torch.zeros((horizon, 1), device=self.device, dtype=torch.float64)

        # Benchmark JAX iLQR
        print("\nJAX iLQR:")
        jax_result = self.time_function(
            jax_optimizers.ilqr,
            cost_jax, dynamics_jax, x0_jax, U0_jax,
            maxiter=50, make_psd=False
        )
        X_jax, U_jax, obj_jax, _, _, _, iters_jax = jax_result['result']
        print(f"  Time: {jax_result['mean']*1000:.2f} ± {jax_result['std']*1000:.2f} ms")
        print(f"  Iterations: {iters_jax}, Objective: {obj_jax:.6f}")

        # Benchmark PyTorch iLQR
        print("\nPyTorch iLQR:")
        torch_result = self.time_function(
            torch_optimizers.ilqr,
            cost_torch, dynamics_torch, x0_torch, U0_torch,
            maxiter=50, make_psd=False
        )
        X_torch, U_torch, obj_torch, _, _, _, iters_torch = torch_result['result']
        print(f"  Time: {torch_result['mean']*1000:.2f} ± {torch_result['std']*1000:.2f} ms")
        print(f"  Iterations: {iters_torch}, Objective: {obj_torch:.6f}")

        # Speedup
        speedup = jax_result['mean'] / torch_result['mean']
        print(f"\nSpeedup: {speedup:.2f}x ({'PyTorch faster' if speedup > 1 else 'JAX faster'})")

        self.results.append({
            'name': f'Pendulum_T{horizon}',
            'jax_time': jax_result['mean'],
            'torch_time': torch_result['mean'],
            'speedup': speedup,
            'jax_iters': iters_jax,
            'torch_iters': iters_torch
        })

    def benchmark_rollout(self, horizon=1000, state_dim=10):
        """Benchmark simple rollout."""
        print(f"\n{'='*60}")
        print(f"Rollout Benchmark: T={horizon}, n={state_dim}")
        print(f"{'='*60}")

        # JAX
        def dynamics_jax(x, u, t):
            return 1.05 * x + 0.1 * u

        U_jax = jnp.ones((horizon, state_dim))
        x0_jax = jnp.zeros(state_dim)

        # PyTorch
        def dynamics_torch(x, u, t):
            return 1.05 * x + 0.1 * u

        U_torch = torch.ones((horizon, state_dim), device=self.device, dtype=torch.float64)
        x0_torch = torch.zeros(state_dim, device=self.device, dtype=torch.float64)

        # Benchmark JAX
        print("\nJAX rollout:")
        jax_result = self.time_function(jax_optimizers.rollout, dynamics_jax, U_jax, x0_jax)
        print(f"  Time: {jax_result['mean']*1000:.2f} ± {jax_result['std']*1000:.2f} ms")

        # Benchmark PyTorch
        print("\nPyTorch rollout:")
        torch_result = self.time_function(torch_optimizers.rollout, dynamics_torch, U_torch, x0_torch)
        print(f"  Time: {torch_result['mean']*1000:.2f} ± {torch_result['std']*1000:.2f} ms")

        speedup = jax_result['mean'] / torch_result['mean']
        print(f"\nSpeedup: {speedup:.2f}x ({'PyTorch faster' if speedup > 1 else 'JAX faster'})")

        self.results.append({
            'name': f'Rollout_T{horizon}_n{state_dim}',
            'jax_time': jax_result['mean'],
            'torch_time': torch_result['mean'],
            'speedup': speedup,
            'jax_iters': 'N/A',
            'torch_iters': 'N/A'
        })

    def print_summary(self):
        """Print summary table."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY - Device: {self.device.upper()}")
        print(f"{'='*80}")
        print(f"{'Benchmark':<30} {'JAX (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
        print(f"{'-'*80}")

        for result in self.results:
            jax_ms = result['jax_time'] * 1000
            torch_ms = result['torch_time'] * 1000
            speedup = result['speedup']
            speedup_str = f"{speedup:.2f}x"
            if speedup > 1:
                speedup_str += " (PT)"
            else:
                speedup_str += " (JAX)"

            print(f"{result['name']:<30} {jax_ms:<15.2f} {torch_ms:<15.2f} {speedup_str:<10}")

        print(f"{'='*80}")

        # Overall statistics
        speedups = [r['speedup'] for r in self.results]
        avg_speedup = np.mean(speedups)
        torch_faster_count = sum(1 for s in speedups if s > 1)

        print(f"\nOverall Statistics:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  PyTorch faster: {torch_faster_count}/{len(speedups)} benchmarks")
        print(f"  JAX faster: {len(speedups) - torch_faster_count}/{len(speedups)} benchmarks")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("JAX vs PyTorch Performance Benchmarks")
    print("="*80)

    # CPU Benchmarks
    print("\n" + "="*80)
    print("CPU BENCHMARKS")
    print("="*80)

    runner_cpu = BenchmarkRunner(device='cpu', warmup=2, runs=5)

    # Various problem sizes
    runner_cpu.benchmark_lqr(horizon=20, state_dim=2, control_dim=1)
    runner_cpu.benchmark_lqr(horizon=50, state_dim=4, control_dim=2)
    runner_cpu.benchmark_lqr(horizon=100, state_dim=8, control_dim=4)
    runner_cpu.benchmark_pendulum(horizon=50)
    runner_cpu.benchmark_pendulum(horizon=100)
    runner_cpu.benchmark_rollout(horizon=500, state_dim=10)
    runner_cpu.benchmark_rollout(horizon=1000, state_dim=20)

    runner_cpu.print_summary()

    # GPU Benchmarks if available
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU BENCHMARKS")
        print("="*80)

        runner_gpu = BenchmarkRunner(device='cuda', warmup=3, runs=10)

        runner_gpu.benchmark_lqr(horizon=20, state_dim=2, control_dim=1)
        runner_gpu.benchmark_lqr(horizon=50, state_dim=4, control_dim=2)
        runner_gpu.benchmark_lqr(horizon=100, state_dim=8, control_dim=4)
        runner_gpu.benchmark_lqr(horizon=200, state_dim=16, control_dim=8)
        runner_gpu.benchmark_pendulum(horizon=50)
        runner_gpu.benchmark_pendulum(horizon=100)
        runner_gpu.benchmark_rollout(horizon=500, state_dim=10)
        runner_gpu.benchmark_rollout(horizon=1000, state_dim=20)
        runner_gpu.benchmark_rollout(horizon=2000, state_dim=50)

        runner_gpu.print_summary()
    else:
        print("\n⚠️  GPU not available, skipping GPU benchmarks")

    print("\n✓ Benchmarks complete!\n")
