#!/usr/bin/env python
"""Benchmark constrained_ilqr vs constrained_ilqr_pt_mpc (PyTorch)."""

import argparse
import time

import torch

from trajax.torch import optimizers


def _make_problem(horizon, n_state, n_ctrl, device, dtype):
  torch.manual_seed(0)
  A = torch.eye(n_state, device=device, dtype=dtype).unsqueeze(0).repeat(
      horizon + 1, 1, 1)
  A = A + 0.01 * torch.randn(horizon + 1, n_state, n_state, device=device,
                             dtype=dtype)
  B = 0.1 * torch.randn(horizon + 1, n_state, n_ctrl, device=device,
                        dtype=dtype)
  Q = torch.eye(n_state, device=device, dtype=dtype).unsqueeze(0).repeat(
      horizon + 1, 1, 1)
  R = torch.eye(n_ctrl, device=device, dtype=dtype).unsqueeze(0).repeat(
      horizon + 1, 1, 1)

  def dynamics(x, u, t):
    return torch.matmul(A[t], x) + torch.matmul(B[t], u)

  def cost(x, u, t):
    return 0.5 * torch.dot(x, torch.matmul(Q[t], x)) + 0.5 * torch.dot(
        u, torch.matmul(R[t], u))

  x0 = torch.randn(n_state, device=device, dtype=dtype)
  U0 = torch.zeros(horizon, n_ctrl, device=device, dtype=dtype)
  return dynamics, cost, x0, U0


def _time_fn(fn, runs, warmup, device):
  for _ in range(warmup):
    fn()
  if device.type == 'cuda':
    torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(runs):
    fn()
  if device.type == 'cuda':
    torch.cuda.synchronize()
  end = time.perf_counter()
  return (end - start) / runs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--horizon', type=int, default=30)
  parser.add_argument('--n-state', type=int, default=8)
  parser.add_argument('--n-ctrl', type=int, default=4)
  parser.add_argument('--maxiter-al', type=int, default=3)
  parser.add_argument('--maxiter-ilqr', type=int, default=20)
  parser.add_argument('--runs', type=int, default=10)
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--device', type=str, default='cuda'
                      if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--skip-mpc', action='store_true')
  args = parser.parse_args()

  device = torch.device(args.device)
  dtype = torch.float32
  dynamics, cost, x0, U0 = _make_problem(
      args.horizon, args.n_state, args.n_ctrl, device, dtype)

  def run_baseline():
    optimizers.constrained_ilqr(
        cost,
        dynamics,
        x0,
        U0.clone(),
        maxiter_al=args.maxiter_al,
        maxiter_ilqr=args.maxiter_ilqr,
        make_psd=True)

  baseline_time = _time_fn(run_baseline, args.runs, args.warmup, device)
  print(f"constrained_ilqr avg: {baseline_time:.6f}s")

  if args.skip_mpc:
    return

  try:
    import mpc  # noqa: F401
  except Exception as exc:
    print(f"mpc.pytorch not available: {exc}")
    return

  def run_mpc():
    optimizers.constrained_ilqr_pt_mpc(
        cost,
        dynamics,
        x0,
        U0.clone(),
        maxiter_al=args.maxiter_al,
        maxiter_ilqr=args.maxiter_ilqr,
        make_psd=True)

  mpc_time = _time_fn(run_mpc, args.runs, args.warmup, device)
  print(f"constrained_ilqr_pt_mpc avg: {mpc_time:.6f}s")


if __name__ == '__main__':
  main()
