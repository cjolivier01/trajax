"""Micro-benchmarks comparing Trajax JAX vs PyTorch backends on GPU.

Examples:
  python benchmarks/compare_backends.py tvlqr_solve --T 100 --n 20 --m 30
  python benchmarks/compare_backends.py tvlqr_rollout --T 100 --n 20 --m 30
  python benchmarks/compare_backends.py ilqr_cartpole --horizon 50
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable, Dict, Tuple

import numpy as onp

import jax
import jax.numpy as jnp
import torch

from trajax import optimizers as jax_optim
from trajax import tvlqr as jax_tvlqr
from trajax.torch import optimizers as torch_optim
from trajax.torch import tvlqr as torch_tvlqr


def _time_jax(fn: Callable[..., Any],
              args: Tuple[Any, ...],
              warmup: int,
              iters: int) -> float:
  for _ in range(warmup):
    out = fn(*args)
    if isinstance(out, tuple):
      for o in out:
        o.block_until_ready()
    else:
      out.block_until_ready()

  start = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
    if isinstance(out, tuple):
      for o in out:
        o.block_until_ready()
    else:
      out.block_until_ready()
  end = time.perf_counter()
  return (end - start) / iters


def _time_torch(fn: Callable[..., Any],
                args: Tuple[Any, ...],
                warmup: int,
                iters: int) -> float:
  for _ in range(warmup):
    out = fn(*args)
    del out
  torch.cuda.synchronize()

  start_evt = torch.cuda.Event(enable_timing=True)
  end_evt = torch.cuda.Event(enable_timing=True)
  start_evt.record()
  for _ in range(iters):
    out = fn(*args)
    del out
  end_evt.record()
  torch.cuda.synchronize()
  ms = start_evt.elapsed_time(end_evt)
  return (ms / 1e3) / iters


def _rand_spd(rng: onp.random.RandomState, shape, eps=1e-2, dtype=onp.float32):
  A = rng.randn(*shape).astype(dtype)
  if len(shape) == 2:
    return (A + A.T) * 0.5 + eps * onp.eye(shape[0], dtype=dtype)
  if len(shape) == 3:
    return (A + onp.swapaxes(A, -1, -2)) * 0.5 + eps * onp.eye(shape[-1], dtype=dtype)
  raise ValueError(shape)

def _count_elems(arrs) -> int:
  total = 0
  for a in arrs:
    if hasattr(a, "size"):
      total += int(a.size)
  return total


def bench_tvlqr_solve(T: int, n: int, m: int, iters: int, warmup: int,
                      dtype: str, torch_compile: bool):
  rng = onp.random.RandomState(0)
  np_dtype = onp.float32 if dtype == "float32" else onp.float64
  j_dtype = jnp.float32 if dtype == "float32" else jnp.float64
  t_dtype = torch.float32 if dtype == "float32" else torch.float64

  Q = _rand_spd(rng, (T + 1, n, n), dtype=np_dtype)
  q = rng.randn(T + 1, n).astype(np_dtype)
  R = _rand_spd(rng, (T, m, m), dtype=np_dtype)
  r = rng.randn(T, m).astype(np_dtype)
  M = rng.randn(T, n, m).astype(np_dtype)
  A = rng.randn(T, n, n).astype(np_dtype)
  B = rng.randn(T, n, m).astype(np_dtype)
  c = rng.randn(T, n).astype(np_dtype)
  elems = _count_elems([Q, q, R, r, M, A, B, c])

  Qj = jax.device_put(jnp.asarray(Q, dtype=j_dtype))
  qj = jax.device_put(jnp.asarray(q, dtype=j_dtype))
  Rj = jax.device_put(jnp.asarray(R, dtype=j_dtype))
  rj = jax.device_put(jnp.asarray(r, dtype=j_dtype))
  Mj = jax.device_put(jnp.asarray(M, dtype=j_dtype))
  Aj = jax.device_put(jnp.asarray(A, dtype=j_dtype))
  Bj = jax.device_put(jnp.asarray(B, dtype=j_dtype))
  cj = jax.device_put(jnp.asarray(c, dtype=j_dtype))

  Qt = torch.as_tensor(Q, device="cuda", dtype=t_dtype)
  qt = torch.as_tensor(q, device="cuda", dtype=t_dtype)
  Rt = torch.as_tensor(R, device="cuda", dtype=t_dtype)
  rt = torch.as_tensor(r, device="cuda", dtype=t_dtype)
  Mt = torch.as_tensor(M, device="cuda", dtype=t_dtype)
  At = torch.as_tensor(A, device="cuda", dtype=t_dtype)
  Bt = torch.as_tensor(B, device="cuda", dtype=t_dtype)
  ct = torch.as_tensor(c, device="cuda", dtype=t_dtype)

  j_fn = jax.jit(jax_tvlqr.tvlqr)
  def t_fn(Q, q, R, r, M, A, B, c):
    # `torch.compile` currently struggles with `torch.linalg.lstsq` and with
    # nested compile regions. The scan+solve path is itself captured via
    # `torch.compile` internally by the higher-order ops.
    if torch_compile:
      return torch_tvlqr.tvlqr(Q, q, R, r, M, A, B, c, solver="solve", use_scan=True)
    return torch_tvlqr.tvlqr(Q, q, R, r, M, A, B, c)

  j_s = _time_jax(j_fn, (Qj, qj, Rj, rj, Mj, Aj, Bj, cj), warmup, iters)
  t_s = _time_torch(t_fn, (Qt, qt, Rt, rt, Mt, At, Bt, ct), warmup, iters)
  return {"jax_s": j_s, "torch_s": t_s, "elements": elems}


def bench_tvlqr_rollout(T: int, n: int, m: int, iters: int, warmup: int,
                        dtype: str, torch_compile: bool):
  rng = onp.random.RandomState(0)
  np_dtype = onp.float32 if dtype == "float32" else onp.float64
  j_dtype = jnp.float32 if dtype == "float32" else jnp.float64
  t_dtype = torch.float32 if dtype == "float32" else torch.float64

  A = rng.randn(T, n, n).astype(np_dtype)
  B = rng.randn(T, n, m).astype(np_dtype)
  c = rng.randn(T, n).astype(np_dtype)
  K = rng.randn(T, m, n).astype(np_dtype)
  k = rng.randn(T, m).astype(np_dtype)
  x0 = rng.randn(n).astype(np_dtype)
  elems = _count_elems([A, B, c, K, k, x0])

  Aj = jax.device_put(jnp.asarray(A, dtype=j_dtype))
  Bj = jax.device_put(jnp.asarray(B, dtype=j_dtype))
  cj = jax.device_put(jnp.asarray(c, dtype=j_dtype))
  Kj = jax.device_put(jnp.asarray(K, dtype=j_dtype))
  kj = jax.device_put(jnp.asarray(k, dtype=j_dtype))
  x0j = jax.device_put(jnp.asarray(x0, dtype=j_dtype))

  At = torch.as_tensor(A, device="cuda", dtype=t_dtype)
  Bt = torch.as_tensor(B, device="cuda", dtype=t_dtype)
  ct = torch.as_tensor(c, device="cuda", dtype=t_dtype)
  Kt = torch.as_tensor(K, device="cuda", dtype=t_dtype)
  kt = torch.as_tensor(k, device="cuda", dtype=t_dtype)
  x0t = torch.as_tensor(x0, device="cuda", dtype=t_dtype)

  def j_fn(K, k, x0, A, B, c):
    return jax_tvlqr.rollout(K, k, x0, A, B, c)[1]

  def t_fn(K, k, x0, A, B, c):
    return torch_tvlqr.rollout(K, k, x0, A, B, c, use_scan=torch_compile)[1]

  j_fn = jax.jit(j_fn)
  # Same as tvlqr_solve: `use_scan=True` is captured internally.

  j_s = _time_jax(j_fn, (Kj, kj, x0j, Aj, Bj, cj), warmup, iters)
  t_s = _time_torch(t_fn, (Kt, kt, x0t, At, Bt, ct), warmup, iters)
  return {"jax_s": j_s, "torch_s": t_s, "elements": elems}


def bench_ilqr_cartpole(horizon: int, iters: int, warmup: int, dtype: str,
                        torch_compile: bool):
  j_dtype = jnp.float32 if dtype == "float32" else jnp.float64
  t_dtype = torch.float32 if dtype == "float32" else torch.float64

  dt = 0.1
  mc, mp, l = 10.0, 1.0, 0.5
  g = 9.81

  def squish_j(u):
    return 5 * jnp.tanh(u)

  def squish_t(u):
    return 5 * torch.tanh(u)

  @jax.jit
  def cartpole_j(state, action, timestep):
    del timestep
    q = state[0:2]
    qd = state[2:]
    s = jnp.sin(q[1])
    c = jnp.cos(q[1])
    H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]],
                  dtype=j_dtype)
    C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]], dtype=j_dtype)
    G = jnp.array([[0.0], [mp * g * l * s]], dtype=j_dtype)
    B = jnp.array([[1.0], [0.0]], dtype=j_dtype)
    CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
    f = jnp.concatenate(
        (qd, jnp.squeeze(-jnp.linalg.solve(H, CqdG))))
    v = jnp.squeeze(jnp.linalg.solve(H, B))
    gg = jnp.concatenate((jnp.zeros(2, dtype=j_dtype), v))
    return f + gg * action

  mc_mp_t = torch.tensor(mc + mp, device="cuda", dtype=t_dtype)
  mp_l_t = torch.tensor(mp * l, device="cuda", dtype=t_dtype)
  mp_l_l_t = torch.tensor(mp * l * l, device="cuda", dtype=t_dtype)
  mp_g_l_t = torch.tensor(mp * g * l, device="cuda", dtype=t_dtype)
  Bm_t = torch.tensor([[1.0], [0.0]], device="cuda", dtype=t_dtype)

  def cartpole_t(state, action, timestep):
    del timestep
    q = state[0:2]
    qd = state[2:]
    s = torch.sin(q[1])
    c = torch.cos(q[1])
    H = torch.stack([
        torch.stack([mc_mp_t, mp_l_t * c]),
        torch.stack([mp_l_t * c, mp_l_l_t]),
    ])
    C = torch.stack([
        torch.stack([torch.zeros((), device=state.device, dtype=state.dtype),
                     -(mp_l_t * qd[1] * s)]),
        torch.stack([torch.zeros((), device=state.device, dtype=state.dtype),
                     torch.zeros((), device=state.device, dtype=state.dtype)]),
    ])
    Gv = torch.stack(
        [torch.zeros((), device=state.device, dtype=state.dtype),
         mp_g_l_t * s]).unsqueeze(-1)
    CqdG = C @ qd.unsqueeze(-1) + Gv
    f = torch.cat([qd, (-torch.linalg.solve(H, CqdG)).squeeze(-1)], dim=0)
    v = torch.linalg.solve(H, Bm_t).squeeze(-1)
    gg = torch.cat([torch.zeros(2, device=state.device, dtype=state.dtype), v], dim=0)
    return f + gg * action

  eq_point_j = jnp.array([0.0, jnp.pi, 0.0, 0.0], dtype=j_dtype)
  eq_point_t = torch.tensor([0.0, float(onp.pi), 0.0, 0.0], device="cuda", dtype=t_dtype)
  two_pi_t = torch.tensor(2.0 * float(onp.pi), device="cuda", dtype=t_dtype)
  t_h = torch.tensor(horizon, device="cuda", dtype=torch.int64)

  def angle_wrap_j(th):
    return th % (2 * jnp.pi)

  def angle_wrap_t(th):
    return torch.remainder(th, two_pi_t)

  def state_wrap_j(s):
    return jnp.array([s[0], angle_wrap_j(s[1]), s[2], s[3]], dtype=j_dtype)

  def state_wrap_t(s):
    return torch.stack([s[0], angle_wrap_t(s[1]), s[2], s[3]])

  def cost_j(x, u, t):
    err = state_wrap_j(x - eq_point_j)
    stage = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    final = 1000 * jnp.dot(err, err)
    return jnp.where(t == horizon, final, stage)

  def dynamics_j(x, u, t):
    return x + dt * cartpole_j(x, squish_j(u), t)

  def cost_t(x, u, t):
    err = state_wrap_t(x - eq_point_t)
    stage = 0.1 * (err @ err) + 0.01 * (u @ u)
    final = 1000 * (err @ err)
    return torch.where(t == t_h, final, stage)

  def dynamics_t(x, u, t):
    return x + dt * cartpole_t(x, squish_t(u), t)

  x0_j = jax.device_put(jnp.array([0.0, 0.2, 0.0, -0.1], dtype=j_dtype))
  U0_j = jax.device_put(jnp.zeros((horizon, 1), dtype=j_dtype))

  x0_t = torch.tensor([0.0, 0.2, 0.0, -0.1], device="cuda", dtype=t_dtype)
  U0_t = torch.zeros((horizon, 1), device="cuda", dtype=t_dtype)
  elems = horizon + 4

  def j_fn(x0):
    return jax_optim.ilqr(cost_j, dynamics_j, x0, U0_j, maxiter=50)[1]

  def t_fn(x0):
    return torch_optim.ilqr(cost_t, dynamics_t, x0, U0_t, maxiter=50)[1]

  j_fn = jax.jit(j_fn)
  if torch_compile:
    t_fn = torch.compile(t_fn, fullgraph=True)
  j_s = _time_jax(j_fn, (x0_j,), warmup, iters)
  t_s = _time_torch(t_fn, (x0_t,), warmup, iters)
  return {"jax_s": j_s, "torch_s": t_s, "elements": elems}


def bench_ilqr_linear(T: int, n: int, m: int, maxiter: int, iters: int,
                      warmup: int, dtype: str):
  rng = onp.random.RandomState(0)
  np_dtype = onp.float32 if dtype == "float32" else onp.float64
  j_dtype = jnp.float32 if dtype == "float32" else jnp.float64
  t_dtype = torch.float32 if dtype == "float32" else torch.float64

  # Keep dynamics stable over long horizons.
  A = (0.95 * onp.eye(n, dtype=np_dtype) +
       0.001 * rng.randn(n, n).astype(np_dtype))
  B = 0.1 * rng.randn(n, m).astype(np_dtype)
  Q = _rand_spd(rng, (n, n), dtype=np_dtype)
  R = _rand_spd(rng, (m, m), dtype=np_dtype)
  x0 = rng.randn(n).astype(np_dtype)
  x_goal = rng.randn(n).astype(np_dtype)
  U0 = (0.2 * rng.randn(T, m)).astype(np_dtype)
  elems = _count_elems([A, B, Q, R, x0, x_goal, U0])

  Aj = jax.device_put(jnp.asarray(A, dtype=j_dtype))
  Bj = jax.device_put(jnp.asarray(B, dtype=j_dtype))
  Qj = jax.device_put(jnp.asarray(Q, dtype=j_dtype))
  Rj = jax.device_put(jnp.asarray(R, dtype=j_dtype))
  x0j = jax.device_put(jnp.asarray(x0, dtype=j_dtype))
  xgj = jax.device_put(jnp.asarray(x_goal, dtype=j_dtype))
  U0j = jax.device_put(jnp.asarray(U0, dtype=j_dtype))

  At = torch.as_tensor(A, device="cuda", dtype=t_dtype)
  Bt = torch.as_tensor(B, device="cuda", dtype=t_dtype)
  Qt = torch.as_tensor(Q, device="cuda", dtype=t_dtype)
  Rt = torch.as_tensor(R, device="cuda", dtype=t_dtype)
  x0t = torch.as_tensor(x0, device="cuda", dtype=t_dtype)
  xgt = torch.as_tensor(x_goal, device="cuda", dtype=t_dtype)
  U0t = torch.as_tensor(U0, device="cuda", dtype=t_dtype)

  def dynamics_j(x, u, t):
    del t
    return Aj @ x + Bj @ u

  def cost_j(x, u, t):
    dx = x - xgj
    stage = 0.5 * (dx @ (Qj @ dx) + u @ (Rj @ u))
    final = 10.0 * 0.5 * (dx @ (Qj @ dx))
    return jnp.where(t == T, final, stage)

  j_fn = jax.jit(lambda x0: jax_optim.ilqr(cost_j, dynamics_j, x0, U0j, maxiter=maxiter)[1])

  def dynamics_t(x, u, t):
    del t
    return At @ x + Bt @ u

  def cost_t(x, u, t):
    dx = x - xgt
    stage = 0.5 * (dx @ (Qt @ dx) + u @ (Rt @ u))
    final = 10.0 * 0.5 * (dx @ (Qt @ dx))
    t_h = torch.tensor(T, device="cuda", dtype=torch.int64)
    return torch.where(t == t_h, final, stage)

  def t_fn(x0):
    return torch_optim.ilqr(cost_t, dynamics_t, x0, U0t, maxiter=maxiter)[1]

  j_s = _time_jax(j_fn, (x0j,), warmup, iters)
  t_s = _time_torch(t_fn, (x0t,), warmup, iters)
  return {"jax_s": j_s, "torch_s": t_s, "elements": elems}


def bench_constrained_ilqr_linear(T: int, n: int, m: int, maxiter_al: int,
                                 maxiter_ilqr: int, iters: int, warmup: int,
                                 dtype: str, torch_compile: bool):
  rng = onp.random.RandomState(0)
  np_dtype = onp.float32 if dtype == "float32" else onp.float64
  j_dtype = jnp.float32 if dtype == "float32" else jnp.float64
  t_dtype = torch.float32 if dtype == "float32" else torch.float64

  A = (0.95 * onp.eye(n, dtype=np_dtype) +
       0.001 * rng.randn(n, n).astype(np_dtype))
  B = 0.01 * rng.randn(n, m).astype(np_dtype)
  Q = _rand_spd(rng, (n, n), dtype=np_dtype)
  R = _rand_spd(rng, (m, m), dtype=np_dtype)
  x0 = rng.randn(n).astype(np_dtype)
  # Start infeasible to force the solver to do work.
  U0 = rng.randn(T, m).astype(np_dtype) * 0.2
  umax = onp.ones(m, dtype=np_dtype) * 0.1
  x_goal = rng.randn(n).astype(np_dtype)
  elems = _count_elems([A, B, Q, R, x0, U0, umax, x_goal])

  Aj = jax.device_put(jnp.asarray(A, dtype=j_dtype))
  Bj = jax.device_put(jnp.asarray(B, dtype=j_dtype))
  Qj = jax.device_put(jnp.asarray(Q, dtype=j_dtype))
  Rj = jax.device_put(jnp.asarray(R, dtype=j_dtype))
  x0j = jax.device_put(jnp.asarray(x0, dtype=j_dtype))
  U0j = jax.device_put(jnp.asarray(U0, dtype=j_dtype))
  umaxj = jax.device_put(jnp.asarray(umax, dtype=j_dtype))
  x_goal_j = jax.device_put(jnp.asarray(x_goal, dtype=j_dtype))

  At = torch.as_tensor(A, device="cuda", dtype=t_dtype)
  Bt = torch.as_tensor(B, device="cuda", dtype=t_dtype)
  Qt = torch.as_tensor(Q, device="cuda", dtype=t_dtype)
  Rt = torch.as_tensor(R, device="cuda", dtype=t_dtype)
  x0t = torch.as_tensor(x0, device="cuda", dtype=t_dtype)
  U0t = torch.as_tensor(U0, device="cuda", dtype=t_dtype)
  umaxt = torch.as_tensor(umax, device="cuda", dtype=t_dtype)
  x_goal_t = torch.as_tensor(x_goal, device="cuda", dtype=t_dtype)

  def dynamics_j(x, u, t):
    del t
    return Aj @ x + Bj @ u

  def cost_j(x, u, t):
    del t
    return 0.5 * (x @ (Qj @ x) + u @ (Rj @ u))

  def eq_j(x, u, t):
    del u
    t_h = jnp.array(T, dtype=t.dtype)
    return jnp.where(t == t_h, x - x_goal_j, jnp.zeros_like(x))

  def ineq_j(x, u, t):
    del x, t
    return jnp.concatenate([u - umaxj, -u - umaxj], axis=0)

  j_fn = jax.jit(
      lambda x0: jax_optim.constrained_ilqr(cost_j,
                                           dynamics_j,
                                           x0,
                                           U0j,
                                           equality_constraint=eq_j,
                                           inequality_constraint=ineq_j,
                                           maxiter_al=maxiter_al,
                                           maxiter_ilqr=maxiter_ilqr)[1])

  def dynamics_t(x, u, t):
    del t
    return At @ x + Bt @ u

  def cost_t(x, u, t):
    del t
    return 0.5 * (x @ (Qt @ x) + u @ (Rt @ u))

  def eq_t(x, u, t):
    del u
    t_h = torch.tensor(T, device="cuda", dtype=torch.int64)
    return torch.where(t == t_h, x - x_goal_t, torch.zeros_like(x))

  def ineq_t(x, u, t):
    del x, t
    return torch.cat([u - umaxt, -u - umaxt], dim=0)

  def t_fn(x0):
    if torch_compile:
      # Use the specialized compile-friendly LQ solver (no torch.func).
      return torch_optim.constrained_ilqr_linear_quadratic_box(
          x0=x0,
          U=U0t,
          A=At,
          B=Bt,
          Q=Qt,
          R=Rt,
          x_goal=x_goal_t,
          umax=umaxt,
          maxiter_al=maxiter_al,
          maxiter_ilqr=maxiter_ilqr,
          final_weight=0.0,  # match `cost_t` in this benchmark (no goal in cost)
      )[1]
    return torch_optim.constrained_ilqr(cost_t,
                                        dynamics_t,
                                        x0,
                                        U0t,
                                        equality_constraint=eq_t,
                                        inequality_constraint=ineq_t,
                                        maxiter_al=maxiter_al,
                                        maxiter_ilqr=maxiter_ilqr)[1]

  j_s = _time_jax(j_fn, (x0j,), warmup, iters)
  t_s = _time_torch(t_fn, (x0t,), warmup, iters)
  return {"jax_s": j_s, "torch_s": t_s, "elements": elems}


def main():
  if not torch.cuda.is_available():
    raise SystemExit("CUDA not available for torch benchmarks.")
  if not any(d.platform == "gpu" for d in jax.devices()):
    raise SystemExit("JAX is not running with a CUDA backend.")

  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest="cmd", required=True)

  def add_common(sp):
    sp.add_argument("--iters", type=int, default=50)
    sp.add_argument("--warmup", type=int, default=10)
    sp.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    sp.add_argument("--torch-compile", action="store_true")
    sp.add_argument(
        "--preset",
        choices=("custom", "big"),
        default="custom",
        help=("Use a predefined large benchmark configuration. "
              "`custom` uses explicit flags; `big` overrides sizes."),
    )

  p1 = sub.add_parser("tvlqr_solve")
  add_common(p1)
  p1.add_argument("--T", type=int, default=100)
  p1.add_argument("--n", type=int, default=20)
  p1.add_argument("--m", type=int, default=30)

  p2 = sub.add_parser("tvlqr_rollout")
  add_common(p2)
  p2.add_argument("--T", type=int, default=100)
  p2.add_argument("--n", type=int, default=20)
  p2.add_argument("--m", type=int, default=30)

  p3 = sub.add_parser("ilqr_cartpole")
  add_common(p3)
  p3.add_argument("--horizon", type=int, default=50)

  p3b = sub.add_parser("ilqr_linear")
  add_common(p3b)
  p3b.add_argument("--T", type=int, default=200)
  p3b.add_argument("--n", type=int, default=32)
  p3b.add_argument("--m", type=int, default=32)
  p3b.add_argument("--maxiter", type=int, default=25)

  p4 = sub.add_parser("constrained_ilqr_linear")
  add_common(p4)
  p4.add_argument("--T", type=int, default=200)
  p4.add_argument("--n", type=int, default=64)
  p4.add_argument("--m", type=int, default=96)
  p4.add_argument("--maxiter-al", type=int, default=3)
  p4.add_argument("--maxiter-ilqr", type=int, default=20)

  args = p.parse_args()
  if args.dtype == "float32":
    torch.set_float32_matmul_precision("high")

  # Apply large presets (picked to keep memory reasonable while increasing
  # element count significantly for the simple TVLQR benchmarks).
  if args.preset == "big":
    if args.cmd in ("tvlqr_solve", "tvlqr_rollout"):
      # ~100x more input elements vs default T=100,n=20,m=30.
      args.T, args.n, args.m = 625, 80, 120
    elif args.cmd == "ilqr_linear":
      # iLQR scales very steeply with (n,m) and stores O(T*(n^2+m^2+nm)) terms,
      # so this "big" preset is primarily compute-heavy but still fits on a
      # single GPU.
      args.T, args.n, args.m = 625, 80, 120
      args.maxiter = min(args.maxiter, 10)
      args.iters = min(args.iters, 3)
      args.warmup = min(args.warmup, 1)
    elif args.cmd == "constrained_ilqr_linear":
      # Constrained iLQR is substantially more expensive; use a larger problem
      # but reduce the outer/inner iterations to keep runtime bounded.
      args.T, args.n, args.m = 200, 64, 96
      args.maxiter_al = min(args.maxiter_al, 1)
      args.maxiter_ilqr = min(args.maxiter_ilqr, 3)
      args.iters = min(args.iters, 3)
      args.warmup = min(args.warmup, 1)
    elif args.cmd == "ilqr_cartpole":
      args.horizon = 200
      args.iters = min(args.iters, 10)
      args.warmup = min(args.warmup, 2)
    else:
      raise AssertionError(args.cmd)
  if args.cmd == "ilqr_cartpole" and args.torch_compile:
    # `torch.compile` can currently struggle with nested `torch.func` transforms
    # (e.g. Hessians inside iLQR) depending on the PyTorch version.
    print("note: disabling --torch-compile for ilqr_cartpole (unsupported)")
    args.torch_compile = False

  if args.cmd == "tvlqr_solve":
    res = bench_tvlqr_solve(args.T, args.n, args.m, args.iters, args.warmup,
                            args.dtype, args.torch_compile)
  elif args.cmd == "tvlqr_rollout":
    res = bench_tvlqr_rollout(args.T, args.n, args.m, args.iters, args.warmup,
                              args.dtype, args.torch_compile)
  elif args.cmd == "ilqr_cartpole":
    res = bench_ilqr_cartpole(args.horizon, args.iters, args.warmup, args.dtype,
                              args.torch_compile)
  elif args.cmd == "ilqr_linear":
    if args.torch_compile:
      print("note: --torch-compile is not supported for ilqr_linear")
    res = bench_ilqr_linear(args.T, args.n, args.m, args.maxiter, args.iters,
                            args.warmup, args.dtype)
  elif args.cmd == "constrained_ilqr_linear":
    res = bench_constrained_ilqr_linear(args.T, args.n, args.m, args.maxiter_al,
                                        args.maxiter_ilqr, args.iters,
                                        args.warmup, args.dtype,
                                        args.torch_compile)
  else:
    raise AssertionError(args.cmd)

  speedup = res["jax_s"] / res["torch_s"]
  print(f"{args.cmd}: dtype={args.dtype}")
  if "elements" in res:
    print(f"  input elements: {res['elements']:,d}")
  print(f"  jax   : {res['jax_s'] * 1e3:.3f} ms/iter")
  print(f"  torch : {res['torch_s'] * 1e3:.3f} ms/iter")
  print(f"  jax/torch speedup: {speedup:.2f}x")


if __name__ == "__main__":
  main()
