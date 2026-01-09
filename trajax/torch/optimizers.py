"""PyTorch implementations of Trajax optimizer building blocks (GPU-first).

Design goals:
  - All computations stay in torch on the GPU (no NumPy/SciPy/JAX in this path).
  - Shapes and APIs mirror the JAX backend where practical.
  - Functions are compatible with `torch.compile` and CUDA graph capture when
    called with fixed shapes and fixed `maxiter`.
"""

from __future__ import annotations

import os
import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from .tvlqr import rollout as tvlqr_rollout
from .tvlqr import tvlqr

from torch import _higher_order_ops as _ho

# Convenience routine to pad zeros for vectorization purposes.
pad = lambda A: torch.cat(  # noqa: E731
    [A,
     torch.zeros((1,) + A.shape[1:], device=A.device, dtype=A.dtype)],
    dim=0,
)


def _require_cuda(*tensors: torch.Tensor):
  allow_cpu = os.environ.get("TRAJAX_TORCH_ALLOW_CPU", "0") not in ("", "0")
  if allow_cpu:
    return
  for t in tensors:
    if isinstance(t, torch.Tensor) and t.device.type != "cuda":
      raise ValueError(
          "Trajax torch backend requires CUDA tensors. Set "
          "`TRAJAX_TORCH_ALLOW_CPU=1` to override for debugging.")


def vectorize(fun: Callable[..., Any], argnums: int = 3) -> Callable[..., Any]:
  """Returns a vectorized version of the input function (via `torch.vmap`)."""

  def vfun(*args):
    in_dims = (0,) * argnums + (None,) * (len(args) - argnums)
    try:
      return torch.vmap(fun, in_dims=in_dims)(*args)
    except Exception:
      # Fallback for user functions that are not vmap-safe (e.g. they index into
      # tensors using `t.item()` or otherwise trigger vmap limitations).
      batch = args[0].shape[0]

      def slice_arg(a, i, batched):
        return a[i] if batched else a

      outs = []
      for i in range(batch):
        sliced = [
            slice_arg(a, i, j < argnums) for j, a in enumerate(args)
        ]
        outs.append(fun(*sliced))

      def stack_out(o_list):
        if isinstance(o_list[0], torch.Tensor):
          return torch.stack(o_list, dim=0)
        return tuple(
            stack_out([o[k] for o in o_list]) for k in range(len(o_list[0])))

      return stack_out(outs)

  return vfun


def linearize(fun: Callable[..., torch.Tensor],
              argnums: int = 3) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
  """Vectorized gradient/jacobian operator wrt (x, u)."""
  jac_x = torch.func.jacrev(fun, argnums=0)
  jac_u = torch.func.jacrev(fun, argnums=1)

  def linearizer(*args):
    return jac_x(*args), jac_u(*args)

  return vectorize(linearizer, argnums=argnums)


def quadratize(fun: Callable[..., torch.Tensor],
               argnums: int = 3) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
  """Vectorized Hessian operator for a scalar function (wrt x, u, and cross term)."""
  hess_x = torch.func.hessian(fun, argnums=0)
  hess_u = torch.func.hessian(fun, argnums=1)
  hess_x_u = torch.func.jacrev(torch.func.grad(fun, argnums=0), argnums=1)

  def quadratizer(*args):
    return hess_x(*args), hess_u(*args), hess_x_u(*args)

  return vectorize(quadratizer, argnums=argnums)


def rollout(dynamics: Callable[..., torch.Tensor], U: torch.Tensor,
            x0: torch.Tensor, dynamics_args: Sequence[Any] = ()) -> torch.Tensor:
  """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0."""
  return _rollout(dynamics, U, x0, *dynamics_args)


def _rollout(dynamics: Callable[..., torch.Tensor], U: torch.Tensor, x0: torch.Tensor,
             *dynamics_args) -> torch.Tensor:
  _require_cuda(U, x0)
  T = U.shape[0]
  timesteps = torch.arange(T, device=U.device, dtype=torch.int64)
  n = x0.shape[0]
  X = torch.empty((T + 1, n), device=x0.device, dtype=x0.dtype)
  X[0] = x0
  for t in range(T):
    X[t + 1] = dynamics(X[t], U[t], timesteps[t], *dynamics_args)
  return X


def evaluate(cost: Callable[..., torch.Tensor], X: torch.Tensor, U: torch.Tensor,
             cost_args: Sequence[Any] = ()) -> torch.Tensor:
  """Evaluates cost(x, u, t) along a trajectory."""
  _require_cuda(X, U)
  timesteps = torch.arange(X.shape[0], device=X.device, dtype=torch.int64)
  return vectorize(cost)(X, U, timesteps, *cost_args)


def objective(cost: Callable[..., torch.Tensor], dynamics: Callable[..., torch.Tensor],
              U: torch.Tensor, x0: torch.Tensor, cost_args: Sequence[Any] = (),
              dynamics_args: Sequence[Any] = ()) -> torch.Tensor:
  """Evaluates total cost for a control sequence."""
  X = _rollout(dynamics, U, x0, *dynamics_args)
  return evaluate(cost, X, pad(U), cost_args=cost_args).sum()


def adjoint(A: torch.Tensor, B: torch.Tensor, q: torch.Tensor,
            r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Solve adjoint equations (backward recursion)."""
  _require_cuda(A, B, q, r)
  T = q.shape[0] - 1
  n = q.shape[1]
  m = r.shape[1]
  g = torch.empty((T, m), device=q.device, dtype=q.dtype)
  adj = torch.empty((T, n), device=q.device, dtype=q.dtype)

  p = q[T]
  for t in range(T - 1, -1, -1):
    adj[t] = p
    g[t] = r[t] + B[t].transpose(-1, -2) @ p
    p = A[t].transpose(-1, -2) @ p + q[t]

  return g, adj, p


def grad_wrt_controls(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    U: torch.Tensor,
    x0: torch.Tensor,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
) -> torch.Tensor:
  """Gradient of total cost w.r.t. controls (adjoint method)."""
  jacobians = linearize(dynamics)
  grad_cost = linearize(cost)

  X = _rollout(dynamics, U, x0, *dynamics_args)
  T = U.shape[0]
  # Avoid requiring `dynamics(..., t=T)` to be defined: only linearize dynamics
  # for timesteps 0..T-1. (The terminal Jacobian row is unused.)
  dyn_timesteps = torch.arange(T, device=X.device, dtype=torch.int64)
  A, B = jacobians(X[:-1], U, dyn_timesteps, *dynamics_args)
  cost_timesteps = torch.arange(T + 1, device=X.device, dtype=torch.int64)
  q, r = grad_cost(X, pad(U), cost_timesteps, *cost_args)
  gradient, _, _ = adjoint(A, B, q, r)
  return gradient


def hvp(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    U: torch.Tensor,
    x0: torch.Tensor,
    V: torch.Tensor,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Hessian-vector product of the objective wrt U via forward-over-reverse."""
  grad_fn = partial(grad_wrt_controls,
                    cost,
                    dynamics,
                    x0=x0,
                    cost_args=cost_args,
                    dynamics_args=dynamics_args)

  def g(u):
    return grad_fn(u)

  # torch.func.jvp returns (primals_out, tangents_out)
  return torch.func.jvp(g, (U,), (V,))


def ddp_rollout(
    dynamics: Callable[..., torch.Tensor],
    X: torch.Tensor,
    U: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    dynamics_args: Sequence[Any] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
  """DDP rollout used for iLQR line search."""
  _require_cuda(X, U, K, k, alpha)
  T = U.shape[0]
  timesteps = torch.arange(T, device=U.device, dtype=torch.int64)
  n = X.shape[1]
  m = U.shape[1]
  Xnew = torch.empty((T + 1, n), device=X.device, dtype=X.dtype)
  Unew = torch.empty((T, m), device=X.device, dtype=X.dtype)
  Xnew[0] = X[0]
  for t in range(T):
    del_u = alpha * k[t] + K[t] @ (Xnew[t] - X[t])
    u = U[t] + del_u
    Xnew[t + 1] = dynamics(Xnew[t], u, timesteps[t], *dynamics_args)
    Unew[t] = u
  return Xnew, Unew


def _total_cost(cost: Callable[..., torch.Tensor], X: torch.Tensor, U: torch.Tensor,
                cost_args: Sequence[Any]) -> torch.Tensor:
  return evaluate(cost, X, pad(U), cost_args=cost_args).sum()


def line_search_ddp(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    X: torch.Tensor,
    U: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    obj: torch.Tensor,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
    alpha_0: float = 1.0,
    alpha_min: float = 0.00005,
    max_steps: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Fixed-budget line search over DDP rollouts (GPU-only control flow)."""
  obj_safe = torch.where(torch.isnan(obj), torch.tensor(float("inf"),
                                                       device=obj.device,
                                                       dtype=obj.dtype), obj)

  # Pre-compute alpha candidates (fixed length for graph/compile friendliness).
  alphas = alpha_0 * (0.5**torch.arange(max_steps,
                                        device=U.device,
                                        dtype=U.dtype))
  alphas = torch.clamp(alphas, min=alpha_min)

  X_cands = []
  U_cands = []
  obj_cands = []
  for i in range(max_steps):
    alpha = alphas[i]
    Xnew, Unew = ddp_rollout(dynamics,
                             X,
                             U,
                             K,
                             k,
                             alpha,
                             dynamics_args=dynamics_args)
    obj_new = _total_cost(cost, Xnew, Unew, cost_args)
    obj_new = torch.where(torch.isnan(obj_new), obj_safe, obj_new)
    X_cands.append(Xnew)
    U_cands.append(Unew)
    obj_cands.append(obj_new)

  obj_stack = torch.stack(obj_cands, dim=0)  # (S,)
  improved_mask = obj_stack < obj_safe
  any_improved = improved_mask.any()
  first_idx = torch.argmax(improved_mask.to(torch.int64))
  chosen_idx = torch.where(any_improved, first_idx,
                           torch.zeros((), device=U.device, dtype=torch.int64))

  chosen_obj = obj_stack[chosen_idx]
  X_chosen = X_cands[0]
  U_chosen = U_cands[0]
  for i in range(max_steps):
    mask = (chosen_idx == i)
    X_chosen = torch.where(mask, X_cands[i], X_chosen)
    U_chosen = torch.where(mask, U_cands[i], U_chosen)

  X_out = torch.where(any_improved, X_chosen, X)
  U_out = torch.where(any_improved, U_chosen, U)
  obj_out = torch.where(any_improved, chosen_obj, obj_safe)
  alpha_out = 0.5 * alphas[chosen_idx]
  return X_out, U_out, obj_out, alpha_out


def project_psd_cone(Q: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
  """Projects to the cone of PSD matrices by clamping eigenvalues."""
  S, V = torch.linalg.eigh(Q)
  S = torch.clamp(S, min=delta)
  Q_plus = V @ torch.diag(S) @ V.transpose(-1, -2)
  return 0.5 * (Q_plus + Q_plus.transpose(-1, -2))


def ilqr(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    U: torch.Tensor,
    maxiter: int = 100,
    grad_norm_threshold: float = 1e-4,
    relative_grad_norm_threshold: float = 0.0,
    obj_step_threshold: float = 0.0,
    inputs_step_threshold: float = 0.0,
    make_psd: bool = False,
    psd_delta: float = 0.0,
    alpha_0: float = 1.0,
    alpha_min: float = 0.00005,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
    static_loop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Tuple[torch.Tensor, ...], int]:
  """Iterative LQR (iLQR) in PyTorch.

  Returns (X, U, obj, gradient, adjoints, lqr, iteration).
  """
  _require_cuda(x0, U)
  T = U.shape[0]
  n = x0.shape[0]

  roll = partial(_rollout, dynamics)
  quadratizer = quadratize(cost)
  dynamics_jacobians = linearize(dynamics)
  cost_gradients = linearize(cost)

  X = roll(U, x0, *dynamics_args)
  timesteps = torch.arange(X.shape[0], device=X.device, dtype=torch.int64)
  obj = _total_cost(cost, X, U, cost_args)

  psd = torch.vmap(partial(project_psd_cone, delta=psd_delta))

  def get_lqr_params(X_in: torch.Tensor, U_in: torch.Tensor):
    Q, R, M = quadratizer(X_in, pad(U_in), timesteps, *cost_args)
    if make_psd:
      Q = psd(Q)
      R = psd(R)
    q, r = cost_gradients(X_in, pad(U_in), timesteps, *cost_args)
    dyn_timesteps = torch.arange(T, device=U_in.device, dtype=torch.int64)
    A, B = dynamics_jacobians(X_in[:-1], U_in, dyn_timesteps, *dynamics_args)
    return (Q, q, R[:T], r[:T], M[:T], A, B)

  c = torch.zeros((T, n), device=U.device, dtype=U.dtype)

  lqr = get_lqr_params(X, U)
  _, q, _, r, _, A, B = lqr
  gradient, adjoints, _ = adjoint(A, B, q, torch.cat([r, torch.zeros_like(r[:1])], dim=0))
  grad_norm_initial = torch.linalg.vector_norm(gradient)
  if float(grad_norm_initial.item()) <= float(grad_norm_threshold):
    # Match JAX behavior: if the initial gradient is already small enough,
    # return without performing an LQR solve/line-search step.
    return X, U, obj, gradient, adjoints, lqr, 0
  grad_norm_initial_safe = torch.where(torch.isnan(grad_norm_initial),
                                       torch.tensor(0.0,
                                                    device=U.device,
                                                    dtype=U.dtype),
                                       grad_norm_initial)
  grad_norm_threshold_val = max(
      float(grad_norm_threshold),
      float(relative_grad_norm_threshold) * (float(grad_norm_initial_safe.item()) + 1.0),
  )
  grad_norm_threshold_t = torch.tensor(grad_norm_threshold_val,
                                       device=U.device,
                                       dtype=U.dtype)

  for i in range(maxiter):
    iteration = i
    Q, q, R, r, M, A, B = lqr
    # Solve LQR subproblem.
    K, kk, _, _ = tvlqr(Q, q, R, r, M, A, B, c)

    X_new, U_new, obj_new, alpha = line_search_ddp(cost,
                                                   dynamics,
                                                   X,
                                                   U,
                                                   K,
                                                   kk,
                                                   obj,
                                                   cost_args=cost_args,
                                                   dynamics_args=dynamics_args,
                                                   alpha_0=alpha_0,
                                                   alpha_min=alpha_min)

    # Gradient based on current linearization.
    r_pad = torch.cat([r, torch.zeros_like(r[:1])], dim=0)
    gradient, adjoints, _ = adjoint(A, B, q, r_pad)
    grad_norm = torch.linalg.vector_norm(gradient)
    grad_norm = torch.where(torch.isnan(grad_norm),
                            torch.tensor(float("inf"),
                                         device=U.device,
                                         dtype=U.dtype), grad_norm)

    lqr_new = get_lqr_params(X_new, U_new)
    U_step = torch.linalg.vector_norm(U_new - U)
    obj_step = torch.abs(obj_new - obj)

    still_improving_obj = obj_step > obj_step_threshold * (torch.abs(obj_new) + 1.0)
    still_moving_U = U_step > inputs_step_threshold * (torch.linalg.vector_norm(U_new) + 1.0)
    still_progressing = still_improving_obj & still_moving_U
    has_potential = (grad_norm > grad_norm_threshold_t) & still_progressing & (alpha > alpha_min)

    if not static_loop:
      if not bool(has_potential.item()):
        break

    update = has_potential if static_loop else torch.tensor(True,
                                                           device=U.device)
    X = torch.where(update, X_new, X)
    U = torch.where(update, U_new, U)
    obj = torch.where(update, obj_new, obj)
    lqr = tuple(torch.where(update, new, old) for new, old in zip(lqr_new, lqr))

  return X, U, obj, gradient, adjoints, lqr, iteration + 1


def default_cem_hyperparams() -> Dict[str, float]:
  return {
      "sampling_smoothing": 0.0,
      "evolution_smoothing": 0.1,
      "elite_portion": 0.1,
      "max_iter": 10,
      "num_samples": 400,
  }


def cem_update_mean_stdev(old_mean: torch.Tensor, old_stdev: torch.Tensor,
                          controls: torch.Tensor, costs: torch.Tensor,
                          hyperparams: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
  num_samples = int(hyperparams["num_samples"])
  num_elites = int(num_samples * hyperparams["elite_portion"])
  best_idx = torch.argsort(costs)[:num_elites]
  elite_controls = controls[best_idx]
  new_mean = elite_controls.mean(dim=0)
  new_stdev = elite_controls.std(dim=0, unbiased=False)
  evo = hyperparams["evolution_smoothing"]
  updated_mean = evo * old_mean + (1.0 - evo) * new_mean
  updated_stdev = evo * old_stdev + (1.0 - evo) * new_stdev
  return updated_mean, updated_stdev


def gaussian_samples(
    generator: torch.Generator,
    mean: torch.Tensor,
    stdev: torch.Tensor,
    control_low: torch.Tensor,
    control_high: torch.Tensor,
    hyperparams: Dict[str, float],
) -> torch.Tensor:
  num_samples = int(hyperparams["num_samples"])
  noises = torch.randn((num_samples,) + mean.shape,
                       device=mean.device,
                       dtype=mean.dtype,
                       generator=generator)
  smoothing_coef = hyperparams["sampling_smoothing"]
  if smoothing_coef != 0.0:
    # Smooth along time axis with simple AR(1) filter.
    horizon = mean.shape[0]
    for t in range(1, horizon):
      noises[:, t] = smoothing_coef * noises[:, t - 1] + (1.0 - smoothing_coef) * noises[:, t]

  controls = mean.unsqueeze(0) + stdev.unsqueeze(0) * noises
  return torch.clamp(controls, min=control_low, max=control_high)


def cem(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    U: torch.Tensor,
    control_low: torch.Tensor,
    control_high: torch.Tensor,
    hyperparams: Optional[Dict[str, float]] = None,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Cross-Entropy Method (CEM) shooting optimizer (GPU-first)."""
  _require_cuda(x0, U, control_low, control_high)
  if hyperparams is None:
    hyperparams = default_cem_hyperparams()

  mean = U
  stdev = torch.ones_like(U)
  gen = torch.Generator(device=x0.device)
  gen.manual_seed(seed)

  def eval_controls(controls: torch.Tensor) -> torch.Tensor:
    # controls: (S, T, m)
    def single(Ui):
      Xi = _rollout(dynamics, Ui, x0, *dynamics_args)
      return _total_cost(cost, Xi, Ui, cost_args)

    return torch.vmap(single)(controls)

  for _ in range(int(hyperparams["max_iter"])):
    controls = gaussian_samples(gen, mean, stdev, control_low, control_high,
                                hyperparams)
    costs = eval_controls(controls)
    mean, stdev = cem_update_mean_stdev(mean, stdev, controls, costs, hyperparams)

  X = _rollout(dynamics, mean, x0, *dynamics_args)
  obj = _total_cost(cost, X, mean, cost_args)
  return X, mean, obj


def random_shooting(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    U: torch.Tensor,
    control_low: torch.Tensor,
    control_high: torch.Tensor,
    num_samples: int = 1024,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Random shooting optimizer (GPU-first)."""
  _require_cuda(x0, U, control_low, control_high)
  gen = torch.Generator(device=x0.device)
  gen.manual_seed(seed)
  noises = torch.rand((num_samples,) + U.shape,
                      device=U.device,
                      dtype=U.dtype,
                      generator=gen)
  controls = control_low + (control_high - control_low) * noises

  def single(Ui):
    Xi = _rollout(dynamics, Ui, x0, *dynamics_args)
    return _total_cost(cost, Xi, Ui, cost_args)

  costs = torch.vmap(single)(controls)
  best = torch.argmin(costs)
  U_best = controls[best]
  X_best = _rollout(dynamics, U_best, x0, *dynamics_args)
  obj = costs[best]
  return X_best, U_best, obj


def constrained_ilqr(
    cost: Callable[..., torch.Tensor],
    dynamics: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    U: torch.Tensor,
    equality_constraint: Callable[..., torch.Tensor] = None,
    inequality_constraint: Callable[..., torch.Tensor] = None,
    maxiter_al: int = 5,
    maxiter_ilqr: int = 100,
    grad_norm_threshold: float = 1.0e-4,
    relative_grad_norm_threshold: float = 0.0,
    obj_step_threshold: float = 0.0,
    inputs_step_threshold: float = 0.0,
    constraints_threshold: float = 1.0e-2,
    penalty_init: float = 1.0,
    penalty_update_rate: float = 10.0,
    make_psd: bool = True,
    psd_delta: float = 0.0,
    alpha_0: float = 1.0,
    alpha_min: float = 0.00005,
    cost_args: Sequence[Any] = (),
    dynamics_args: Sequence[Any] = (),
):
  """Constrained iLQR via an augmented Lagrangian outer loop (GPU-first).

  Mirrors `trajax.optimizers.constrained_ilqr`, but uses PyTorch tensors and
  runs entirely on the GPU.
  """
  _require_cuda(x0, U)
  if equality_constraint is None:
    equality_constraint = lambda x, u, t, *args: torch.zeros(
        (0,), device=x.device, dtype=x.dtype)
  if inequality_constraint is None:
    inequality_constraint = lambda x, u, t, *args: torch.zeros(
        (0,), device=x.device, dtype=x.dtype)

  horizon = U.shape[0] + 1
  t_range = torch.arange(horizon, device=U.device, dtype=torch.int64)

  X = rollout(dynamics, U, x0, dynamics_args=dynamics_args)

  eq_mapped = vectorize(lambda x, u, t: equality_constraint(x, u, t, *cost_args))
  ineq_mapped = vectorize(
      lambda x, u, t: inequality_constraint(x, u, t, *cost_args))

  U_pad = pad(U)
  equality_constraints = eq_mapped(X, U_pad, t_range)
  inequality_constraints = ineq_mapped(X, U_pad, t_range)

  dual_equality = torch.zeros_like(equality_constraints)
  dual_inequality = torch.zeros_like(inequality_constraints)
  penalty = torch.tensor(float(penalty_init), device=U.device, dtype=U.dtype)

  def inequality_projection(d):
    return torch.clamp(d, min=0.0)

  def augmented_lagrangian(x, u, t, dual_eq, dual_ineq, pen):
    J = cost(x, u, t)
    eq = equality_constraint(x, u, t)
    ineq = inequality_constraint(x, u, t)
    active = ~((dual_ineq[t].abs() <= 0.0) & (ineq < 0.0))
    if eq.numel() > 0:
      J = J + dual_eq[t].dot(eq) + 0.5 * pen * eq.dot(eq)
    if ineq.numel() > 0:
      J = J + dual_ineq[t].dot(ineq) + 0.5 * pen * (ineq * (active.to(ineq.dtype))).dot(ineq)
    return J

  # Outer augmented Lagrangian loop.
  iteration_ilqr = 0
  iteration_al = 0
  ineq_proj0 = inequality_projection(inequality_constraints)
  max_constraint_violation = torch.maximum(
      equality_constraints.abs().amax() if equality_constraints.numel() else torch.zeros(
          (), device=U.device, dtype=U.dtype),
      ineq_proj0.amax() if ineq_proj0.numel() else torch.zeros(
          (), device=U.device, dtype=U.dtype),
  )
  obj = torch.tensor(float("inf"), device=U.device, dtype=U.dtype)
  gradient = torch.full_like(U, float("inf"))

  max_comp_slack0 = (inequality_constraints * dual_inequality).abs().amax(
  ) if inequality_constraints.numel() else torch.zeros((), device=U.device, dtype=U.dtype)
  if (float(max_constraint_violation.item()) <= constraints_threshold and
      float(max_comp_slack0.item()) <= constraints_threshold):
    return (X, U, dual_equality, dual_inequality, penalty, equality_constraints,
            inequality_constraints, max_constraint_violation, obj, gradient,
            iteration_ilqr, iteration_al)

  for _ in range(maxiter_al):
    def cost_al(x, u, t):
      return augmented_lagrangian(x, u, t, dual_equality, dual_inequality,
                                  penalty)

    X, U, obj, gradient, _, _, it = ilqr(
        cost_al,
        dynamics,
        x0,
        U,
        maxiter=maxiter_ilqr,
        grad_norm_threshold=grad_norm_threshold,
        relative_grad_norm_threshold=relative_grad_norm_threshold,
        obj_step_threshold=obj_step_threshold,
        inputs_step_threshold=inputs_step_threshold,
        make_psd=make_psd,
        psd_delta=psd_delta,
        alpha_0=alpha_0,
        alpha_min=alpha_min,
        cost_args=(),
        dynamics_args=dynamics_args,
    )
    iteration_ilqr += int(it)
    iteration_al += 1

    U_pad = pad(U)
    equality_constraints = eq_mapped(X, U_pad, t_range)
    inequality_constraints = ineq_mapped(X, U_pad, t_range)
    ineq_proj = inequality_projection(inequality_constraints)
    max_constraint_violation = torch.maximum(
        equality_constraints.abs().amax() if equality_constraints.numel() else torch.zeros(
            (), device=U.device, dtype=U.dtype),
        ineq_proj.amax() if ineq_proj.numel() else torch.zeros(
            (), device=U.device, dtype=U.dtype),
    )

    dual_equality = dual_equality + penalty * equality_constraints
    dual_inequality = inequality_projection(dual_inequality +
                                            penalty * inequality_constraints)
    penalty = penalty * float(penalty_update_rate)

    max_comp_slack = (inequality_constraints * dual_inequality).abs().amax(
    ) if inequality_constraints.numel() else torch.zeros((), device=U.device, dtype=U.dtype)
    if (float(max_constraint_violation.item()) <= constraints_threshold and
        float(max_comp_slack.item()) <= constraints_threshold):
      break

  return (X, U, dual_equality, dual_inequality, penalty, equality_constraints,
          inequality_constraints, max_constraint_violation, obj, gradient,
          iteration_ilqr, iteration_al)


def constrained_ilqr_linear_quadratic_box(
    x0: torch.Tensor,
    U: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    x_goal: torch.Tensor,
    umax: torch.Tensor,
    maxiter_al: int = 5,
    maxiter_ilqr: int = 50,
    constraints_threshold: float = 1.0e-2,
    penalty_init: float = 1.0,
    penalty_update_rate: float = 10.0,
    final_weight: float = 10.0,
    delta: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
  """Compile-friendly constrained iLQR for LQ problems with box constraints.

  This specialized solver is intended for performance benchmarking and for
  workloads where:
    - dynamics: x[t+1] = A @ x[t] + B @ u[t]
    - stage cost: 0.5 * (x^T Q x + u^T R u)
    - terminal cost: 0.5 * x^T Q x + 0.5 * final_weight * (x-x_goal)^T Q (x-x_goal)
    - equality constraint: x[T] == x_goal
    - inequality constraint: |u[t]| <= umax (box), enforced via augmented Lagrangian

  It avoids `torch.func` transforms and uses `torch._higher_order_ops` control
  flow (scan/while_loop) for device-side looping without `.item()` syncs.
  """
  _require_cuda(x0, U, A, B, Q, R, x_goal, umax)
  T = U.shape[0]
  n = x0.shape[0]
  m = U.shape[1]

  # Avoid view aliasing: higher-order ops tracing forbids input aliasing.
  A_seq = A.unsqueeze(0).repeat(T, 1, 1)
  B_seq = B.unsqueeze(0).repeat(T, 1, 1)
  Q_stage = Q.unsqueeze(0).repeat(T, 1, 1)
  Qx_goal = Q @ x_goal
  c_seq = torch.zeros((T, n), device=U.device, dtype=U.dtype)

  def rollout_linear(U_in: torch.Tensor) -> torch.Tensor:
    timesteps = torch.arange(T, device=U_in.device, dtype=torch.int64)

    def step(x, xs_t):
      u_t, t = xs_t
      del t
      x_next = A @ x + B @ u_t
      y = x_next + 0  # avoid scan aliasing
      return x_next, y

    _, ys = _ho.scan(step, x0, (U_in, timesteps))
    return torch.cat([x0.unsqueeze(0), ys], dim=0)

  def constraints(X_in: torch.Tensor, U_in: torch.Tensor,
                  dual_ineq: torch.Tensor):
    U_pad = pad(U_in)
    eq_T = X_in[-1] - x_goal
    eq = torch.cat([torch.zeros((T, n), device=U.device, dtype=U.dtype),
                    eq_T.unsqueeze(0)],
                   dim=0)
    ineq = torch.cat([U_pad - umax, -U_pad - umax], dim=1)
    ineq_proj = torch.clamp(ineq, min=0.0)
    max_violation = torch.maximum(eq.abs().amax(),
                                  ineq_proj.amax() if ineq_proj.numel() else
                                  torch.zeros((), device=U.device, dtype=U.dtype))
    max_comp_slack = (ineq * dual_ineq).abs().amax() if ineq.numel() else torch.zeros(
        (), device=U.device, dtype=U.dtype)
    converged = (max_violation <= constraints_threshold) & (
        max_comp_slack <= constraints_threshold)
    return eq, ineq, max_violation, max_comp_slack, converged

  X = rollout_linear(U)
  dual_eq = torch.zeros((T + 1, n), device=U.device, dtype=U.dtype)
  dual_ineq = torch.zeros((T + 1, 2 * m), device=U.device, dtype=U.dtype)
  penalty = torch.tensor(float(penalty_init), device=U.device, dtype=U.dtype)
  iteration_ilqr0 = torch.zeros((), device=U.device, dtype=torch.int64)

  eq, ineq, max_violation, max_comp_slack, converged0 = constraints(X, U, dual_ineq)
  obj = torch.tensor(float("nan"), device=U.device, dtype=U.dtype)
  gradient = torch.zeros_like(U)
  I_n = torch.eye(n, device=U.device, dtype=U.dtype)

  def active_mask(U_in: torch.Tensor, dual_ineq_in: torch.Tensor):
    U_pad = pad(U_in)
    ineq_in = torch.cat([U_pad - umax, -U_pad - umax], dim=1)
    active = ~((dual_ineq_in.abs() <= 0.0) & (ineq_in < 0.0))
    return active.contiguous(), ineq_in

  def solve_lq_given_duals(U_init: torch.Tensor, dual_eq_in: torch.Tensor,
                           dual_ineq_in: torch.Tensor, penalty_in: torch.Tensor):
    # Inner loop: update the active set until stable (or maxiter_ilqr).
    active0, _ = active_mask(U_init, dual_ineq_in)

    def cond_fn(it, _X, _U, _active_prev, active_changed):
      return (it < maxiter_ilqr) & active_changed

    def body_fn(it, X_in, U_in, active_prev, _active_changed):
      active, ineq_in = active_mask(U_in, dual_ineq_in)
      a = active[:T].to(U.dtype)
      d = dual_ineq_in[:T]
      a1, a2 = a[:, :m], a[:, m:]
      d1, d2 = d[:, :m], d[:, m:]

      diag_pen = a1 + a2  # (T, m)
      R_seq = R.unsqueeze(0) + penalty_in * torch.diag_embed(diag_pen)
      r_seq = (d1 - d2) - penalty_in * ((a1 - a2) * umax)

      Q_seq = torch.cat([Q_stage,
                         (Q + final_weight * Q + penalty_in * I_n).unsqueeze(0)],
                        dim=0)
      q_T = (-final_weight * Qx_goal +
             dual_eq_in[-1] - penalty_in * x_goal)
      q_seq = torch.cat([torch.zeros((T, n), device=U.device, dtype=U.dtype),
                         q_T.unsqueeze(0)],
                        dim=0)

      M_seq = torch.zeros((T, n, m), device=U.device, dtype=U.dtype)
      K, k, _, _ = tvlqr(Q_seq,
                         q_seq,
                         R_seq,
                         r_seq,
                         M_seq,
                         A_seq,
                         B_seq,
                         c_seq,
                         delta=delta,
                         solver="solve",
                         use_scan=True)
      X_new, U_new = tvlqr_rollout(K, k, x0, A_seq, B_seq, c_seq, use_scan=True)
      active_new, _ = active_mask(U_new, dual_ineq_in)
      changed = (active_new != active).any()
      return (it + 1, X_new, U_new, active, changed)

    init_it = torch.zeros((), device=U.device, dtype=torch.int64)
    init_changed = torch.tensor(True, device=U.device)
    it, X_out, U_out, _active, _ = _ho.while_loop(cond_fn, body_fn,
                                                  (init_it, rollout_linear(U_init), U_init, active0, init_changed))
    return it, X_out, U_out

  def cond_al(it_al, it_ilqr, _X, _U, _dual_eq, _dual_ineq, _penalty, converged):
    del it_ilqr
    return (it_al < maxiter_al) & (~converged)

  def body_al(it_al, it_ilqr, X_in, U_in, dual_eq_in, dual_ineq_in, penalty_in, _converged):
    it_i, X_sol, U_sol = solve_lq_given_duals(U_in, dual_eq_in, dual_ineq_in,
                                              penalty_in)
    it_ilqr = it_ilqr + it_i
    eq_i, ineq_i, max_v_i, max_cs_i, conv_i = constraints(X_sol, U_sol, dual_ineq_in)
    del X_in, max_v_i, max_cs_i
    dual_eq_out = dual_eq_in + penalty_in * eq_i
    dual_ineq_out = torch.clamp(dual_ineq_in + penalty_in * ineq_i, min=0.0)
    penalty_out = penalty_in * float(penalty_update_rate)
    return (it_al + 1, it_ilqr, X_sol, U_sol, dual_eq_out, dual_ineq_out,
            penalty_out, conv_i)

  init = (torch.zeros((), device=U.device, dtype=torch.int64), iteration_ilqr0,
          X, U, dual_eq, dual_ineq, penalty, converged0)
  it_al, it_ilqr, X, U, dual_eq, dual_ineq, penalty, converged = _ho.while_loop(
      cond_al, body_al, init)
  iteration_al = it_al
  iteration_ilqr = it_ilqr

  eq, ineq, max_violation, _max_comp_slack, _ = constraints(X, U, dual_ineq)
  return (X, U, dual_eq, dual_ineq, penalty, eq, ineq, max_violation, obj,
          gradient, iteration_ilqr, iteration_al)
