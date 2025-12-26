"""PyTorch building blocks for trajectory optimizers."""

from functools import partial

import torch
import torch.func as tfunc

from trajax.torch.tvlqr import rollout as tvlqr_rollout
from trajax.torch.tvlqr import tvlqr

def const_tensor(scalar: float | int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.ones((), dtype=dtype, device=device)
    return t * scalar

def const_tensor_1d(scalar: float | int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.ones((1), dtype=dtype, device=device)
    return t * scalar



pad = lambda A: torch.vstack(
    (A, torch.zeros((1,) + A.shape[1:], device=A.device, dtype=A.dtype)))


def vectorize(fun, argnums=3):
  """Returns a vmap'ed version of fun over the first argnums arguments."""
  def vfun(*args):
    batched = args[:argnums]
    static = args[argnums:]
    length = batched[0].shape[0]
    outputs = []
    for i in range(length):
      per_args = [arg[i] for arg in batched]
      outputs.append(fun(*per_args, *static))
    return torch.stack(outputs)
  return vfun


def linearize(fun, argnums=3):
  """Vectorized jacobian operator for dynamics or cost."""
  def jacobian_x(x, u, t, *args):
    return tfunc.jacrev(lambda state: fun(state, u, t, *args))(x)

  def jacobian_u(x, u, t, *args):
    return tfunc.jacrev(lambda control: fun(x, control, t, *args))(u)

  def linearizer(*args):
    X, U, timesteps, *rest = args
    jx_list, ju_list = [], []
    for x, u, tt in zip(X, U, timesteps):
      jx_list.append(jacobian_x(x, u, tt, *rest))
      ju_list.append(jacobian_u(x, u, tt, *rest))
    return torch.stack(jx_list), torch.stack(ju_list)
  return linearizer


def quadratize(fun, argnums=3):
  """Vectorized Hessian operator for a scalar function."""
  def hessian_x(x, u, t, *args):
    return tfunc.hessian(lambda state: fun(state, u, t, *args))(x)

  def hessian_u(x, u, t, *args):
    return tfunc.hessian(lambda control: fun(x, control, t, *args))(u)

  def hessian_x_u(x, u, t, *args):
    def grad_x(state, control):
      return tfunc.grad(lambda s: fun(s, control, t, *args))(state)
    return tfunc.jacrev(lambda control: grad_x(x, control))(u)

  def quadratizer(*args):
    X, U, timesteps, *rest = args
    Q_list, R_list, M_list = [], [], []
    for x, u, tt in zip(X, U, timesteps):
      Q_list.append(hessian_x(x, u, tt, *rest))
      R_list.append(hessian_u(x, u, tt, *rest))
      M_list.append(hessian_x_u(x, u, tt, *rest))
    return torch.stack(Q_list), torch.stack(R_list), torch.stack(M_list)
  return quadratizer


def rollout(dynamics, U, x0):
  """Rolls out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0."""
  return _rollout(dynamics, U, x0)


# def _rollout(dynamics, U, x0, *args):
#   device, dtype = x0.device, x0.dtype
#   T, _ = U.shape
#   X = torch.zeros((T + 1, x0.shape[0]), device=device, dtype=dtype)
#   X[0] = x0
#   for t in range(T):
#     X[t + 1] = dynamics(X[t], U[t], t, *args)
#   return X

# vmap-friendly version
def _rollout(dynamics, U, x0, *args):
    T = U.shape[0]
    xs = [x0]
    for t in range(T):
        xs.append(dynamics(xs[-1], U[t], t, *args))
    return torch.stack(xs, dim=0)


def evaluate(cost, X, U, *args):
  """Evaluates cost(x, u, t) along a trajectory."""
  timesteps = torch.arange(X.shape[0], device=X.device)
  return vectorize(cost)(X, U, timesteps, *args)


def objective(cost, dynamics, U, x0):
  """Evaluates total cost for a control sequence."""
  X = _rollout(dynamics, U, x0)
  costs = evaluate(cost, X, pad(U))
  return torch.sum(costs)


def adjoint(A, B, q, r):
  """Solve adjoint equations."""
  T = q.shape[0] - 1
  # device, dtype = q.device, q.dtype
  P_store = []
  g_store = []
  p = q[T]
  for t in range(T - 1, -1, -1):
    g_t = r[t] + torch.matmul(B[t].transpose(-1, -2), p)
    p = torch.matmul(A[t].transpose(-1, -2), p) + q[t]
    P_store.append(p)
    g_store.append(g_t)

  gradient = torch.flip(torch.stack(g_store, dim=0), dims=[0])
  if len(P_store) > 1:
    adjoints = torch.vstack(
        (torch.flip(torch.stack(P_store[:-1], dim=0), dims=[0]),
         q[T].reshape(1, -1)))
  else:
    adjoints = q[T].reshape(1, -1)
  return gradient, adjoints, p


def grad_wrt_controls(cost, dynamics, U, x0, cost_args=(), dynamics_args=()):
  """Evaluates gradient at a control sequence."""
  jacobians = linearize(dynamics)
  grad_cost = linearize(cost)
  X = _rollout(dynamics, U, x0, *dynamics_args)
  timesteps = torch.arange(X.shape[0], device=U.device)
  A, B = jacobians(X, pad(U), timesteps, *dynamics_args)
  q, r = grad_cost(X, pad(U), timesteps, *cost_args)
  gradient, _, _ = adjoint(A, B, q, r)
  return gradient


def project_psd_cone(Q, delta=0.0):
  """Projects to the cone of positive semi-definite matrices."""
  S, V = torch.linalg.eigh(Q)
  S = torch.clamp(S, min=delta)
  Q_plus = V @ torch.diag(S) @ V.transpose(-1, -2)
  return 0.5 * (Q_plus + Q_plus.transpose(-1, -2))


def hamiltonian(cost, dynamics):
  """Returns function to evaluate associated Hamiltonian."""
  def fun(x, u, t, p, cost_args=(), dynamics_args=()):
    return cost(x, u, t, *cost_args) + torch.dot(p, dynamics(x, u, t,
                                                             *dynamics_args))
  return fun


def ddp_rollout(dynamics, X, U, K, k, alpha, *args):
  """Rollouts used in Differential Dynamic Programming."""
  T, m = U.shape
  xs = [X[0]]
  us = []
  for t in range(T):
    del_u = alpha * k[t] + torch.matmul(K[t], xs[-1] - X[t])
    u = U[t] + del_u
    x = dynamics(xs[-1], u, t, *args)
    us.append(u)
    xs.append(x)
  Xnew = torch.stack(xs, dim=0)
  Unew = torch.stack(us, dim=0)
  return Xnew, Unew


def line_search_ddp(cost,
                    dynamics,
                    X,
                    U,
                    K,
                    k,
                    obj,
                    cost_args=(),
                    dynamics_args=(),
                    alpha_0=1.0,
                    alpha_min=0.00005,
                    max_steps=20):
  """Performs line search with respect to DDP rollouts."""
  total_cost = lambda X1, U1: torch.sum(evaluate(cost, X1, pad(U1), *cost_args))
  alpha = alpha_0
  best_X, best_U, best_obj = X, U, obj
  for _ in range(max_steps):
    Xnew, Unew = ddp_rollout(dynamics, X, U, K, k, alpha, *dynamics_args)
    obj_new = total_cost(Xnew, Unew)
    improved = (obj_new < best_obj).to(X.dtype)
    best_X = improved.view(1, 1) * Xnew + (1 - improved.view(1, 1)) * best_X
    best_U = improved.view(1, 1) * Unew + (1 - improved.view(1, 1)) * best_U
    best_obj = torch.where(improved.bool(), obj_new, best_obj)
    alpha *= 0.5
    if alpha < alpha_min:
      break
  return best_X, best_U, best_obj, alpha


def ilqr(cost,
         dynamics,
         x0,
         U,
         maxiter=100,
         grad_norm_threshold=1e-4,
         relative_grad_norm_threshold=0.0,
         obj_step_threshold=0.0,
         inputs_step_threshold=0.0,
         make_psd=False,
         psd_delta=0.0,
         alpha_0=1.0,
         alpha_min=0.00005,
         cost_args=(),
         dynamics_args=()):
  """Iterative Linear Quadratic Regulator implemented in PyTorch."""
  del relative_grad_norm_threshold, obj_step_threshold, inputs_step_threshold
  T, _ = U.shape
  n = x0.shape[0]
  quadratizer = quadratize(cost)
  dynamics_jacobians = linearize(dynamics)
  cost_gradients = linearize(cost)
  evaluator = partial(evaluate, cost)
  psd = lambda mats: tfunc.vmap(lambda m: project_psd_cone(m, psd_delta))(mats)

  X = _rollout(dynamics, U, x0, *dynamics_args)
  timesteps = torch.arange(X.shape[0], device=x0.device)
  obj = torch.sum(evaluator(X, pad(U), *cost_args))

  def get_lqr_params(X_cur, U_cur):
    Q, R, M = quadratizer(X_cur, pad(U_cur), timesteps, *cost_args)
    if make_psd:
      Q = psd(Q)
      R = psd(R)
    q, r = cost_gradients(X_cur, pad(U_cur), timesteps, *cost_args)
    A, B = dynamics_jacobians(X_cur, pad(U_cur), timesteps, *dynamics_args)
    return (Q, q, R, r, M, A, B)

  c = torch.zeros((T, n), device=x0.device, dtype=x0.dtype)
  lqr = get_lqr_params(X, U)
  _, q, _, r, _, A, B = lqr
  gradient, adjoints, _ = adjoint(A, B, q, r)

  iteration = 0
  while iteration < maxiter:
    Q, q, R, r, M, A, B = lqr
    K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)
    X, U, obj, _ = line_search_ddp(cost, dynamics, X, U, K, k, obj,
                                   cost_args, dynamics_args, alpha_0,
                                   alpha_min)
    lqr = get_lqr_params(X, U)
    _, q, _, r, _, A, B = lqr
    gradient, adjoints, _ = adjoint(A, B, q, r)
    iteration += 1

  return X, U, obj, gradient, adjoints, lqr, iteration


# Sampling-based optimizers (CEM and random shooting)


def default_cem_hyperparams():
  return {
      'sampling_smoothing': 0.,
      'evolution_smoothing': 0.1,
      'elite_portion': 0.1,
      'max_iter': 10,
      'num_samples': 400
  }


def gaussian_samples(generator, mean, stdev, control_low, control_high,
                     hyperparams):
  """Samples a batch of controls based on Gaussian distribution."""
  device, dtype = mean.device, mean.dtype
  if generator is None:
    generator = torch.Generator(device=device)
  num_samples = hyperparams['num_samples']
  horizon, dim_control = mean.shape
  smoothing_coef = hyperparams['sampling_smoothing']
  sqrt_term = torch.sqrt(torch.tensor(1 - smoothing_coef**2,
                                      device=device, dtype=dtype))

  noise_list = [torch.randn((num_samples, dim_control), generator=generator,
                            device=device, dtype=mean.dtype)]
  for _ in range(1, horizon):
    eps = torch.randn((num_samples, dim_control), generator=generator,
                      device=device, dtype=mean.dtype)
    next_noise = smoothing_coef * noise_list[-1] + sqrt_term * eps
    noise_list.append(next_noise)

  noises = torch.stack(noise_list, dim=1)
  samples = noises * stdev + mean
  control_low = control_low.unsqueeze(0).expand_as(samples)
  control_high = control_high.unsqueeze(0).expand_as(samples)
  samples = torch.max(torch.min(samples, control_high), control_low)
  return samples


def cem(cost,
        dynamics,
        init_state,
        init_controls,
        control_low,
        control_high,
        generator=None,
        hyperparams=None):
  """Cross Entropy Method implemented with PyTorch."""
  if generator is None:
    generator = torch.Generator(device=init_controls.device)
  if hyperparams is None:
    hyperparams = default_cem_hyperparams()
  mean = init_controls.clone()
  stdev = ((control_high - control_low) / 2.).clone()
  obj_fn = partial(objective, cost, dynamics)

  for _ in range(hyperparams['max_iter']):
    controls = gaussian_samples(generator, mean, stdev, control_low,
                                control_high, hyperparams)
    costs = torch.stack([obj_fn(ctrls, init_state) for ctrls in controls])
    num_elites = max(1, int(hyperparams['num_samples'] *
                            hyperparams['elite_portion']))
    elite_costs, elite_idx = torch.topk(costs, num_elites, largest=False)
    elite_controls = controls[elite_idx]
    new_mean = elite_controls.mean(dim=0)
    new_stdev = elite_controls.std(dim=0)
    mean = (hyperparams['evolution_smoothing'] * mean +
            (1 - hyperparams['evolution_smoothing']) * new_mean)
    stdev = (hyperparams['evolution_smoothing'] * stdev +
             (1 - hyperparams['evolution_smoothing']) * new_stdev)

  X = rollout(dynamics, mean, init_state)
  obj = objective(cost, dynamics, mean, init_state)
  return X, mean, obj


def random_shooting(cost,
                    dynamics,
                    init_state,
                    init_controls,
                    control_low,
                    control_high,
                    generator=None,
                    hyperparams=None):
  """Random shooting method implemented with PyTorch."""
  if generator is None:
    generator = torch.Generator(device=init_controls.device)
  if hyperparams is None:
    hyperparams = default_cem_hyperparams()
  mean = init_controls.clone()
  stdev = ((control_high - control_low) / 2.).clone()
  controls = gaussian_samples(generator, mean, stdev, control_low,
                              control_high, hyperparams)
  costs = torch.stack([objective(cost, dynamics, ctrls, init_state)
                       for ctrls in controls])
  best_idx = torch.argmin(costs)
  U = controls[best_idx]
  X = rollout(dynamics, mean, init_state)
  obj = objective(cost, dynamics, mean, init_state)
  return X, U, obj


def constrained_ilqr(cost,
                     dynamics,
                     x0,
                     U,
                     equality_constraint=lambda x, u, t: torch.empty(
                         0, device=x.device, dtype=x.dtype),
                     inequality_constraint=lambda x, u, t: torch.empty(
                         0, device=x.device, dtype=x.dtype),
                     maxiter_al=5,
                     maxiter_ilqr=100,
                     grad_norm_threshold=1.0e-4,
                     relative_grad_norm_threshold=0.0,
                     obj_step_threshold=0.0,
                     inputs_step_threshold=0.0,
                     constraints_threshold=1.0e-2,
                     penalty_init=1.0,
                     penalty_update_rate=10.0,
                     make_psd=True,
                     psd_delta=0.0,
                     alpha_0=1.0,
                     alpha_min=0.00005):
  """Constrained Iterative Linear Quadratic Regulator (PyTorch)."""

  device, dtype = x0.device, x0.dtype
  horizon = len(U) + 1
  t_range = torch.arange(horizon, device=device)

  X = rollout(dynamics, U, x0)

  def augmented_lagrangian(x, u, t, dual_equality, dual_inequality, penalty):
    J = cost(x, u, t)
    equality = equality_constraint(x, u, t)
    inequality = inequality_constraint(x, u, t)

    if equality.numel():
      J = J + torch.dot(dual_equality[t], equality)
      J = J + 0.5 * penalty * torch.dot(equality, equality)

    if inequality.numel():
      active_set = torch.logical_not(
          torch.isclose(dual_inequality[t],
                        torch.tensor(0.0, device=device, dtype=dtype))
          & (inequality < 0.0)).to(dtype)
      J = J + torch.dot(dual_inequality[t], inequality)
      J = J + 0.5 * penalty * torch.dot(active_set * inequality, inequality)

    return J

  def dual_update(constraint, dual, penalty):
    return dual + penalty * constraint

  def inequality_projection(dual):
    return torch.maximum(dual, torch.tensor(0.0, device=device, dtype=dtype))

  equality_constraint_mapped = vectorize(equality_constraint)
  inequality_constraint_mapped = vectorize(inequality_constraint)

  U_pad = pad(U)
  equality_constraints = equality_constraint_mapped(X, U_pad, t_range)
  inequality_constraints = inequality_constraint_mapped(X, U_pad, t_range)

  dual_equality = torch.zeros_like(equality_constraints)
  dual_inequality = torch.zeros_like(inequality_constraints)

  penalty = torch.tensor(penalty_init, device=device, dtype=dtype)

  def _safe_max_abs(x):
    return torch.max(torch.abs(x)) if x.numel() else torch.tensor(
        0.0, device=device, dtype=dtype)

  iteration_ilqr_t = torch.tensor(0, device=device, dtype=torch.int64)
  iteration_al_t = torch.tensor(0, device=device, dtype=torch.int64)
  done = torch.tensor(False, device=device)

  X_out, U_out = X, U
  obj_out = torch.tensor(0.0, device=device, dtype=dtype)
  gradient_out = torch.zeros_like(U)
  dual_equality_out, dual_inequality_out = dual_equality, dual_inequality
  penalty_out = penalty
  equality_constraints_out = equality_constraints
  inequality_constraints_out = inequality_constraints
  max_constraint_violation_out = _safe_max_abs(equality_constraints)
  max_complementary_slack_out = _safe_max_abs(inequality_constraints * dual_inequality)

  for _ in range(maxiter_al):
    al_args = {
        'dual_equality': dual_equality,
        'dual_inequality': dual_inequality,
        'penalty': penalty,
    }

    X_new, U_new, obj_new, gradient_new, _, _, iteration = ilqr(
        partial(augmented_lagrangian, **al_args),
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
        alpha_min=alpha_min)

    U_pad = pad(U_new)
    equality_constraints_new = equality_constraint_mapped(X_new, U_pad, t_range)
    inequality_constraints_new = inequality_constraint_mapped(X_new, U_pad, t_range)
    inequality_constraints_projected = inequality_projection(inequality_constraints_new)

    max_constraint_violation = torch.maximum(
        _safe_max_abs(equality_constraints_new),
        torch.max(inequality_constraints_projected) if inequality_constraints_projected.numel() else torch.tensor(
            0.0, device=device, dtype=dtype))

    complementary_slack = (inequality_constraints_new * dual_inequality)
    max_complementary_slack = _safe_max_abs(complementary_slack)

    stop_now = torch.logical_and(
        max_constraint_violation <= constraints_threshold,
        max_complementary_slack <= constraints_threshold)

    keep_prev = done
    active = ~keep_prev

    X_out = torch.where(keep_prev.view(1, 1), X_out, X_new)
    U_out = torch.where(keep_prev.view(1, 1), U_out, U_new)
    obj_out = torch.where(keep_prev, obj_out, obj_new)
    gradient_out = torch.where(keep_prev.view(1, 1), gradient_out,
                               gradient_new)
    equality_constraints_out = torch.where(keep_prev.view(1, 1),
                                           equality_constraints_out,
                                           equality_constraints_new)
    inequality_constraints_out = torch.where(keep_prev.view(1, 1),
                                             inequality_constraints_out,
                                             inequality_constraints_new)
    max_constraint_violation_out = torch.where(keep_prev,
                                               max_constraint_violation_out,
                                               max_constraint_violation)
    max_complementary_slack_out = torch.where(keep_prev,
                                              max_complementary_slack_out,
                                              max_complementary_slack)

    dual_equality_new = dual_update(equality_constraints_new,
                                    dual_equality, penalty)
    dual_inequality_new = inequality_projection(
        dual_update(inequality_constraints_new, dual_inequality, penalty))

    dual_equality = torch.where(keep_prev.view(1, 1), dual_equality,
                                dual_equality_new)
    dual_inequality = torch.where(keep_prev.view(1, 1), dual_inequality,
                                  dual_inequality_new)

    dual_equality_out = torch.where(keep_prev.view(1, 1), dual_equality_out,
                                    dual_equality)
    dual_inequality_out = torch.where(keep_prev.view(1, 1),
                                      dual_inequality_out, dual_inequality)

    penalty_new = torch.where(keep_prev, penalty, penalty * penalty_update_rate)
    penalty_out = torch.where(keep_prev, penalty_out, penalty_new)
    penalty = penalty_new

    iteration_tensor = torch.tensor(iteration, device=device, dtype=torch.int64)
    iteration_ilqr_t = torch.where(keep_prev, iteration_ilqr_t,
                                   iteration_ilqr_t + iteration_tensor)
    iteration_al_t = torch.where(keep_prev, iteration_al_t,
                                 iteration_al_t + 1)

    done = torch.logical_or(done, stop_now)
    U = torch.where(active.view(1, 1), U_new, U)
    X = torch.where(active.view(1, 1), X_new, X)

  return (X_out, U_out, dual_equality_out, dual_inequality_out, penalty_out,
          equality_constraints_out, inequality_constraints_out,
          max_constraint_violation_out, obj_out, gradient_out,
          int(iteration_ilqr_t.item()), int(iteration_al_t.item()))
