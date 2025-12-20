"""PyTorch building blocks for trajectory optimizers."""

from functools import partial

import torch
import torch.func as tfunc

from trajax.torch.tvlqr import rollout as tvlqr_rollout
from trajax.torch.tvlqr import tvlqr

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


def _rollout(dynamics, U, x0, *args):
  device, dtype = x0.device, x0.dtype
  T, _ = U.shape
  X = torch.zeros((T + 1, x0.shape[0]), device=device, dtype=dtype)
  X[0] = x0
  for t in range(T):
    X[t + 1] = dynamics(X[t], U[t], t, *args)
  return X


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
  device, dtype = q.device, q.dtype
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
  device, dtype = X.device, X.dtype
  T, m = U.shape
  n = X.shape[1]
  Xnew = torch.zeros((T + 1, n), device=device, dtype=dtype)
  Unew = torch.zeros((T, m), device=device, dtype=dtype)
  Xnew[0] = X[0]
  for t in range(T):
    del_u = alpha * k[t] + torch.matmul(K[t], Xnew[t] - X[t])
    u = U[t] + del_u
    x = dynamics(Xnew[t], u, t, *args)
    Unew[t] = u
    Xnew[t + 1] = x
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
