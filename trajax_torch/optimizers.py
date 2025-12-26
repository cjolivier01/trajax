# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
"""Building Blocks for Gradient-based Trajectory Optimizers (PyTorch version).

Notation:

- x denotes state, a 1D tensor of shape [n]
- u denotes control, a 1D tensor of shape [m]
- t denotes time, a scalar integer time index.

A Trajectory optimization problem is specified via three components:

(1) A scalar-valued cost function with signature,
              c = cost(x, u, t, *args)

(2) A vector-valued dynamics function with signature,
              xdot = dynamics(x, u, t, *args)

    where xdot is state time derivative of shape [n].

(3) The initial state x0, a 1D tensor of shape [n].

The problem is to minimize over a sequence u[0], u[1]...u[T-1],

    sum_{t=0}^{T-1} cost(x[t], u[t], t) + cost(x[T], zeros(m), T)

    subject to:

      x[t+1] = dynamics(x[t], u[t], t)
      x[0] = x0 is given.
"""

from functools import partial
import torch
import scipy.optimize as osp_optimize
from .tvlqr import rollout as tvlqr_rollout
from .tvlqr import tvlqr


# Convenience routine to pad zeros for vectorization purposes.
def pad(A):
    """Pad tensor A with zeros along first dimension."""
    return torch.cat([A, torch.zeros((1,) + A.shape[1:], device=A.device, dtype=A.dtype)], dim=0)


def vectorize(fun, argnums=3):
    """Returns a vectorized version of the input function.

    Args:
        fun: a function f(*args) to be mapped over.
        argnums: number of leading arguments of fun to vectorize.

    Returns:
        Vectorized/Batched function with arguments corresponding to fun, but extra
        batch dimension in axis 0 for first argnums arguments (x, u, t typically).
        Remaining arguments are not batched.
    """

    def vfun(*args):
        batched_args = args[:argnums]
        remaining_args = args[argnums:]

        # Determine batch size from first batched argument
        batch_size = batched_args[0].shape[0]
        results = []

        for i in range(batch_size):
            batch_inputs = tuple(arg[i] for arg in batched_args)
            results.append(fun(*batch_inputs, *remaining_args))

        # Handle tuple returns (e.g., from linearize, quadratize)
        if isinstance(results[0], tuple):
            num_outputs = len(results[0])
            stacked_results = tuple(
                torch.stack([results[i][j] for i in range(batch_size)])
                for j in range(num_outputs)
            )
            return stacked_results
        else:
            return torch.stack(results)

    return vfun


def linearize(fun, argnums=3):
    """Vectorized gradient or jacobian operator.

    Args:
        fun: scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to vectorize.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory, e.g.,

            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically torch.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """

    def jacobian_x_fn(*args):
        x, u, t = args[:3]
        remaining_args = args[3:]
        x_var = x.clone().detach().requires_grad_(True)

        output = fun(x_var, u, t, *remaining_args)

        if output.dim() == 0:  # Scalar output (gradient)
            if x_var.grad is not None:
                x_var.grad.zero_()
            grad_tuple = torch.autograd.grad(output, x_var, create_graph=True, allow_unused=True)
            grad = grad_tuple[0] if grad_tuple[0] is not None else torch.zeros_like(x_var)
            return grad
        else:  # Vector output (jacobian)
            jac = torch.autograd.functional.jacobian(
                lambda x_: fun(x_, u, t, *remaining_args), x_var
            )
            return jac

    def jacobian_u_fn(*args):
        x, u, t = args[:3]
        remaining_args = args[3:]
        u_var = u.clone().detach().requires_grad_(True)

        output = fun(x, u_var, t, *remaining_args)

        if output.dim() == 0:  # Scalar output (gradient)
            if u_var.grad is not None:
                u_var.grad.zero_()
            grad_tuple = torch.autograd.grad(output, u_var, create_graph=True, allow_unused=True)
            grad = grad_tuple[0] if grad_tuple[0] is not None else torch.zeros_like(u_var)
            return grad
        else:  # Vector output (jacobian)
            jac = torch.autograd.functional.jacobian(
                lambda u_: fun(x, u_, t, *remaining_args), u_var
            )
            return jac

    def linearizer(*args):
        return jacobian_x_fn(*args), jacobian_u_fn(*args)

    return vectorize(linearizer, argnums)


def quadratize(fun, argnums=3):
    """Vectorized Hessian operator for a scalar function.

    Args:
        fun: scalar function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to vectorize.

    Returns:
        A function that evaluates Hessians with respect to state and controls along
        a trajectory, e.g.,

          Q, R, M = quadratize(cost)(X, pad(U), timesteps)

         where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically torch.arange(T+1)

        and,
              Q is [T+1, n, n] Hessian wrt state: partial^2 fun/ partial^2 x,
              R is [T+1, m, m] Hessian wrt control: partial^2 fun/ partial^2 u,
              M is [T+1, n, m] mixed derivatives: partial^2 fun/partial_x partial_u
    """

    def hessian_x_fn(*args):
        x, u, t = args[:3]
        remaining_args = args[3:]
        x_var = x.detach().requires_grad_(True)

        hess = torch.autograd.functional.hessian(
            lambda x_: fun(x_, u, t, *remaining_args), x_var
        )
        return hess

    def hessian_u_fn(*args):
        x, u, t = args[:3]
        remaining_args = args[3:]
        u_var = u.detach().requires_grad_(True)

        hess = torch.autograd.functional.hessian(
            lambda u_: fun(x, u_, t, *remaining_args), u_var
        )
        return hess

    def hessian_xu_fn(*args):
        x, u, t = args[:3]
        remaining_args = args[3:]
        x_var = x.detach().requires_grad_(True)
        u_var = u.detach().requires_grad_(True)

        # Compute mixed partial derivative
        output = fun(x_var, u_var, t, *remaining_args)
        grad_x = torch.autograd.grad(output, x_var, create_graph=True)[0]

        # Jacobian of grad_x with respect to u
        M = torch.autograd.functional.jacobian(
            lambda u_: torch.autograd.grad(
                fun(x_var, u_, t, *remaining_args), x_var, create_graph=True
            )[0],
            u_var
        )
        return M

    def quadratizer(*args):
        return hessian_x_fn(*args), hessian_u_fn(*args), hessian_xu_fn(*args)

    return vectorize(quadratizer, argnums)


def rollout(dynamics, U, x0):
    """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0.

    Args:
        dynamics: a function f(x, u, t) to rollout.
        U: (T, m) tensor for control sequence.
        x0: (n, ) tensor for initial state.

    Returns:
         X: (T+1, n) state trajectory.
    """
    return _rollout(dynamics, U, x0)


def _rollout(dynamics, U, x0, *args):
    """Internal rollout with additional arguments."""
    T, m = U.shape
    n = x0.shape[0]
    device = U.device
    dtype = U.dtype

    X = torch.zeros((T + 1, n), device=device, dtype=dtype)
    X[0] = x0

    for t in range(T):
        X[t + 1] = dynamics(X[t], U[t], t, *args)

    return X


def evaluate(cost, X, U, *args):
    """Evaluates cost(x, u, t) along a trajectory.

    Args:
        cost: cost_fn with signature cost(x, u, t, *args)
        X: (T, n) state trajectory.
        U: (T, m) control sequence.
        *args: args for cost_fn

    Returns:
        objectives: (T, ) array of objectives.
    """
    timesteps = torch.arange(X.shape[0], device=X.device)
    return vectorize(cost)(X, U, timesteps, *args)


def objective(cost, dynamics, U, x0):
    """Evaluates total cost for a control sequence.

    Args:
        cost: cost_fn with signature cost(x, u, t)
        dynamics: dynamics_fn with signature dynamics(x, u, t)
        U: (T, m) control sequence.
        x0: (n, ) initial state.

    Returns:
        objectives: total objective summed across time.
    """
    X = _rollout(dynamics, U, x0)
    return torch.sum(evaluate(cost, X, pad(U)))


def adjoint(A, B, q, r):
    """Solve adjoint equations.

    Args:
        A: dynamics Jacobians with respect to state.
        B: dynamics Jacobians with respect to control.
        q: cost gradients with respect to state.
        r: cost gradients with respect to control.

    Returns:
        gradient, adjoints, final adjoint variable.

    Usage:
      q, r = linearize(cost)(X, pad(U), timesteps)
      A, B = linearize(dynamics)(X, pad(U), torch.arange(T + 1))
      gradient, adjoints, _ = adjoint(A, B, q, r)
    """

    n = q.shape[1]
    T = q.shape[0] - 1
    m = r.shape[1]
    device = A.device
    dtype = A.dtype

    P = torch.zeros((T, n), device=device, dtype=dtype)
    g = torch.zeros((T, m), device=device, dtype=dtype)

    p = q[T]

    for tt in range(T):
        t = T - 1 - tt
        g[t] = r[t] + torch.matmul(B[t].T, p)
        p = torch.matmul(A[t].T, p) + q[t]
        if t > 0:
            P[t - 1] = p

    P_full = torch.cat([P, q[T].unsqueeze(0)], dim=0)

    return g, P_full, p


def grad_wrt_controls(cost, dynamics, U, x0, cost_args=(), dynamics_args=()):
    """Evaluates gradient at a control sequence.

    Args:
        cost: cost_fn
        dynamics: dynamics_fn
        U: (T, m) control sequence.
        x0: (n, ) initial state.
        cost_args: args passed to cost
        dynamics_args: args passed to dynamics.

    Returns:
        gradient (T, m) of total cost with respect to controls.
    """
    X = _rollout(dynamics, U, x0, *dynamics_args)
    timesteps = torch.arange(X.shape[0], device=X.device)

    jacobians = linearize(dynamics)
    grad_cost = linearize(cost)

    A, B = jacobians(X, pad(U), timesteps, *dynamics_args)
    q, r = grad_cost(X, pad(U), timesteps, *cost_args)
    gradient, _, _ = adjoint(A, B, q, r)
    return gradient


def project_psd_cone(Q, delta=0.0):
    """Projects to the cone of positive semi-definite matrices.

    Args:
        Q: [n, n] symmetric matrix.
        delta: minimum eigenvalue of the projection.

    Returns:
        [n, n] symmetric matrix projection of the input.
    """
    S, V = torch.linalg.eigh(Q)
    S = torch.maximum(S, torch.tensor(delta, device=Q.device, dtype=Q.dtype))
    Q_plus = torch.matmul(V, torch.matmul(torch.diag(S), V.T))
    return 0.5 * (Q_plus + Q_plus.T)


def ddp_rollout(dynamics, X, U, K, k, alpha, *args):
    """Rollouts used in Differential Dynamic Programming.

    Args:
        dynamics: function with signature dynamics(x, u, t, *args).
        X: [T+1, n] current state trajectory.
        U: [T, m] current control sequence.
        K: [T, m, n] state feedback gains.
        k: [T, m] affine terms in state feedback.
        alpha: line search parameter.
        *args: passed to dynamics.

    Returns:
        Xnew, Unew: updated state trajectory and control sequence, via:

          del_u = alpha * k[t] + torch.matmul(K[t], Xnew[t] - X[t])
          u = U[t] + del_u
          x = dynamics(Xnew[t], u, t)
    """
    n = X.shape[1]
    T, m = U.shape
    device = X.device
    dtype = X.dtype

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
                    alpha_min=0.00005):
    """Performs line search with respect to DDP rollouts."""

    # Handle NaN
    if torch.isnan(obj):
        obj = torch.tensor(float('inf'), device=obj.device, dtype=obj.dtype)

    total_cost = lambda X, U: torch.sum(evaluate(cost, X, pad(U), *cost_args))

    alpha = alpha_0
    X_return = X
    U_return = U
    obj_return = obj

    while alpha > alpha_min:
        Xnew, Unew = ddp_rollout(dynamics, X, U, K, k, alpha, *dynamics_args)
        obj_new = total_cost(Xnew, Unew)

        if torch.isnan(obj_new):
            obj_new = obj

        # Only return new trajs if leads to a strict cost decrease
        if obj_new < obj:
            X_return = Xnew
            U_return = Unew
            obj_return = obj_new
            break
        else:
            alpha = 0.5 * alpha

    return X_return, U_return, obj_return, alpha


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
         alpha_min=0.00005):
    """Iterative Linear Quadratic Regulator.

    Optimization terminates if any one these conditions is true:
     1) Reached maximum iteration `maxiter`.
     2) The line-search step, relative to a full step, was less than `alpha_min`.
     3) The norm of the gradient is less than `grad_norm_threshold` or
        `relative_grad_norm_threshold` times one plus the gradient norm at the
        initial guess (1 + norm(grad(U_0))), whichever is the larger threshold.
     4) The norm of the step taken in the input (controls) space is less than one
        plus the norm of the current control inputs (1 + norm(U)) times
        `inputs_step_threshold`.
     5) The improvement in objective value is less than one plus the objective
        value (1 + abs(obj)) times `obj_step_threshold`.

    Args:
        cost:      cost(x, u, t) returns scalar.
        dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
        x0: initial_state - 1D tensor of shape (n, ).
        U: initial_controls - 2D tensor of shape (T, m).
        maxiter: maximum iterations.
        grad_norm_threshold: tolerance for stopping optimization.
        relative_grad_norm_threshold: tolerance on gradient norm for stopping
          optimization, relative to the gradient norm at the initial guess.
        obj_step_threshold: tolerance on objective value steps for stopping
          optimization, relative to the objective value itself.
        inputs_step_threshold: tolerance on input steps for stopping
          optimization, relative to the initial input (aka controls).
        make_psd: whether to zero negative eigenvalues after quadratization.
        psd_delta: The delta value to make the problem PSD.
        alpha_0: initial line search value.
        alpha_min: minimum line search value.

    Returns:
        X: optimal state trajectory - tensor of shape (T+1, n).
        U: optimal control trajectory - tensor of shape (T, m).
        obj: final objective achieved.
        gradient: gradient at the solution returned.
        adjoints: associated adjoint variables.
        lqr: inputs to the final LQR solve.
        iteration: number of iterations upon convergence.
    """

    T, m = U.shape
    n = x0.shape[0]
    device = U.device
    dtype = U.dtype

    quadratizer = quadratize(cost)
    dynamics_jacobians = linearize(dynamics)
    cost_gradients = linearize(cost)

    X = _rollout(dynamics, U, x0)
    timesteps = torch.arange(X.shape[0], device=device)
    obj = torch.sum(evaluate(cost, X, pad(U)))

    def get_lqr_params(X, U):
        Q, R, M = quadratizer(X, pad(U), timesteps)

        if make_psd:
            Q = torch.stack([project_psd_cone(Q[i], psd_delta) for i in range(Q.shape[0])])
            R = torch.stack([project_psd_cone(R[i], psd_delta) for i in range(R.shape[0])])

        q, r = cost_gradients(X, pad(U), timesteps)
        A, B = dynamics_jacobians(X, pad(U), timesteps)

        return (Q, q, R, r, M, A, B)

    c = torch.zeros((T, n), device=device, dtype=dtype)

    lqr = get_lqr_params(X, U)
    _, q, _, r, _, A, B = lqr
    gradient, adjoints, _ = adjoint(A, B, q, r)
    grad_norm_initial = torch.linalg.norm(gradient)
    grad_norm_threshold = max(
        grad_norm_threshold,
        relative_grad_norm_threshold *
        (1.0 if torch.isnan(grad_norm_initial) else float(grad_norm_initial + 1.0))
    )

    alpha = alpha_0
    iteration = 0
    obj_step = float('inf')
    U_step = float('inf')

    while iteration < maxiter:
        Q, q, R, r, M, A, B = lqr

        K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)
        X_new, U_new, obj_new, alpha = line_search_ddp(
            cost, dynamics, X, U, K, k, obj,
            (), (), alpha_0, alpha_min
        )

        lqr = get_lqr_params(X_new, U_new)
        _, q_new, _, r_new, _, A_new, B_new = lqr
        gradient, adjoints, _ = adjoint(A_new, B_new, q_new, r_new)

        U_step = torch.linalg.norm(U_new - U).item()
        obj_step = abs(float(obj_new - obj))

        # Update to new solution
        X, U, obj = X_new, U_new, obj_new
        iteration = iteration + 1

        # Check stopping criteria
        grad_norm = torch.linalg.norm(gradient).item()
        if torch.isnan(gradient).any():
            grad_norm = float('inf')

        still_improving_obj = obj_step > obj_step_threshold * (abs(float(obj)) + 1.0)
        still_moving_U = U_step > inputs_step_threshold * (torch.linalg.norm(U).item() + 1.0)
        still_progressing = still_improving_obj and still_moving_U
        has_potential_to_improve = grad_norm > grad_norm_threshold and still_progressing

        if not (has_potential_to_improve and alpha > alpha_min):
            break

    return X, U, obj, gradient, adjoints, lqr, iteration


def scipy_minimize(cost,
                   dynamics,
                   x0,
                   U,
                   method='CG',
                   bounds=None,
                   options=None,
                   callback=None):
    """First Order Optimizers from scipy.optimize.minimize for Optimal Control.

    Args:
        cost:      cost(x, u, t) returns scalar.
        dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
        x0: initial_state - 1D tensor of shape (n, ).
        U: initial_controls - 2D tensor of shape (T, m).
        method: 'CG', 'Newton-CG', 'BFGS', 'LBFGS'
        bounds: Passed to scipy.optimize.minimize for bound constraints.
        options: dictionary of solver options.
        callback: called after each iteration. See scipy.optimize.minimize docs.

    Returns:
        X: optimal state trajectory - tensor of shape (T+1, n).
        U: optimal control trajectory - tensor of shape (T, m).
        obj: final objective achieved.
        gradient: gradient at the solution returned.
        iteration: number of iterations upon convergence.
    """

    T, m = U.shape
    device = U.device
    dtype = U.dtype

    def fun(u):
        u_tensor = torch.tensor(u.reshape((T, m)), device=device, dtype=dtype)
        return float(objective(cost, dynamics, u_tensor, x0))

    def grad_fun(u):
        u_tensor = torch.tensor(u.reshape((T, m)), device=device, dtype=dtype, requires_grad=True)
        obj = objective(cost, dynamics, u_tensor, x0)
        grad = torch.autograd.grad(obj, u_tensor)[0]
        return grad.cpu().detach().numpy().flatten()

    res = osp_optimize.minimize(
        fun,
        U.cpu().detach().numpy().flatten(),
        method=method,
        jac=grad_fun,
        bounds=bounds,
        options=options,
        callback=callback)

    uopt = res.x
    U_opt = torch.tensor(uopt.reshape((T, m)), device=device, dtype=dtype)
    X_opt = rollout(dynamics, U_opt, x0)

    return X_opt, U_opt, res.fun, torch.tensor(res.jac.reshape((T, m)), device=device, dtype=dtype), res.nit


# Sampling based Zeroth Order Optimization via Cross-Entropy Method


def default_cem_hyperparams():
    return {
        'sampling_smoothing': 0.,
        'evolution_smoothing': 0.1,
        'elite_portion': 0.1,
        'max_iter': 10,
        'num_samples': 400
    }


def cem_update_mean_stdev(old_mean, old_stdev, controls, costs, hyperparams):
    """Computes new mean and standard deviation from elite samples."""
    num_samples = hyperparams['num_samples']
    num_elites = int(num_samples * hyperparams['elite_portion'])
    best_control_idx = torch.argsort(costs)[:num_elites]
    elite_controls = controls[best_control_idx]
    new_mean = torch.mean(elite_controls, dim=0)
    new_stdev = torch.std(elite_controls, dim=0)
    updated_mean = hyperparams['evolution_smoothing'] * old_mean + (
        1 - hyperparams['evolution_smoothing']) * new_mean
    updated_stdev = hyperparams['evolution_smoothing'] * old_stdev + (
        1 - hyperparams['evolution_smoothing']) * new_stdev
    return updated_mean, updated_stdev


def gaussian_samples(random_generator, mean, stdev, control_low, control_high,
                     hyperparams):
    """Samples a batch of controls based on Gaussian distribution.

    Args:
        random_generator: a torch.Generator for reproducible randomness
        mean: mean of control sequence, has dimension (horizon, dim_control).
        stdev: stdev of control sequence, has dimension (horizon, dim_control).
        control_low: lower bound of control space.
        control_high: upper bound of control space.
        hyperparams: dictionary of hyperparameters with following keys: num_samples
          -- number of control sequences to sample sampling_smoothing -- a number in
          [0, 1] to control amount of smoothing,
            see eq. 3-4 in https://arxiv.org/pdf/1907.03613.pdf for more details.

    Returns:
        Array of sampled controls, with dimension (num_samples, horizon,
        dim_control).
    """
    num_samples = hyperparams['num_samples']
    horizon = mean.shape[0]
    dim_control = mean.shape[1]
    device = mean.device
    dtype = mean.dtype

    noises = torch.randn(num_samples, horizon, dim_control, device=device, dtype=dtype, generator=random_generator)

    # Smoothens noise along time axis.
    smoothing_coef = hyperparams['sampling_smoothing']

    for t in range(1, horizon):
        noises[:, t] = smoothing_coef * noises[:, t - 1] + \
                       torch.sqrt(torch.tensor(1 - smoothing_coef**2, device=device, dtype=dtype)) * noises[:, t]

    samples = noises * stdev
    samples = samples + mean
    samples = torch.clamp(samples, control_low, control_high)
    return samples


def cem(cost,
        dynamics,
        init_state,
        init_controls,
        control_low,
        control_high,
        random_seed=None,
        hyperparams=None):
    """Cross Entropy Method (CEM).

    CEM is a sampling-based optimization algorithm. At each iteration, CEM samples
    a batch of candidate actions and computes the mean and standard deviation of
    top-performing samples, which are used to sample from in the next iteration.

    Args:
        cost: cost(x, u, t) returns a scalar
        dynamics: dynamics(x, u, t) returns next state
        init_state: initial state
        init_controls: initial controls, of the shape (horizon, dim_control)
        control_low: lower bound of control space
        control_high: upper bound of control space
        random_seed: random seed for reproducibility
        hyperparams: a dictionary of algorithm hyperparameters

    Returns:
        X: Optimal state trajectory.
        U: Optimized control sequence, an array of shape (horizon, dim_control)
        obj: scalar objective achieved.
    """
    if random_seed is None:
        random_seed = 0

    if hyperparams is None:
        hyperparams = default_cem_hyperparams()

    device = init_controls.device
    dtype = init_controls.dtype
    generator = torch.Generator(device=device).manual_seed(random_seed)

    mean = init_controls.clone()
    stdev = torch.full_like(init_controls, (control_high - control_low) / 2.)

    for _ in range(hyperparams['max_iter']):
        controls = gaussian_samples(generator, mean, stdev, control_low, control_high,
                                    hyperparams)
        costs = torch.stack([objective(cost, dynamics, controls[i], init_state) for i in range(controls.shape[0])])
        mean, stdev = cem_update_mean_stdev(mean, stdev, controls, costs,
                                            hyperparams)

    X = rollout(dynamics, mean, init_state)
    obj = objective(cost, dynamics, mean, init_state)
    return X, mean, obj


def random_shooting(cost,
                    dynamics,
                    init_state,
                    init_controls,
                    control_low,
                    control_high,
                    random_seed=None,
                    hyperparams=None):
    """Random shooting method.

    Random shooting is a very simple optimization procedure where the function
    to be optimized is evaluated at K random points and the point with the lowest
    cost is declared to be the optimal value. This method applies random shooting
    to trajectory optimization.

    Args:
        cost: cost(x, u, t) returns a scalar
        dynamics: dynamics(x, u, t) returns next state
        init_state: initial state
        init_controls: initial controls, of the shape (horizon, dim_control)
        control_low: lower bound of control space
        control_high: upper bound of control space
        random_seed: random seed for reproducibility
        hyperparams: a dictionary of algorithm hyperparameters

    Returns:
        X: Optimal state trajectory.
        U: Optimized control sequence, an array of shape (horizon, dim_control)
        obj: scalar objective achieved.
    """
    if random_seed is None:
        random_seed = 0

    if hyperparams is None:
        hyperparams = default_cem_hyperparams()

    device = init_controls.device
    dtype = init_controls.dtype
    generator = torch.Generator(device=device).manual_seed(random_seed)

    mean = init_controls.clone()
    stdev = torch.full_like(init_controls, (control_high - control_low) / 2.)

    controls = gaussian_samples(generator, mean, stdev, control_low,
                                control_high, hyperparams)
    costs = torch.stack([objective(cost, dynamics, controls[i], init_state) for i in range(controls.shape[0])])
    best_idx = torch.argmin(costs)

    U = controls[best_idx]
    X = rollout(dynamics, mean, init_state)
    obj = objective(cost, dynamics, mean, init_state)
    return X, U, obj


# Constrained Trajectory Optimization


def constrained_ilqr(cost,
                     dynamics,
                     x0,
                     U,
                     equality_constraint=None,
                     inequality_constraint=None,
                     maxiter_al=5,
                     maxiter_ilqr=100,
                     grad_norm_threshold=1.0e-4,
                     relative_grad_norm_threshold=0.0,
                     obj_step_threshold=0.0,
                     inputs_step_threshold=0.0,
                     constraints_threshold=1.0e-2,
                     penalty_init=1.0,
                     penalty_update_rate=10.0,
                     make_psd=False,
                     psd_delta=0.0,
                     alpha_0=1.0,
                     alpha_min=0.00005):
    """Constrained Iterative Linear Quadratic Regulator (PyTorch GPU version).

    Uses augmented Lagrangian method to handle equality and inequality constraints.
    Everything stays on GPU - no CPU transfers or numpy usage.

    Args:
        cost: cost(x, u, t) returns scalar.
        dynamics: dynamics(x, u, t) returns next state (n,) tensor.
        x0: initial state - 1D tensor of shape (n,); should satisfy constraints at t==0.
        U: initial controls - 2D tensor of shape (T, m); does not need to be initially feasible.
        equality_constraint: equality_constraint(x, u, t) == 0 returns (num_equality,) tensor.
                           If None, no equality constraints.
        inequality_constraint: inequality_constraint(x, u, t) <= 0 returns (num_inequality,) tensor.
                             If None, no inequality constraints.
        maxiter_al: maximum number of outer-loop augmented Lagrangian iterations.
        maxiter_ilqr: maximum iterations for iLQR.
        grad_norm_threshold: tolerance for stopping iLQR before augmented Lagrangian update.
        relative_grad_norm_threshold: relative tolerance on gradient norm.
        obj_step_threshold: tolerance on objective value steps.
        inputs_step_threshold: tolerance on input steps.
        constraints_threshold: tolerance for constraint violation (infinity norm).
        penalty_init: initial penalty value.
        penalty_update_rate: rate for increasing penalty.
        make_psd: whether to zero negative eigenvalues after quadratization.
        psd_delta: delta value to make problem PSD.
        alpha_0: initial line search value.
        alpha_min: minimum line search value.

    Returns:
        X: optimal state trajectory - tensor of shape (T+1, n).
        U: optimal control trajectory - tensor of shape (T, m).
        dual_equality: approximate dual (equality) - tensor of shape (T+1, num_equality).
        dual_inequality: approximate dual (inequality) - tensor of shape (T+1, num_inequality).
        penalty: final penalty value.
        equality_constraints: final equality constraint violations - tensor of shape (T+1, num_equality).
        inequality_constraints: final inequality constraint violations - tensor of shape (T+1, num_inequality).
        max_constraint_violation: maximum constraint violation (scalar tensor).
        obj: final augmented Lagrangian objective.
        gradient: gradient at solution.
        iteration_ilqr: cumulative number of iLQR iterations.
        iteration_al: number of augmented Lagrangian iterations.
    """
    device = U.device
    dtype = U.dtype
    T = U.shape[0]
    n = x0.shape[0]
    m = U.shape[1]

    # Default constraint functions if not provided
    if equality_constraint is None:
        def equality_constraint(x, u, t):
            return torch.empty(0, device=device, dtype=dtype)

    if inequality_constraint is None:
        def inequality_constraint(x, u, t):
            return torch.empty(0, device=device, dtype=dtype)

    # Rollout initial trajectory
    X = rollout(dynamics, U, x0)

    # Time range
    horizon = T + 1

    # Determine constraint dimensions by evaluating at t=0
    sample_eq = equality_constraint(x0, torch.zeros(m, device=device, dtype=dtype), 0)
    sample_ineq = inequality_constraint(x0, torch.zeros(m, device=device, dtype=dtype), 0)
    num_equality = sample_eq.numel()
    num_inequality = sample_ineq.numel()

    # Vectorized constraint functions
    def evaluate_constraints(X, U):
        """Evaluate constraints along trajectory."""
        U_pad = pad(U)
        eq_constraints = torch.zeros((horizon, num_equality), device=device, dtype=dtype)
        ineq_constraints = torch.zeros((horizon, num_inequality), device=device, dtype=dtype)

        for t in range(horizon):
            eq_t = equality_constraint(X[t], U_pad[t], t)
            ineq_t = inequality_constraint(X[t], U_pad[t], t)

            if eq_t.numel() > 0:
                eq_constraints[t] = eq_t
            if ineq_t.numel() > 0:
                ineq_constraints[t] = ineq_t

        return eq_constraints, ineq_constraints

    # Initialize constraint evaluations
    equality_constraints, inequality_constraints = evaluate_constraints(X, U)

    # Get constraint dimensions
    num_equality = equality_constraints.shape[1]
    num_inequality = inequality_constraints.shape[1]

    # Initialize dual variables
    dual_equality = torch.zeros((horizon, num_equality), device=device, dtype=dtype)
    dual_inequality = torch.zeros((horizon, num_inequality), device=device, dtype=dtype)

    # Initialize penalty
    penalty = torch.tensor(penalty_init, device=device, dtype=dtype)

    # Counters
    iteration_ilqr = 0
    iteration_al = 0

    # Augmented Lagrangian cost function
    def augmented_lagrangian(x, u, t, dual_eq, dual_ineq, pen):
        """Augmented Lagrangian cost = original cost + penalty terms."""
        # Original cost
        J = cost(x, u, t)

        # Equality constraint contribution
        eq = equality_constraint(x, u, t)
        if eq.numel() > 0:
            J = J + torch.dot(dual_eq[t], eq) + 0.5 * pen * torch.sum(eq ** 2)

        # Inequality constraint contribution
        ineq = inequality_constraint(x, u, t)
        if ineq.numel() > 0:
            # Active set: constraint is active if dual > 0 OR constraint is violated
            active_set = ~((torch.abs(dual_ineq[t]) < 1e-10) & (ineq < 0.0))
            J = J + torch.dot(dual_ineq[t], ineq) + 0.5 * pen * torch.sum((active_set * ineq) ** 2)

        return J

    # Dual update functions
    def dual_update(constraint, dual, pen):
        """Update dual variables."""
        return dual + pen * constraint

    def inequality_projection(dual):
        """Project inequality duals to positive orthant."""
        return torch.maximum(dual, torch.zeros_like(dual))

    # Augmented Lagrangian loop
    max_constraint_violation = torch.tensor(float('inf'), device=device, dtype=dtype)
    obj = torch.tensor(float('inf'), device=device, dtype=dtype)
    gradient = torch.full_like(U, float('inf'))

    while iteration_al < maxiter_al:
        # Create augmented cost with current dual variables and penalty
        def aug_cost(x, u, t):
            return augmented_lagrangian(x, u, t, dual_equality, dual_inequality, penalty)

        # Solve iLQR with augmented cost
        X, U, obj, gradient, _, _, ilqr_iter = ilqr(
            aug_cost,
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
            alpha_min=alpha_min
        )

        # Accumulate iLQR iterations
        iteration_ilqr += ilqr_iter

        # Evaluate constraints at new trajectory
        equality_constraints, inequality_constraints = evaluate_constraints(X, U)

        # Compute constraint violations
        inequality_constraints_projected = torch.maximum(
            inequality_constraints,
            torch.zeros_like(inequality_constraints)
        )

        max_eq_violation = torch.max(torch.abs(equality_constraints)) if num_equality > 0 else torch.tensor(0.0, device=device, dtype=dtype)
        max_ineq_violation = torch.max(inequality_constraints_projected) if num_inequality > 0 else torch.tensor(0.0, device=device, dtype=dtype)
        max_constraint_violation = torch.maximum(max_eq_violation, max_ineq_violation)

        # Compute complementary slackness violation
        if num_inequality > 0:
            max_complementary_slack = torch.max(torch.abs(inequality_constraints * dual_inequality))
        else:
            max_complementary_slack = torch.tensor(0.0, device=device, dtype=dtype)

        # Check convergence
        constraint_satisfied = max_constraint_violation <= constraints_threshold
        complementarity_satisfied = max_complementary_slack <= constraints_threshold

        if constraint_satisfied and complementarity_satisfied:
            break

        # Update dual variables
        dual_equality = dual_update(equality_constraints, dual_equality, penalty)
        dual_inequality = dual_update(inequality_constraints, dual_inequality, penalty)
        dual_inequality = inequality_projection(dual_inequality)

        # Update penalty
        penalty = penalty * penalty_update_rate

        # Increment AL iteration counter
        iteration_al += 1

    return (X, U, dual_equality, dual_inequality, penalty,
            equality_constraints, inequality_constraints,
            max_constraint_violation, obj, gradient,
            iteration_ilqr, iteration_al)
