# Copyright 2026
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

"""Parity tests between JAX and PyTorch backends."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import jax
from jax import config as jax_config
import jax.numpy as jnp
import numpy as onp
import torch

from trajax import optimizers as jax_optim
from trajax import tvlqr as jax_tvlqr
from trajax.torch import optimizers as torch_optim
from trajax.torch import tvlqr as torch_tvlqr

jax_config.update("jax_enable_x64", True)


def _to_torch(x, dtype=torch.float64):
  return torch.as_tensor(onp.asarray(x), device="cuda", dtype=dtype)


class TorchParityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for torch backend tests.")

  def test_tvlqr_parity(self):
    key = jax.random.PRNGKey(0)
    T, n, m = 32, 6, 4
    key, *ks = jax.random.split(key, 9)
    Q = jax.random.normal(ks[0], (T + 1, n, n), dtype=jnp.float64)
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2)) + 1e-2 * jnp.eye(n, dtype=jnp.float64)
    q = jax.random.normal(ks[1], (T + 1, n), dtype=jnp.float64)
    R = jax.random.normal(ks[2], (T, m, m), dtype=jnp.float64)
    R = 0.5 * (R + jnp.swapaxes(R, -1, -2)) + 1e-2 * jnp.eye(m, dtype=jnp.float64)
    r = jax.random.normal(ks[3], (T, m), dtype=jnp.float64)
    M = jax.random.normal(ks[4], (T, n, m), dtype=jnp.float64)
    A = jax.random.normal(ks[5], (T, n, n), dtype=jnp.float64)
    B = jax.random.normal(ks[6], (T, n, m), dtype=jnp.float64)
    c = jax.random.normal(ks[7], (T, n), dtype=jnp.float64)

    K_j, k_j, P_j, p_j = jax_tvlqr.tvlqr(Q, q, R, r, M, A, B, c)
    K_t, k_t, P_t, p_t = torch_tvlqr.tvlqr(
        _to_torch(Q), _to_torch(q), _to_torch(R), _to_torch(r), _to_torch(M),
        _to_torch(A), _to_torch(B), _to_torch(c))

    self.assertEqual(K_t.device.type, "cuda")
    self.assertEqual(P_t.device.type, "cuda")

    # `torch.linalg.solve` (with LM damping) vs `jax.numpy.linalg.lstsq` can
    # differ at the ~1e-5 level on some random instances.
    atol, rtol = 1e-5, 1e-5
    onp.testing.assert_allclose(onp.asarray(K_j), K_t.detach().cpu().numpy(), atol=atol, rtol=rtol)
    onp.testing.assert_allclose(onp.asarray(k_j), k_t.detach().cpu().numpy(), atol=atol, rtol=rtol)
    onp.testing.assert_allclose(onp.asarray(P_j), P_t.detach().cpu().numpy(), atol=atol, rtol=rtol)
    onp.testing.assert_allclose(onp.asarray(p_j), p_t.detach().cpu().numpy(), atol=atol, rtol=rtol)

  def test_objective_and_grad_parity(self):
    T, n, m = 20, 5, 3
    rng = onp.random.RandomState(0)
    H = rng.randn(T + 1, n + m, n + m)
    H = H + onp.transpose(H, (0, 2, 1))
    h = rng.randn(T + 1, n + m)
    A = rng.randn(T, n, n)
    B = rng.randn(T, n, m)
    x0 = rng.randn(n)
    U0 = rng.randn(T, m)

    H_j = jnp.asarray(H, dtype=jnp.float64)
    h_j = jnp.asarray(h, dtype=jnp.float64)
    A_j = jnp.asarray(A, dtype=jnp.float64)
    B_j = jnp.asarray(B, dtype=jnp.float64)

    def cost_j(x, u, t, params):
      z = jnp.concatenate([x, u])
      return params[0] * 0.5 * z.T @ (H_j[t] @ z) + params[1] * (h_j[t] @ z)

    def dyn_j(x, u, t, params):
      return params[0] * (A_j[t] @ x) + params[1] * (B_j[t] @ u)

    params = (jnp.array(1.7, dtype=jnp.float64), jnp.array(0.9, dtype=jnp.float64))
    U_j = jnp.asarray(U0, dtype=jnp.float64)
    x0_j = jnp.asarray(x0, dtype=jnp.float64)
    cost_j_closed = lambda x, u, t: cost_j(x, u, t, params)
    dyn_j_closed = lambda x, u, t: dyn_j(x, u, t, params)
    obj_j = jax_optim.objective(cost_j_closed, dyn_j_closed, U_j, x0_j)
    grad_j = jax_optim.grad_wrt_controls(cost_j_closed, dyn_j_closed, U_j, x0_j, (), ())

    H_t = _to_torch(H)
    h_t = _to_torch(h)
    A_t = _to_torch(A)
    B_t = _to_torch(B)

    def cost_t(x, u, t, params_t):
      z = torch.cat([x, u], dim=0)
      Ht = H_t[t]
      ht = h_t[t]
      return params_t[0] * 0.5 * (z @ (Ht @ z)) + params_t[1] * (ht @ z)

    def dyn_t(x, u, t, params_t):
      At = A_t[t]
      Bt = B_t[t]
      return params_t[0] * (At @ x) + params_t[1] * (Bt @ u)

    params_t = (_to_torch(onp.array(1.7)), _to_torch(onp.array(0.9)))
    U_t = _to_torch(U0)
    x0_t = _to_torch(x0)
    cost_t_closed = lambda x, u, t: cost_t(x, u, t, params_t)
    dyn_t_closed = lambda x, u, t: dyn_t(x, u, t, params_t)
    obj_t = torch_optim.objective(cost_t_closed, dyn_t_closed, U_t, x0_t)
    grad_t = torch_optim.grad_wrt_controls(cost_t_closed, dyn_t_closed, U_t, x0_t)

    onp.testing.assert_allclose(onp.asarray(obj_j), float(obj_t.detach().cpu().numpy()), atol=1e-5, rtol=1e-5)
    onp.testing.assert_allclose(onp.asarray(grad_j), grad_t.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

  def test_ilqr_parity_linear_quadratic(self):
    T, n, m = 30, 4, 2
    rng = onp.random.RandomState(0)
    A = rng.randn(T, n, n) * 0.05
    for t in range(T):
      A[t] += onp.eye(n)
    B = rng.randn(T, n, m) * 0.1
    Q = onp.eye(n)
    R = 0.1 * onp.eye(m)
    x_goal = rng.randn(n)

    A_j = jnp.asarray(A, dtype=jnp.float64)
    B_j = jnp.asarray(B, dtype=jnp.float64)
    Q_j = jnp.asarray(Q, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    x_goal_j = jnp.asarray(x_goal, dtype=jnp.float64)

    def cost_j(x, u, t):
      dx = x - x_goal_j
      stage = 0.5 * (dx @ (Q_j @ dx) + u @ (R_j @ u))
      final = 10.0 * 0.5 * (dx @ (Q_j @ dx))
      return jnp.where(t == T, final, stage)

    def dyn_j(x, u, t):
      return (A_j[t] @ x) + (B_j[t] @ u)

    x0 = rng.randn(n)
    U0 = onp.zeros((T, m))
    X_j, U_j, obj_j, *_ = jax_optim.ilqr(cost_j, dyn_j, jnp.asarray(x0, dtype=jnp.float64),
                                         jnp.asarray(U0, dtype=jnp.float64),
                                         maxiter=25, alpha_0=1.0, alpha_min=1e-4)

    A_t = _to_torch(A)
    B_t = _to_torch(B)
    Q_t = _to_torch(Q)
    R_t = _to_torch(R)
    x_goal_t = _to_torch(x_goal)
    T_t = torch.tensor(T, device="cuda", dtype=torch.int64)

    def cost_t(x, u, t):
      dx = x - x_goal_t
      stage = 0.5 * (dx @ (Q_t @ dx) + u @ (R_t @ u))
      final = 10.0 * 0.5 * (dx @ (Q_t @ dx))
      return torch.where(t == T_t, final, stage)

    def dyn_t(x, u, t):
      return (A_t[t] @ x) + (B_t[t] @ u)

    X_t, U_t, obj_t, *_ = torch_optim.ilqr(cost_t, dyn_t, _to_torch(x0), _to_torch(U0),
                                           maxiter=25, alpha_0=1.0, alpha_min=1e-4)

    self.assertEqual(X_t.device.type, "cuda")
    self.assertEqual(U_t.device.type, "cuda")
    onp.testing.assert_allclose(onp.asarray(obj_j), float(obj_t.detach().cpu().numpy()), atol=1e-3, rtol=1e-3)
    onp.testing.assert_allclose(onp.asarray(U_j), U_t.detach().cpu().numpy(), atol=1e-2, rtol=1e-2)

  def test_ilqr_vmap_parity_linear_quadratic(self):
    T, n, m = 12, 3, 2
    batch = 2
    rng = onp.random.RandomState(0)
    A = rng.randn(T, n, n) * 0.05
    for t in range(T):
      A[t] += onp.eye(n)
    B = rng.randn(T, n, m) * 0.1
    Q = onp.eye(n)
    R = 0.1 * onp.eye(m)
    x_goal = rng.randn(n)

    A_j = jnp.asarray(A, dtype=jnp.float64)
    B_j = jnp.asarray(B, dtype=jnp.float64)
    Q_j = jnp.asarray(Q, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    x_goal_j = jnp.asarray(x_goal, dtype=jnp.float64)

    def cost_j(x, u, t):
      dx = x - x_goal_j
      stage = 0.5 * (dx @ (Q_j @ dx) + u @ (R_j @ u))
      final = 10.0 * 0.5 * (dx @ (Q_j @ dx))
      return jnp.where(t == T, final, stage)

    def dyn_j(x, u, t):
      return (A_j[t] @ x) + (B_j[t] @ u)

    x0 = rng.randn(batch, n)
    U0 = rng.randn(batch, T, m) * 0.1
    solve_j = lambda x0_i, U0_i: jax_optim.ilqr(
        cost_j,
        dyn_j,
        x0_i,
        U0_i,
        maxiter=10,
        alpha_0=1.0,
        alpha_min=1e-4,
    )
    X_j, U_j, obj_j, *_ = jax.vmap(solve_j)(jnp.asarray(x0, dtype=jnp.float64),
                                            jnp.asarray(U0, dtype=jnp.float64))

    A_t = _to_torch(A)
    B_t = _to_torch(B)
    Q_t = _to_torch(Q)
    R_t = _to_torch(R)
    x_goal_t = _to_torch(x_goal)
    T_t = torch.tensor(T, device="cuda", dtype=torch.int64)

    def cost_t(x, u, t):
      dx = x - x_goal_t
      stage = 0.5 * (dx @ (Q_t @ dx) + u @ (R_t @ u))
      final = 10.0 * 0.5 * (dx @ (Q_t @ dx))
      return torch.where(t == T_t, final, stage)

    def dyn_t(x, u, t):
      return (A_t[t] @ x) + (B_t[t] @ u)

    def solve_t(x0_i, U0_i):
      return torch_optim.ilqr(
          cost_t,
          dyn_t,
          x0_i,
          U0_i,
          maxiter=10,
          alpha_0=1.0,
          alpha_min=1e-4,
          vmap_safe=True,
      )

    X_t, U_t, obj_t, *_ = torch.vmap(solve_t)(_to_torch(x0), _to_torch(U0))

    onp.testing.assert_allclose(onp.asarray(obj_j),
                                obj_t.detach().cpu().numpy(),
                                atol=1e-3,
                                rtol=1e-3)
    onp.testing.assert_allclose(onp.asarray(U_j),
                                U_t.detach().cpu().numpy(),
                                atol=1e-2,
                                rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
