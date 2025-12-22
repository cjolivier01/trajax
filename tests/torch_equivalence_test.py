# Copyright 2024 Google LLC
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

"""Equivalence tests between JAX and PyTorch implementations."""

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.random as jrandom
import jax.numpy as jnp
import torch

from trajax import integrators as jax_integrators
from trajax import optimizers as jax_optimizers
from trajax.torch import integrators as torch_integrators
from trajax.torch import optimizers as torch_optimizers
from trajax.torch import tvlqr as torch_tvlqr
from trajax import tvlqr as jax_tvlqr


torch.set_default_dtype(torch.float32)


class TorchEquivalenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    onp.random.seed(0)
    torch.manual_seed(0)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def test_integrators_match(self):
    dt = 0.1

    def dyn_jax(x, u, t):
      del t
      return x + u

    def dyn_torch(x, u, t):
      del t
      return x + u

    x0_np = jnp.array([1.0, 2.0], dtype=jnp.float32)
    u_np = jnp.array([0.3, -0.1], dtype=jnp.float32)
    x0_torch = torch.tensor([1.0, 2.0], device=self.device, dtype=torch.float32)
    u_torch = torch.tensor([0.3, -0.1], device=self.device,
                           dtype=torch.float32)

    euler_jax = jax_integrators.euler(dyn_jax, dt)
    euler_torch = torch_integrators.euler(dyn_torch, dt)
    rk4_jax = jax_integrators.rk4(dyn_jax, dt)
    rk4_torch = torch_integrators.rk4(dyn_torch, dt)

    euler_out = euler_torch(x0_torch, u_torch, 0)
    rk4_out = rk4_torch(x0_torch, u_torch, 0)

    onp.testing.assert_allclose(euler_jax(x0_np, u_np, 0),
                                euler_out.detach().cpu().numpy(),
                                rtol=1e-6, atol=1e-6)
    onp.testing.assert_allclose(rk4_jax(x0_np, u_np, 0),
                                rk4_out.detach().cpu().numpy(),
                                rtol=1e-6, atol=1e-6)
    self.assertEqual(euler_out.device.type, self.device.type)
    self.assertEqual(rk4_out.device.type, self.device.type)

  def test_rollout_and_objective_match(self):
    T, n, m = 4, 3, 2
    A = onp.random.randn(T, n, n).astype(onp.float32)
    B = onp.random.randn(T, n, m).astype(onp.float32)
    Q = onp.stack([onp.eye(n, dtype=onp.float32) for _ in range(T + 1)])
    R = onp.stack([onp.eye(m, dtype=onp.float32) for _ in range(T + 1)])

    x0_np = jnp.array(onp.random.randn(n), dtype=jnp.float32)
    U_np = jnp.array(onp.random.randn(T, m), dtype=jnp.float32)
    x0_torch = torch.tensor(onp.asarray(x0_np), device=self.device,
                            dtype=torch.float32)
    U_torch = torch.tensor(onp.asarray(U_np), device=self.device,
                           dtype=torch.float32)
    A_torch = torch.tensor(A, device=self.device, dtype=torch.float32)
    B_torch = torch.tensor(B, device=self.device, dtype=torch.float32)
    Q_torch = torch.tensor(Q, device=self.device, dtype=torch.float32)
    R_torch = torch.tensor(R, device=self.device, dtype=torch.float32)

    A_jax = jnp.array(A)
    B_jax = jnp.array(B)
    Q_jax = jnp.array(Q)
    R_jax = jnp.array(R)

    def dyn_jax(x, u, t):
      return A_jax[t] @ x + B_jax[t] @ u

    def dyn_torch(x, u, t):
      return torch.matmul(A_torch[t], x) + torch.matmul(B_torch[t], u)

    def cost_jax(x, u, t):
      return 0.5 * x.T @ (Q_jax[t] @ x) + 0.5 * u.T @ (R_jax[t] @ u)

    def cost_torch(x, u, t):
      return 0.5 * torch.dot(x, torch.matmul(Q_torch[t], x)) + 0.5 * torch.dot(
          u, torch.matmul(R_torch[t], u))

    obj_jax = jax_optimizers.objective(cost_jax, dyn_jax, U_np, x0_np)
    obj_torch = torch_optimizers.objective(cost_torch, dyn_torch, U_torch,
                                           x0_torch)

    onp.testing.assert_allclose(obj_jax, obj_torch.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)

    X_torch = torch_optimizers.rollout(dyn_torch, U_torch, x0_torch)
    self.assertEqual(X_torch.device.type, self.device.type)

  def test_linearize_and_quadratize_match(self):
    T, n, m = 3, 2, 1
    A = onp.random.randn(T, n, n).astype(onp.float32)
    B = onp.random.randn(T, n, m).astype(onp.float32)
    H = onp.random.randn(T + 1, n + m, n + m).astype(onp.float32)
    # Symmetrize for stability.
    H = H + onp.transpose(H, (0, 2, 1))
    x0_np = jnp.array(onp.random.randn(n), dtype=jnp.float32)
    U_np = jnp.array(onp.random.randn(T, m), dtype=jnp.float32)
    A_pad = onp.concatenate([A, onp.zeros((1, n, n), dtype=onp.float32)], axis=0)
    B_pad = onp.concatenate([B, onp.zeros((1, n, m), dtype=onp.float32)], axis=0)
    A_jax = jnp.array(A_pad)
    B_jax = jnp.array(B_pad)
    H_jax = jnp.array(H)
    X_np = jax_optimizers.rollout(
        lambda x, u, t: A_jax[t] @ x + B_jax[t] @ u, U_np, x0_np)
    times_np = jnp.arange(X_np.shape[0])

    A_torch = torch.tensor(A_pad, device=self.device, dtype=torch.float32)
    B_torch = torch.tensor(B_pad, device=self.device, dtype=torch.float32)
    H_torch = torch.tensor(H, device=self.device, dtype=torch.float32)
    X_torch = torch.tensor(onp.asarray(X_np), device=self.device,
                           dtype=torch.float32)
    U_torch = torch.tensor(onp.asarray(U_np), device=self.device,
                           dtype=torch.float32)
    times_torch = torch.arange(X_torch.shape[0], device=self.device)

    def dynamics_jax(x, u, t):
      return A_jax[t] @ x + B_jax[t] @ u

    def dynamics_torch(x, u, t):
      return torch.matmul(A_torch[t], x) + torch.matmul(B_torch[t], u)

    def cost_jax(x, u, t):
      z = jnp.concatenate([x, u])
      return 0.5 * z.T @ (H_jax[t] @ z)

    def cost_torch(x, u, t):
      z = torch.cat([x, u])
      return 0.5 * torch.dot(z, torch.matmul(H_torch[t], z))

    jx, ju = jax_optimizers.linearize(dynamics_jax)(X_np, jax_optimizers.pad(U_np),
                                                    times_np)
    jx_t, ju_t = torch_optimizers.linearize(dynamics_torch)(
        X_torch, torch_optimizers.pad(U_torch), times_torch)
    onp.testing.assert_allclose(jx, jx_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(ju, ju_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)

    Q_np, R_np, M_np = jax_optimizers.quadratize(cost_jax)(X_np,
                                                          jax_optimizers.pad(U_np),
                                                          times_np)
    Q_t, R_t, M_t = torch_optimizers.quadratize(cost_torch)(
        X_torch, torch_optimizers.pad(U_torch), times_torch)
    onp.testing.assert_allclose(Q_np, Q_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(R_np, R_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(M_np, M_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)

  def test_tvlqr_matches(self):
    T, n, m = 3, 2, 1
    Q = onp.stack([onp.eye(n, dtype=onp.float32) for _ in range(T + 1)])
    q = onp.random.randn(T + 1, n).astype(onp.float32)
    R = onp.stack([onp.eye(m, dtype=onp.float32) for _ in range(T + 1)])
    r = onp.random.randn(T + 1, m).astype(onp.float32)
    M = onp.random.randn(T + 1, n, m).astype(onp.float32)
    A = onp.random.randn(T, n, n).astype(onp.float32)
    B = onp.random.randn(T, n, m).astype(onp.float32)
    c = onp.zeros((T, n), dtype=onp.float32)

    K_jax, k_jax, P_jax, p_jax = jax_tvlqr.tvlqr(
        jnp.array(Q), jnp.array(q), jnp.array(R), jnp.array(r), jnp.array(M),
        jnp.array(A), jnp.array(B), jnp.array(c))

    Q_t = torch.tensor(Q, device=self.device, dtype=torch.float32)
    q_t = torch.tensor(q, device=self.device, dtype=torch.float32)
    R_t = torch.tensor(R, device=self.device, dtype=torch.float32)
    r_t = torch.tensor(r, device=self.device, dtype=torch.float32)
    M_t = torch.tensor(M, device=self.device, dtype=torch.float32)
    A_t = torch.tensor(A, device=self.device, dtype=torch.float32)
    B_t = torch.tensor(B, device=self.device, dtype=torch.float32)
    c_t = torch.tensor(c, device=self.device, dtype=torch.float32)

    K_t, k_t, P_t, p_t = torch_tvlqr.tvlqr(Q_t, q_t, R_t, r_t, M_t, A_t, B_t,
                                           c_t)

    onp.testing.assert_allclose(K_jax, K_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(k_jax, k_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(P_jax, P_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    onp.testing.assert_allclose(p_jax, p_t.detach().cpu().numpy(), rtol=1e-5,
                                atol=1e-5)
    self.assertEqual(K_t.device.type, self.device.type)
    self.assertEqual(k_t.device.type, self.device.type)

  def test_ilqr_parity(self):
    T, n, m = 5, 3, 2
    A = onp.random.randn(n, n).astype(onp.float32)
    B = onp.random.randn(n, m).astype(onp.float32)
    Q = jnp.eye(n, dtype=jnp.float32)
    R = jnp.eye(m, dtype=jnp.float32)

    def dynamics_jax(x, u, t):
      del t
      return A @ x + B @ u

    def cost_jax(x, u, t):
      del t
      return 0.5 * x.T @ (Q @ x) + 0.1 * u.T @ (R @ u)

    def dynamics_torch(x, u, t):
      del t
      return torch.tensor(A, device=self.device) @ x + torch.tensor(
          B, device=self.device) @ u

    def cost_torch(x, u, t):
      del t
      return 0.5 * torch.dot(x, torch.matmul(torch.eye(n, device=self.device),
                                             x)) + 0.1 * torch.dot(
                                                 u, torch.matmul(
                                                     torch.eye(
                                                         m, device=self.device),
                                                     u))

    x0_np = jnp.array(onp.random.randn(n), dtype=jnp.float32)
    U_np = jnp.zeros((T, m), dtype=jnp.float32)
    x0_torch = torch.tensor(onp.asarray(x0_np), device=self.device,
                            dtype=torch.float32)
    U_torch = torch.zeros((T, m), device=self.device, dtype=torch.float32)

    X_jax, U_jax, obj_jax, *_ = jax_optimizers.ilqr(
        cost_jax, dynamics_jax, x0_np, U_np, maxiter=10, make_psd=True)

    X_torch, U_torch_opt, obj_torch, *_ = torch_optimizers.ilqr(
        cost_torch, dynamics_torch, x0_torch, U_torch, maxiter=10,
        make_psd=True)

    onp.testing.assert_allclose(X_jax, X_torch.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    onp.testing.assert_allclose(U_jax, U_torch_opt.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    onp.testing.assert_allclose(obj_jax, obj_torch.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)

  def test_random_shooting_parity(self):
    T, n, m = 3, 2, 1
    A = onp.random.randn(n, n).astype(onp.float32)
    B = onp.random.randn(n, m).astype(onp.float32)
    Q = jnp.eye(n, dtype=jnp.float32)
    R = jnp.eye(m, dtype=jnp.float32)

    def dynamics_jax(x, u, t):
      del t
      return A @ x + B @ u

    def cost_jax(x, u, t):
      del t
      return 0.5 * x.T @ (Q @ x) + 0.1 * u.T @ (R @ u)

    def dynamics_torch(x, u, t):
      del t
      return torch.tensor(A, device=self.device) @ x + torch.tensor(
          B, device=self.device) @ u

    def cost_torch(x, u, t):
      del t
      return 0.5 * torch.dot(x, torch.matmul(torch.eye(n, device=self.device),
                                             x)) + 0.1 * torch.dot(
                                                 u, torch.matmul(
                                                     torch.eye(
                                                         m, device=self.device),
                                                     u))

    x0_np = jnp.array(onp.random.randn(n), dtype=jnp.float32)
    x0_torch = torch.tensor(onp.asarray(x0_np), device=self.device,
                            dtype=torch.float32)
    init_controls_np = jnp.zeros((T, m), dtype=jnp.float32)
    init_controls_torch = torch.zeros((T, m), device=self.device,
                                      dtype=torch.float32)
    control_low_np = jnp.zeros((T, m), dtype=jnp.float32)
    control_high_np = jnp.zeros((T, m), dtype=jnp.float32)
    control_low_torch = torch.zeros((T, m), device=self.device,
                                    dtype=torch.float32)
    control_high_torch = torch.zeros((T, m), device=self.device,
                                     dtype=torch.float32)
    expected_X = jax_optimizers.rollout(dynamics_jax, init_controls_np, x0_np)
    expected_obj = jax_optimizers.objective(cost_jax, dynamics_jax,
                                            init_controls_np, x0_np)

    X_torch, U_torch_opt, obj_torch = torch_optimizers.random_shooting(
        cost_torch, dynamics_torch, x0_torch, init_controls_torch,
        control_low_torch, control_high_torch, generator=None,
        hyperparams=None)

    onp.testing.assert_allclose(expected_X, X_torch.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(init_controls_np,
                                U_torch_opt.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(expected_obj, obj_torch.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)

  def test_cem_parity(self):
    T, n, m = 3, 2, 1
    A = onp.random.randn(n, n).astype(onp.float32)
    B = onp.random.randn(n, m).astype(onp.float32)
    Q = jnp.eye(n, dtype=jnp.float32)
    R = jnp.eye(m, dtype=jnp.float32)

    def dynamics_jax(x, u, t):
      del t
      return A @ x + B @ u

    def cost_jax(x, u, t):
      del t
      return 0.5 * x.T @ (Q @ x) + 0.1 * u.T @ (R @ u)

    def dynamics_torch(x, u, t):
      del t
      return torch.tensor(A, device=self.device) @ x + torch.tensor(
          B, device=self.device) @ u

    def cost_torch(x, u, t):
      del t
      return 0.5 * torch.dot(x, torch.matmul(torch.eye(n, device=self.device),
                                             x)) + 0.1 * torch.dot(
                                                 u, torch.matmul(
                                                     torch.eye(
                                                         m, device=self.device),
                                                     u))

    x0_np = jnp.array(onp.random.randn(n), dtype=jnp.float32)
    x0_torch = torch.tensor(onp.asarray(x0_np), device=self.device,
                            dtype=torch.float32)
    init_controls_np = jnp.zeros((T, m), dtype=jnp.float32)
    init_controls_torch = torch.zeros((T, m), device=self.device,
                                      dtype=torch.float32)
    control_low_np = jnp.zeros((T, m), dtype=jnp.float32)
    control_high_np = jnp.zeros((T, m), dtype=jnp.float32)
    control_low_torch = torch.zeros((T, m), device=self.device,
                                    dtype=torch.float32)
    control_high_torch = torch.zeros((T, m), device=self.device,
                                     dtype=torch.float32)
    expected_X = jax_optimizers.rollout(dynamics_jax, init_controls_np, x0_np)
    expected_obj = jax_optimizers.objective(cost_jax, dynamics_jax,
                                            init_controls_np, x0_np)

    X_torch, U_torch_opt, obj_torch = torch_optimizers.cem(
        cost_torch, dynamics_torch, x0_torch, init_controls_torch,
        control_low_torch, control_high_torch, generator=None,
        hyperparams=None)

    onp.testing.assert_allclose(expected_X, X_torch.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(init_controls_np,
                                U_torch_opt.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(expected_obj, obj_torch.detach().cpu().numpy(),
                                rtol=1e-5, atol=1e-5)

  def test_torch_ilqr_device_dtype_smoke(self):
    T, n, m = 4, 2, 1
    A = torch.eye(n, device=self.device, dtype=torch.float32)
    B = torch.ones((n, m), device=self.device, dtype=torch.float32)

    def dynamics_torch(x, u, t):
      del t
      return A @ x + B @ u

    def cost_torch(x, u, t):
      del t
      return 0.5 * torch.dot(x, x) + 0.1 * torch.dot(u, u)

    x0 = torch.ones((n,), device=self.device, dtype=torch.float32)
    U0 = torch.zeros((T, m), device=self.device, dtype=torch.float32)

    X, U, obj, grad, adjoints, _, _ = torch_optimizers.ilqr(
        cost_torch, dynamics_torch, x0, U0, maxiter=5, make_psd=True)

    self.assertEqual(X.device.type, self.device.type)
    self.assertEqual(U.device.type, self.device.type)
    self.assertEqual(obj.device.type, self.device.type)
    self.assertEqual(grad.device.type, self.device.type)

  def test_constrained_ilqr_parity(self):
    T, n, m = 3, 1, 1
    A = 1.0
    B = 1.0

    def dynamics_jax(x, u, t):
      del t
      return A * x + B * u

    def cost_jax(x, u, t):
      del t
      return 0.5 * jnp.sum(x**2) + 0.1 * jnp.sum(u**2)

    def equality_constraint_jax(x, u, t):
      del u, t
      return x

    def inequality_constraint_jax(x, u, t):
      del x, u, t
      return jnp.zeros(1, dtype=jnp.float32)

    def dynamics_torch(x, u, t):
      del t
      return torch.tensor(A, device=self.device) * x + torch.tensor(
          B, device=self.device) * u

    def cost_torch(x, u, t):
      del t
      return 0.5 * torch.sum(x**2) + 0.1 * torch.sum(u**2)

    def equality_constraint_torch(x, u, t):
      del u, t
      return x

    def inequality_constraint_torch(x, u, t):
      del x, u, t
      return torch.zeros(1, device=self.device, dtype=torch.float32)

    x0_np = jnp.array([1.0], dtype=jnp.float32)
    U_np = jnp.zeros((T, m), dtype=jnp.float32)
    x0_torch = torch.tensor(onp.asarray(x0_np), device=self.device,
                            dtype=torch.float32)
    U_torch = torch.zeros((T, m), device=self.device, dtype=torch.float32)

    X_jax, U_jax, dual_eq_jax, dual_ineq_jax, penalty_jax, eq_constr_jax, ineq_constr_jax, max_violation_jax, obj_jax, grad_jax, iter_ilqr_jax, iter_al_jax = (
        jax_optimizers.constrained_ilqr(
            cost_jax,
            dynamics_jax,
            x0_np,
            U_np,
            equality_constraint=equality_constraint_jax,
            inequality_constraint=inequality_constraint_jax,
            maxiter_al=3,
            maxiter_ilqr=10,
            constraints_threshold=1e-4,
            make_psd=True))

    X_torch, U_torch_opt, dual_eq_torch, dual_ineq_torch, penalty_torch, eq_constr_torch, ineq_constr_torch, max_violation_torch, obj_torch, grad_torch, iter_ilqr_torch, iter_al_torch = (
        torch_optimizers.constrained_ilqr(
            cost_torch,
            dynamics_torch,
            x0_torch,
            U_torch,
            equality_constraint=equality_constraint_torch,
            inequality_constraint=inequality_constraint_torch,
            maxiter_al=3,
            maxiter_ilqr=10,
            constraints_threshold=1e-4,
            make_psd=True))

    onp.testing.assert_allclose(X_jax, X_torch.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    onp.testing.assert_allclose(U_jax, U_torch_opt.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    onp.testing.assert_allclose(eq_constr_jax,
                                eq_constr_torch.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    onp.testing.assert_allclose(obj_jax, obj_torch.detach().cpu().numpy(),
                                rtol=1e-3, atol=1e-3)
    self.assertEqual(X_torch.device.type, self.device.type)
    self.assertEqual(U_torch_opt.device.type, self.device.type)


if __name__ == '__main__':
  absltest.main()
