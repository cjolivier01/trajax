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

"""CUDA graph capture smoke tests for the Torch backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import torch

from trajax.torch import optimizers as torch_optim


class TorchCUDAGraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available.")

  def test_constrained_ilqr_lq_box_graphable(self):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    T, n, m = 32, 8, 6
    dtype = torch.float32
    device = torch.device("cuda")

    ws = torch_optim.make_constrained_ilqr_linear_quadratic_box_workspace(
        T, n, m, device=device, dtype=dtype)
    ws.compile_for_cuda_graph(delta=1e-6)

    # Static input buffers for capture.
    x0 = torch.empty((n,), device=device, dtype=dtype)
    U0 = torch.empty((T, m), device=device, dtype=dtype)

    # Static problem data.
    A = 0.95 * torch.eye(n, device=device, dtype=dtype)
    B = 0.1 * torch.randn((n, m), device=device, dtype=dtype)
    Q = torch.eye(n, device=device, dtype=dtype)
    R = 0.1 * torch.eye(m, device=device, dtype=dtype)
    x_goal = torch.randn((n,), device=device, dtype=dtype)
    umax = torch.ones((m,), device=device, dtype=dtype)

    def run():
      return torch_optim.constrained_ilqr_linear_quadratic_box_graphable(
          x0=x0,
          U0=U0,
          A=A,
          B=B,
          Q=Q,
          R=R,
          x_goal=x_goal,
          umax=umax,
          workspace=ws,
          maxiter_al=1,
          maxiter_ilqr=2,
          final_weight=0.0,
          delta=1e-6,
      )

    # Warm up.
    x0.copy_(torch.randn((n,), device=device, dtype=dtype))
    U0.copy_(0.2 * torch.randn((T, m), device=device, dtype=dtype))
    for _ in range(3):
      run()
    torch.cuda.synchronize()

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      run()
    torch.cuda.synchronize()

    U_prev = ws.U.clone()

    # Replay with updated inputs.
    x0.copy_(torch.randn((n,), device=device, dtype=dtype))
    U0.copy_(0.2 * torch.randn((T, m), device=device, dtype=dtype))
    g.replay()
    torch.cuda.synchronize()

    # Ensure the replay produced a different output.
    self.assertGreater(torch.linalg.vector_norm(ws.U - U_prev).item(), 0.0)


if __name__ == "__main__":
  absltest.main()
