"""CUDA graph capture tests for the Torch backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import torch

from trajax.torch import optimizers as torch_optim


class TorchCudaGraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for torch backend tests.")
    if not hasattr(torch, "compile"):
      self.skipTest("torch.compile not available.")
    # Torch nightly + Triton nightly combinations can break Inductor compilation
    # of some kernels used in this test (CUDA graph capture requires Inductor).
    try:
      import triton  # type: ignore[import-not-found]
      from packaging import version as _version  # type: ignore[import-not-found]
      if ("a0" in torch.__version__ and
          _version.parse(triton.__version__) >= _version.parse("3.3.0")):
        self.skipTest(
            "Skipping: torch.compile (Inductor) is unstable with triton>=3.3.0 "
            "on this torch nightly build.")
    except ImportError:
      # If Triton or packaging isn't available, proceed (the test will naturally
      # fail/skip based on torch.compile support).
      return

  def test_constrained_ilqr_compiled_cuda_graph(self):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32
    T, n, m = 10, 4, 2

    A = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).repeat(T, 1, 1)
    A = A + 0.01 * torch.randn((T, n, n), device=device, dtype=dtype)
    B = 0.1 * torch.randn((T, n, m), device=device, dtype=dtype)
    Q = torch.eye(n, device=device, dtype=dtype)
    R = 0.1 * torch.eye(m, device=device, dtype=dtype)
    x_goal = torch.randn((n,), device=device, dtype=dtype)
    umax = 0.5 * torch.ones((m,), device=device, dtype=dtype)
    x0 = torch.randn((n,), device=device, dtype=dtype)
    U0 = torch.zeros((T, m), device=device, dtype=dtype)

    workspace = torch_optim.make_constrained_ilqr_linear_quadratic_box_workspace(
        T, n, m, device, dtype)
    workspace.compile_for_cuda_graph()

    def solve(x0_in, U0_in):
      return torch_optim.constrained_ilqr_linear_quadratic_box_graphable(
          x0_in,
          U0_in,
          A,
          B,
          Q,
          R,
          x_goal,
          umax,
          workspace,
          maxiter_al=2,
          maxiter_ilqr=3,
          constraints_threshold=1.0e-2,
          penalty_init=1.0,
          penalty_update_rate=10.0,
          final_weight=0.0,
      )

    compiled = torch.compile(solve, fullgraph=True)
    compiled(x0, U0)
    torch.cuda.synchronize()

    static_x0 = x0.clone()
    static_U0 = U0.clone()
    out_holder = {}
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
      out_holder["out"] = compiled(static_x0, static_U0)

    static_x0.copy_(x0 * 0.9)
    static_U0.copy_(U0)
    graph.replay()

    X, U, *_ = out_holder["out"]
    self.assertEqual(X.shape, (T + 1, n))
    self.assertEqual(U.shape, (T, m))
    self.assertEqual(X.device.type, "cuda")
    self.assertEqual(U.device.type, "cuda")


if __name__ == "__main__":
  absltest.main()
