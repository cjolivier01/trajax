"""Tests for optional Warp CUDA kernels used by the Torch backend."""

from absl.testing import absltest
import torch

from trajax.torch import _warp_kernels


class WarpKernelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available.")
    if not _warp_kernels.is_available():
      self.skipTest("warp-lang not available.")

  def test_box_ineq_active_inplace_matches_torch(self):
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, m = 11, 5
    U = torch.randn((T, m), device=device, dtype=torch.float32)
    umax = 0.5 + torch.rand((m,), device=device, dtype=torch.float32)
    dual_ineq = torch.randn((T + 1, 2 * m), device=device, dtype=torch.float32)

    out_ineq = torch.empty((T + 1, 2 * m), device=device, dtype=torch.float32)
    out_active = torch.empty((T + 1, 2 * m), device=device, dtype=torch.bool)

    _warp_kernels.box_ineq_active_inplace(U, umax, dual_ineq, out_ineq, out_active)

    U_pad = torch.zeros((T + 1, m), device=device, dtype=torch.float32)
    U_pad[:T] = U
    expected_ineq = torch.cat([U_pad - umax, -U_pad - umax], dim=1)
    expected_active = ~((dual_ineq.abs() <= 0.0) & (expected_ineq < 0.0))

    torch.testing.assert_close(out_ineq, expected_ineq)
    self.assertTrue(torch.equal(out_active, expected_active))


if __name__ == "__main__":
  absltest.main()

