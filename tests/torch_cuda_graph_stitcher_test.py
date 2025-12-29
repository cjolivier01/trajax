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

"""Tests for CUDA graph stitching with eager breaks."""

from absl.testing import absltest
import torch

from trajax.torch.cuda_graph_stitcher import CUDAGraphStitcher


class CUDAGraphStitcherTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_stitcher_with_eager_break(self):
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for CUDAGraphStitcher tests.")

    device = torch.device("cuda")
    stitcher = CUDAGraphStitcher(device=device, warmup=2)

    def block1(x):
      return torch.sin(x) + 0.5

    def block2(x):
      return x * 2.0

    x = torch.randn(8, device=device, dtype=torch.float32)

    seg1 = stitcher.wrap(block1)
    seg2 = stitcher.wrap(block2)

    with stitcher.capture():
      y = seg1(x)
      with stitcher.eager() as eager:
        y = eager(lambda t: t + 1.0, y)
        y = eager(lambda t: t * 3.0, y)
      _ = seg2(y)

    for scale in (1.0, 2.0):
      inp = x * scale
      out = stitcher.replay(inp)
      expected = block2((block1(inp) + 1.0) * 3.0)
      torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

  def test_training_step_with_eager_break(self):
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for CUDAGraphStitcher tests.")

    device = torch.device("cuda")
    stitcher = CUDAGraphStitcher(device=device, warmup=1, training_warmup=1)

    lin1 = torch.nn.Linear(4, 4).to(device=device, dtype=torch.float32)
    lin2 = torch.nn.Linear(4, 2).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(
        list(lin1.parameters()) + list(lin2.parameters()), lr=0.1
    )
    loss_fn = torch.nn.MSELoss()

    def segment1(x):
      return torch.relu(lin1(x))

    def segment2(y, target):
      optimizer.zero_grad(set_to_none=False)
      out = lin2(y)
      loss = loss_fn(out, target)
      loss.backward()
      optimizer.step()
      return loss

    x = torch.randn(3, 4, device=device, dtype=torch.float32)
    target1 = torch.randn(3, 2, device=device, dtype=torch.float32)
    target2 = torch.randn(3, 2, device=device, dtype=torch.float32)

    with stitcher.capture(training=True):
      y = stitcher.run(segment1, x)
      with stitcher.eager() as eager:
        with torch.no_grad():
          y = eager(lambda t: t + 0.25, y)
      _ = stitcher.run(segment2, y, target1)

    lin1_weight = lin1.weight.detach().clone()
    lin2_weight = lin2.weight.detach().clone()

    def expected_loss(inp, tgt):
      with torch.no_grad():
        y = torch.relu(lin1(inp))
        y = y + 0.25
        out = lin2(y)
        return loss_fn(out, tgt)

    expected1 = expected_loss(x, target1)
    loss1 = stitcher.replay(x, target1)
    torch.testing.assert_close(loss1, expected1, rtol=1e-4, atol=1e-4)

    expected2 = expected_loss(x, target2)
    loss2 = stitcher.replay(x, target2)
    torch.testing.assert_close(loss2, expected2, rtol=1e-4, atol=1e-4)

    self.assertTrue(torch.allclose(lin1.weight, lin1_weight))
    self.assertFalse(torch.allclose(lin2.weight, lin2_weight))

  def test_training_step_wrapper(self):
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for CUDAGraphStitcher tests.")

    device = torch.device("cuda")
    stitcher = CUDAGraphStitcher(device=device, warmup=1, training_warmup=1)

    lin1 = torch.nn.Linear(4, 4).to(device=device, dtype=torch.float32)
    lin2 = torch.nn.Linear(4, 2).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(
        list(lin1.parameters()) + list(lin2.parameters()), lr=0.1
    )
    loss_fn = torch.nn.MSELoss()

    def segment1(x):
      return torch.relu(lin1(x))

    def segment2(y, target):
      optimizer.zero_grad(set_to_none=False)
      out = lin2(y)
      loss = loss_fn(out, target)
      loss.backward()
      optimizer.step()
      return loss

    def step(x, target):
      y = stitcher.run(segment1, x)
      with stitcher.eager() as eager:
        with torch.no_grad():
          y = eager(lambda t: t + 0.5, y)
      return stitcher.run(segment2, y, target)

    train_step = stitcher.make_training_step(step)

    x = torch.randn(3, 4, device=device, dtype=torch.float32)
    target1 = torch.randn(3, 2, device=device, dtype=torch.float32)
    target2 = torch.randn(3, 2, device=device, dtype=torch.float32)

    _ = train_step(x, target1)

    lin1_weight = lin1.weight.detach().clone()
    lin2_weight = lin2.weight.detach().clone()

    def expected_loss(inp, tgt):
      with torch.no_grad():
        y = torch.relu(lin1(inp))
        y = y + 0.5
        out = lin2(y)
        return loss_fn(out, tgt)

    expected2 = expected_loss(x, target2)
    loss2 = train_step(x, target2)
    torch.testing.assert_close(loss2, expected2, rtol=1e-4, atol=1e-4)

    self.assertTrue(torch.allclose(lin1.weight, lin1_weight))
    self.assertFalse(torch.allclose(lin2.weight, lin2_weight))

  def test_reuse_static_inputs(self):
    if not torch.cuda.is_available():
      self.skipTest("CUDA not available for CUDAGraphStitcher tests.")

    device = torch.device("cuda")
    stitcher = CUDAGraphStitcher(device=device, reuse_static_inputs=True)

    def block(x):
      return x + 1.0

    x = torch.randn(4, device=device, dtype=torch.float32)
    with stitcher.capture():
      _ = stitcher.run(block, x)
      _ = stitcher.run(block, x)

    self.assertIs(
        stitcher._segments[0].static_inputs[0],
        stitcher._segments[1].static_inputs[0],
    )


if __name__ == "__main__":
  absltest.main()
