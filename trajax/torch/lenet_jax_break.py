from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajax.torch.cuda_graph_stitcher import CUDAGraphStitcher


def _jax_tanh(x: torch.Tensor) -> torch.Tensor:
    try:
        import jax.dlpack as jdlpack
        import jax.numpy as jnp
    except Exception as exc:
        raise RuntimeError("JAX is required for the JAX break block.") from exc

    x_jax = jdlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    y_jax = jnp.tanh(x_jax)
    return torch.utils.dlpack.from_dlpack(jdlpack.to_dlpack(y_jax))


class LeNetJaxBreak(nn.Module):
    """LeNet-style model with a JAX-only block between two CUDA Graph segments."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        stitcher: Optional[CUDAGraphStitcher] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.stitcher = stitcher or CUDAGraphStitcher()
        self._captured_mode: Optional[str] = None

    def _segment1(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def _segment2(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _jax_block(self, x: torch.Tensor) -> torch.Tensor:
        return _jax_tanh(x)

    def capture(self, x: torch.Tensor) -> None:
        if not x.is_cuda:
            raise ValueError("LeNetJaxBreak.capture expects a CUDA input tensor.")
        self.stitcher.clear()
        with self.stitcher.capture():
            y = self.stitcher.run(self._segment1, x)
            with self.stitcher.eager() as eager:
                y = eager(self._jax_block, y)
            _ = self.stitcher.run(self._segment2, y)
        self._captured_mode = "forward"

    def capture_training_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not x.is_cuda or not target.is_cuda:
            raise ValueError("LeNetJaxBreak.capture_training_step expects CUDA tensors.")
        self.stitcher.clear()

        def segment2(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=False)
            logits = self._segment2(y)
            loss = loss_fn(logits, t)
            loss.backward()
            optimizer.step()
            return loss

        with self.stitcher.capture(training=True):
            y = self.stitcher.run(self._segment1, x)
            with self.stitcher.eager() as eager:
                with torch.no_grad():
                    y = eager(self._jax_block, y)
            _ = self.stitcher.run(segment2, y, target)
        self._captured_mode = "training"

    def training_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        if self._captured_mode == "training":
            return self.stitcher.replay(x, target)
        optimizer.zero_grad(set_to_none=False)
        y = self._segment1(x)
        with torch.no_grad():
            y = self._jax_block(y)
        logits = self._segment2(y)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._captured_mode == "forward":
            return self.stitcher.replay(x)
        y = self._segment1(x)
        y = self._jax_block(y)
        return self._segment2(y)
