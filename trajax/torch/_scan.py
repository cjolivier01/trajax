"""Internal scan wrapper.

Uses `torch._higher_order_ops.scan` when available; otherwise falls back to a
Python loop (still GPU-safe, but less compile/graph friendly).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch


def scan(
    fn: Callable[[Any, Any], Tuple[Any, Any]],
    init: Any,
    xs: Any,
    reverse: bool = False,
):
  if hasattr(torch, "_higher_order_ops") and hasattr(torch._higher_order_ops,
                                                   "scan"):
    try:
      return torch._higher_order_ops.scan(fn, init, xs, reverse=reverse)
    except Exception:
      # Fall back to eager loop if `scan` can't be compiled (e.g. unsupported
      # ops like `lstsq` in the scan body).
      pass

  # Fallback path: xs must be either a Tensor or a tuple of Tensors with
  # consistent leading dimension.
  if isinstance(xs, torch.Tensor):
    length = xs.shape[0]
    get_i = lambda i: xs[i]
  else:
    length = xs[0].shape[0]

    def get_i(i):
      return tuple(x[i] for x in xs)

  carry = init
  outs = []
  if reverse:
    idxs = range(length - 1, -1, -1)
  else:
    idxs = range(length)

  for i in idxs:
    carry, out = fn(carry, get_i(i))
    outs.append(out)
  if reverse:
    outs.reverse()

  def stack_out(o_list):
    if isinstance(o_list[0], torch.Tensor):
      return torch.stack(o_list, dim=0)
    return tuple(stack_out([o[i] for o in o_list]) for i in range(len(o_list[0])))

  return carry, stack_out(outs)
