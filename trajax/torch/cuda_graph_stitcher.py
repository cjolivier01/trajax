from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class _Segment:
    graph: torch.cuda.CUDAGraph
    static_inputs: Tuple[torch.Tensor, ...]
    static_outputs: Tuple[torch.Tensor, ...]
    fn: Callable[..., Any]


@dataclass
class _PendingEagerBlock:
    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    outputs: Tuple[torch.Tensor, ...]


@dataclass
class _EagerBlock:
    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    arg_eager_bindings: Tuple[Optional[Tuple[int, int]], ...]
    arg_external_bindings: Tuple[Optional[int], ...]
    kwarg_eager_bindings: Dict[str, Tuple[int, int]]
    kwarg_external_bindings: Dict[str, int]
    next_segment_index: int
    next_inputs: Optional[Tuple[torch.Tensor, ...]]
    output_bindings: Tuple[Tuple[int, int], ...]


class CUDAGraphStitcher:
    """
    Stitches multiple CUDA Graph segments with optional eager "break" regions.

    Usage pattern (inside your model forward or a wrapper):
        st = CUDAGraphStitcher()

        with st.capture():
            y = st.run(main_block, x, a, b)
            with st.eager() as eager:
                y = eager(small_eager_block, y)  # not captured, replayed eagerly
            y = st.run(rest_block, y)

        out = st.replay(x, a, b)  # replays graph segments with eager blocks between

    Training wrapper:
        train_step = st.make_training_step(step_fn)
        loss = train_step(x, y)  # captures on first call, replays thereafter

    Segment wrapper:
        block = st.wrap(block_fn)
        with st.capture():
            y = block(x)  # captured as a segment
            with st.eager() as eager:
                y = eager(jax_block, y)
            y = block2(y)

    Notes:
      - Each st.run(fn, *args) becomes a *segment* (captured graph).
      - Anything inside st.eager() runs eagerly and forces a segment boundary.
      - Tensor leaves in args can be nested in lists/tuples/dicts.
      - All Tensor args to st.run must be CUDA tensors with static shapes/dtypes.
      - Eager blocks are replayed between segments and can feed the next segment.
      - replay() expects all external inputs in order of first appearance.
      - For training, capture the full step (forward+loss+backward+step) in a segment.
      - For training, prefer optimizer.zero_grad(set_to_none=False) to reuse grad buffers.
      - training_warmup controls extra warmup steps that also apply optimizer updates.
      - reuse_static_inputs reuses input buffers across segments; only safe for read-only inputs.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        warmup: int = 3,
        training_warmup: int = 1,
        reuse_static_inputs: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.device = device
        self.warmup = warmup
        self.training_warmup = training_warmup
        self.reuse_static_inputs = reuse_static_inputs
        self.stream = stream or torch.cuda.Stream(device=device)

        self._capturing: bool = False
        self._force_eager: bool = False
        self._training: bool = False
        self._capture_device: Optional[torch.device] = None

        self._segments: List[_Segment] = []
        self._pending_eager_blocks: List[_PendingEagerBlock] = []
        self._eager_blocks: List[_EagerBlock] = []
        self._static_tensor_ids: set[int] = set()
        self._internal_tensor_ids: set[int] = set()
        self._replay_input_bindings: List[List[torch.Tensor]] = []
        self._replay_input_id_map: dict[int, int] = {}
        self._external_static_input_cache: dict[int, torch.Tensor] = {}

    @contextlib.contextmanager
    def capture(self, *, training: bool = False):
        """
        Enable capture mode: st.run will create CUDA Graph segments.

        Set training=True to use training warmups suitable for full steps.
        """
        if self._capturing:
            raise RuntimeError("Nested CUDAGraphStitcher.capture() not supported.")
        self._capturing = True
        prev_training = self._training
        self._training = training
        try:
            yield self
        finally:
            self._capturing = False
            self._training = prev_training

    def make_training_step(
        self,
        step_fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Wrap a training step to capture on first call and replay thereafter.

        The step_fn should use this stitcher to define segments (run/eager).
        Only positional Tensor inputs are supported; close over other objects.
        """
        captured = False

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured
            if kwargs:
                raise TypeError("make_training_step wrapper only supports positional args.")
            if not captured:
                self.clear()
                self._seed_replay_inputs(self._flatten_tensors(args))
                with self.capture(training=True):
                    out = step_fn(*args)
                captured = True
                return out
            return self.replay(*args)  # type: ignore[arg-type]

        return wrapper

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callable so it becomes a graph segment when capturing."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                raise TypeError("CUDAGraphStitcher.wrap only supports positional args.")
            return self.run(fn, *args)

        return wrapped

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Wrap an nn.Module so its forward becomes a graph segment when capturing."""

        class _WrappedModule(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module, stitcher: "CUDAGraphStitcher") -> None:
                super().__init__()
                self.inner = inner
                self.stitcher = stitcher

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                if kwargs:
                    raise TypeError(
                        "CUDAGraphStitcher.wrap_module only supports positional args."
                    )
                return self.stitcher.run(self.inner, *args)

        return _WrappedModule(module, self)

    @contextlib.contextmanager
    def eager(self):
        """
        Run code eagerly (not captured). Also forces a segment boundary.

        Use the yielded callable to record eager work for replay.
        """
        prev = self._force_eager
        self._force_eager = True
        try:
            yield self.run_eager
        finally:
            self._force_eager = prev

    def clear(self) -> None:
        """Drop captured graphs/segments."""
        self._segments.clear()
        self._pending_eager_blocks.clear()
        self._eager_blocks.clear()
        self._static_tensor_ids.clear()
        self._internal_tensor_ids.clear()
        self._replay_input_bindings.clear()
        self._replay_input_id_map.clear()
        self._external_static_input_cache.clear()
        self._capture_device = None

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """
        In capture mode: capture fn(*args) into a CUDA graph segment (unless in eager()).
        Outside capture mode: run fn(*args) eagerly.
        """
        if not self._capturing:
            return fn(*args)
        if self._force_eager:
            return self.run_eager(fn, *args)

        targs = tuple(self._flatten_tensors(args))
        self._validate_tensors(targs)
        capture_device = self._resolve_capture_device(targs)
        self._ensure_cuda_device(capture_device)

        current_stream = torch.cuda.current_stream(device=capture_device)
        self.stream.wait_stream(current_stream)

        static_inputs: Tuple[torch.Tensor, ...]
        with torch.cuda.stream(self.stream):
            static_map: dict[int, torch.Tensor] = {}

            def to_static(arg: torch.Tensor) -> torch.Tensor:
                arg_id = id(arg)
                if arg_id in static_map:
                    return static_map[arg_id]
                if arg_id in self._static_tensor_ids:
                    static = arg
                elif self.reuse_static_inputs:
                    static = self._external_static_input_cache.get(arg_id)
                    if static is None or static.shape != arg.shape or static.dtype != arg.dtype or static.device != arg.device:
                        static = arg.clone()
                        self._external_static_input_cache[arg_id] = static
                else:
                    static = arg.clone()
                static_map[arg_id] = static
                return static

            static_args = self._map_tensors(args, to_static)
            static_inputs = tuple(to_static(arg) for arg in targs)
            for tensor in static_inputs:
                self._internal_tensor_ids.add(id(tensor))
            warmup_iters = self.training_warmup if self._training else self.warmup
            for _ in range(warmup_iters):
                _ = fn(*static_args)
            self.stream.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream):
                out = fn(*static_args)

        current_stream.wait_stream(self.stream)

        static_outputs = self._as_tensor_tuple(out)
        for tensor in static_outputs:
            self._static_tensor_ids.add(id(tensor))
            self._internal_tensor_ids.add(id(tensor))

        self._record_replay_inputs(targs, static_inputs)
        self._finalize_pending_blocks(
            next_segment_index=len(self._segments),
            next_args=targs,
            next_static_inputs=static_inputs,
        )

        self._segments.append(
            _Segment(
                graph=g,
                static_inputs=static_inputs,
                static_outputs=static_outputs,
                fn=fn,
            )
        )
        return out

    def run_eager(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Run eagerly. If capturing, record this block to replay between segments.
        """
        if self._capturing:
            with torch.cuda.stream(self.stream):
                out = fn(*args, **kwargs)
        else:
            out = fn(*args, **kwargs)

        if not self._capturing:
            return out

        outputs = self._as_tensor_tuple_or_empty(out)
        for tensor in outputs:
            self._internal_tensor_ids.add(id(tensor))

        self._register_external_inputs(args, kwargs)

        self._pending_eager_blocks.append(
            _PendingEagerBlock(fn=fn, args=args, kwargs=dict(kwargs), outputs=outputs)
        )
        return out

    def replay(self, *replay_inputs: torch.Tensor) -> Any:
        """
        Replay stitched execution. Provide all external inputs in the order
        they first appeared during capture (including inputs to later segments).
        """
        if not self._segments:
            raise RuntimeError("No captured segments to replay.")

        replay_inputs = tuple(self._flatten_tensors(replay_inputs))
        self._validate_tensors(replay_inputs, allow_empty=not self._replay_input_bindings)
        if self._capture_device is not None and any(
            inp.device != self._capture_device for inp in replay_inputs
        ):
            raise ValueError(
                f"Expected device {self._capture_device}, got {[inp.device for inp in replay_inputs]}."
            )

        if self._pending_eager_blocks:
            self._finalize_pending_blocks(
                next_segment_index=len(self._segments),
                next_args=None,
                next_static_inputs=None,
            )

        if len(replay_inputs) != len(self._replay_input_bindings):
            raise ValueError(
                "replay() expected "
                f"{len(self._replay_input_bindings)} inputs (all external inputs in capture order), "
                f"got {len(replay_inputs)}."
            )

        current_stream = torch.cuda.current_stream(device=self._capture_device)
        self.stream.wait_stream(current_stream)

        eager_outputs: List[Tuple[torch.Tensor, ...]] = []

        with torch.cuda.stream(self.stream):
            for src, targets in zip(replay_inputs, self._replay_input_bindings):
                for dst in targets:
                    dst.copy_(src)

            block_idx = 0
            for seg_idx, seg in enumerate(self._segments):
                while (
                    block_idx < len(self._eager_blocks)
                    and self._eager_blocks[block_idx].next_segment_index == seg_idx
                ):
                    eager_out = self._run_eager_block(
                        self._eager_blocks[block_idx],
                        replay_inputs=replay_inputs,
                        eager_outputs=eager_outputs,
                    )
                    eager_outputs.append(eager_out)
                    block_idx += 1
                seg.graph.replay()

            while block_idx < len(self._eager_blocks):
                eager_out = self._run_eager_block(
                    self._eager_blocks[block_idx],
                    replay_inputs=replay_inputs,
                    eager_outputs=eager_outputs,
                )
                eager_outputs.append(eager_out)
                block_idx += 1

        current_stream.wait_stream(self.stream)

        last_out = self._segments[-1].static_outputs
        return last_out[0] if len(last_out) == 1 else last_out

    def _run_eager_block(
        self,
        block: _EagerBlock,
        *,
        replay_inputs: Tuple[torch.Tensor, ...],
        eager_outputs: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, ...]:
        args = list(block.args)
        for idx, binding in enumerate(block.arg_eager_bindings):
            if binding is None:
                continue
            block_idx, out_idx = binding
            args[idx] = eager_outputs[block_idx][out_idx]
        for idx, slot in enumerate(block.arg_external_bindings):
            if slot is None:
                continue
            args[idx] = replay_inputs[slot]

        kwargs = dict(block.kwargs)
        for name, binding in block.kwarg_eager_bindings.items():
            block_idx, out_idx = binding
            kwargs[name] = eager_outputs[block_idx][out_idx]
        for name, slot in block.kwarg_external_bindings.items():
            kwargs[name] = replay_inputs[slot]

        out = block.fn(*args, **kwargs)
        outputs = self._as_tensor_tuple_or_empty(out)
        if block.output_bindings and block.next_inputs is None:
            raise RuntimeError("Eager block has output bindings but no next inputs.")
        for out_idx, in_idx in block.output_bindings:
            block.next_inputs[in_idx].copy_(outputs[out_idx])
        return outputs

    def _finalize_pending_blocks(
        self,
        *,
        next_segment_index: int,
        next_args: Optional[Tuple[torch.Tensor, ...]],
        next_static_inputs: Optional[Tuple[torch.Tensor, ...]],
    ) -> None:
        if not self._pending_eager_blocks:
            return

        output_bindings: List[List[Tuple[int, int]]] = [
            [] for _ in self._pending_eager_blocks
        ]
        output_lookup: dict[int, Tuple[int, int]] = {}
        for block_idx, block in enumerate(self._pending_eager_blocks):
            for out_idx, out in enumerate(block.outputs):
                output_lookup[id(out)] = (block_idx, out_idx)

        if next_args is not None:
            for input_idx, arg in enumerate(next_args):
                match = output_lookup.get(id(arg))
                if match is not None:
                    block_idx, out_idx = match
                    output_bindings[block_idx].append((out_idx, input_idx))

        for block_idx, block in enumerate(self._pending_eager_blocks):
            arg_eager_bindings: List[Optional[Tuple[int, int]]] = [
                None for _ in block.args
            ]
            arg_external_bindings: List[Optional[int]] = [
                None for _ in block.args
            ]
            for arg_idx, arg in enumerate(block.args):
                if not isinstance(arg, torch.Tensor):
                    continue
                arg_id = id(arg)
                eager_match = output_lookup.get(arg_id)
                if eager_match is not None:
                    arg_eager_bindings[arg_idx] = eager_match
                    continue
                external_slot = self._replay_input_id_map.get(arg_id)
                if external_slot is not None:
                    arg_external_bindings[arg_idx] = external_slot

            kwarg_eager_bindings: Dict[str, Tuple[int, int]] = {}
            kwarg_external_bindings: Dict[str, int] = {}
            for name, value in block.kwargs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                value_id = id(value)
                eager_match = output_lookup.get(value_id)
                if eager_match is not None:
                    kwarg_eager_bindings[name] = eager_match
                    continue
                external_slot = self._replay_input_id_map.get(value_id)
                if external_slot is not None:
                    kwarg_external_bindings[name] = external_slot

            self._eager_blocks.append(
                _EagerBlock(
                    fn=block.fn,
                    args=block.args,
                    kwargs=block.kwargs,
                    arg_eager_bindings=tuple(arg_eager_bindings),
                    arg_external_bindings=tuple(arg_external_bindings),
                    kwarg_eager_bindings=kwarg_eager_bindings,
                    kwarg_external_bindings=kwarg_external_bindings,
                    next_segment_index=next_segment_index,
                    next_inputs=next_static_inputs,
                    output_bindings=tuple(output_bindings[block_idx]),
                )
            )
        self._pending_eager_blocks.clear()

    def _record_replay_inputs(
        self,
        targs: Tuple[torch.Tensor, ...],
        static_inputs: Tuple[torch.Tensor, ...],
    ) -> None:
        for arg, static_input in zip(targs, static_inputs):
            arg_id = id(arg)
            if arg_id in self._internal_tensor_ids:
                continue
            slot = self._replay_input_id_map.get(arg_id)
            if slot is None:
                slot = len(self._replay_input_bindings)
                self._replay_input_id_map[arg_id] = slot
                self._replay_input_bindings.append([])
            if all(id(existing) != id(static_input) for existing in self._replay_input_bindings[slot]):
                self._replay_input_bindings[slot].append(static_input)

    def _register_external_inputs(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> None:
        for arg in args:
            for tensor in self._flatten_tensors(arg):
                self._register_external_input(tensor)
        for value in kwargs.values():
            for tensor in self._flatten_tensors(value):
                self._register_external_input(tensor)

    def _register_external_input(self, tensor: torch.Tensor) -> None:
        tensor_id = id(tensor)
        if tensor_id in self._internal_tensor_ids:
            return
        if tensor_id in self._replay_input_id_map:
            return
        slot = len(self._replay_input_bindings)
        self._replay_input_id_map[tensor_id] = slot
        self._replay_input_bindings.append([])

    def _seed_replay_inputs(self, tensors: List[torch.Tensor]) -> None:
        self._replay_input_bindings = [[] for _ in tensors]
        self._replay_input_id_map = {id(tensor): idx for idx, tensor in enumerate(tensors)}

    def _validate_tensors(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        allow_empty: bool = False,
    ) -> None:
        if not tensors:
            if allow_empty:
                return
            raise ValueError("CUDAGraphStitcher.run requires at least one Tensor input.")
        if not all(isinstance(a, torch.Tensor) for a in tensors):
            raise TypeError(
                "CUDAGraphStitcher.run: all tensor inputs must be torch.Tensor."
            )
        if not all(a.is_cuda for a in tensors):
            raise ValueError(
                "CUDAGraphStitcher.run: all tensor inputs must be CUDA tensors."
            )

    def _flatten_tensors(self, obj: Any) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []

        def visit(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
                return
            if isinstance(item, dict):
                for key in item:
                    visit(item[key])
                return
            if isinstance(item, (list, tuple)):
                for value in item:
                    visit(value)

        visit(obj)
        return tensors

    def _map_tensors(self, obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        if isinstance(obj, torch.Tensor):
            return fn(obj)
        if isinstance(obj, dict):
            return {key: self._map_tensors(value, fn) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._map_tensors(value, fn) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._map_tensors(value, fn) for value in obj)
        return obj

    def _resolve_capture_device(self, targs: Tuple[torch.Tensor, ...]) -> torch.device:
        devices = {a.device for a in targs}
        if len(devices) != 1:
            raise ValueError(f"Expected 1 device, got {devices}.")
        device = devices.pop()
        if self.device is not None:
            if device.type != self.device.type:
                raise ValueError(
                    f"Expected device {self.device}, got {[a.device for a in targs]}."
                )
            if self.device.index is not None and device != self.device:
                raise ValueError(
                    f"Expected device {self.device}, got {[a.device for a in targs]}."
                )
        if self._capture_device is None:
            self._capture_device = device
        elif device != self._capture_device:
            raise ValueError(
                f"Expected device {self._capture_device}, got {device}."
            )
        return device

    @staticmethod
    def _ensure_cuda_device(device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDAGraphStitcher only supports CUDA devices.")

    @staticmethod
    def _as_tensor_tuple(x: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(x, torch.Tensor):
            return (x,)
        if isinstance(x, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in x):
            return tuple(x)
        raise TypeError("Expected fn to return a Tensor or a tuple/list of Tensors.")

    @staticmethod
    def _as_tensor_tuple_or_empty(x: Any) -> Tuple[torch.Tensor, ...]:
        if x is None:
            return ()
        return CUDAGraphStitcher._as_tensor_tuple(x)
