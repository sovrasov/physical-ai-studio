# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""History pre/post-processing of the FLUX3 policy (LeRobot ``processor_flux3`` history steps).

Training consumes full temporal windows loaded by the dataset (LeRobot-style):

    images  offsets 1-H .. chunk     (B, H + chunk, C, h, w) per camera
    state   offsets 1-H .. 0         (B, H, D) measured states
    action  offsets  -H .. chunk-1   (B, H + chunk, D) absolute commands

and turns them into normalized model inputs without keeping any state.

Inference is stateful, as in LeRobot: :meth:`Flux3Processor.observe` must run on *every* control tick
with the single current observation. It keeps the last ``H`` images/states and the commands emitted by
the policy, so the conditioning history matches the training windows. :meth:`Flux3Processor.decode_step`
turns each executed normalized command delta into an absolute command and records it as the next
history command. Call :meth:`Flux3Processor.reset` at every episode start.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .config import Flux3Config

_STREAMS = ("state", "action")
_QUANTILES = ("q01", "q99")
_IMAGE_NDIM = 4
_WINDOW_IMAGE_NDIM = 5
_VECTOR_NDIM = 2
_WINDOW_NDIM = 3
_RGB = 3


class Flux3Processor(nn.Module):
    """Quantile normalization, command deltas and the inference-time observation history.

    Quantiles live in buffers (``state_q01``/``state_q99``/``action_q01``/``action_q99``) so they are
    stored with policy checkpoints. ``state`` quantiles describe absolute measured states; ``action``
    quantiles describe the configured action representation (consecutive command deltas with absolute
    ``delta_absolute_dims``).
    """

    def __init__(self, config: Flux3Config, stats: dict[str, dict[str, list[float]]]) -> None:
        """Create the processor from checkpoint-owned quantiles.

        Args:
            config: Policy configuration.
            stats: ``{"state"|"action": {"q01": [...], "q99": [...]}}``.

        Raises:
            ValueError: If the quantiles are malformed.
        """
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.n_obs_steps = config.n_obs_steps
        self.chunk_size = config.chunk_size
        self.absolute_dims = sorted({d % config.action_dim for d in config.delta_absolute_dims})
        self.relative_dims = [d for d in range(config.action_dim) if d not in self.absolute_dims]
        for stream in _STREAMS:
            for q in _QUANTILES:
                values = torch.as_tensor(stats[stream][q], dtype=torch.float32)
                if values.shape != (config.action_dim,) or not torch.isfinite(values).all():
                    msg = f"{stream}.{q} must contain {config.action_dim} finite quantiles"
                    raise ValueError(msg)
                self.register_buffer(f"{stream}_{q}", values)
            if (self._quantile(stream, "q99") < self._quantile(stream, "q01")).any():
                msg = f"{stream} quantiles are inverted"
                raise ValueError(msg)
        self.reset()

    def _quantile(self, stream: str, q: str) -> torch.Tensor:
        return getattr(self, f"{stream}_{q}")

    # ------------------------------------------------------------------ normalization
    def scale(self, value: torch.Tensor, stream: str, *, inverse: bool = False) -> torch.Tensor:
        """Map ``[q01, q99]`` to ``[-1, 1]`` (clipped), or back with ``inverse=True``.

        Returns:
            The (un)normalized tensor.
        """
        lo = self._quantile(stream, "q01").to(value)
        hi = self._quantile(stream, "q99").to(value)
        span = torch.where(hi - lo > 1e-6, hi - lo, torch.ones_like(lo))  # noqa: PLR2004
        if inverse:
            return (value + 1) * span / 2 + lo
        clip = self.config.normalization_clip
        return (2 * (value - lo) / span - 1).clamp(-clip, clip)

    def _normalized_commands(self, commands: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` absolute commands -> ``(B, T - 1, D)`` normalized representation of steps 1..T-1.

        Returns:
            Normalized consecutive deltas (absolute on ``absolute_dims``) or absolute commands.
        """
        if self.config.action_representation == "delta":
            values = commands[:, 1:] - commands[:, :-1]
            values[..., self.absolute_dims] = commands[:, 1:, self.absolute_dims]
        else:
            values = commands[:, 1:]
        return self.scale(values, "action")

    def _past(self, commands: torch.Tensor) -> torch.Tensor:
        """``(B, H, D)`` absolute command history -> ``(B, H, D)`` normalized past actions (first is zero).

        Returns:
            Normalized past-action conditioning.
        """
        return torch.cat([torch.zeros_like(commands[:, :1]), self._normalized_commands(commands)], 1)

    # ------------------------------------------------------------------ batch access
    def _cameras(self, batch: Observation) -> list[torch.Tensor]:
        images = batch.images
        if not isinstance(images, dict):
            msg = "flux3 needs named cameras in Observation.images"
            raise TypeError(msg)
        cams = []
        for key in self.config.dataset_camera_keys:
            if key not in images:
                msg = f"camera {key!r} missing; available: {sorted(images)}. Set camera_keys to map cameras."
                raise KeyError(msg)
            cams.append(_channels_first(torch.as_tensor(images[key])))
        return cams

    def _vector(self, value: Any, name: str) -> torch.Tensor:  # noqa: ANN401
        if not isinstance(value, torch.Tensor):
            msg = f"flux3 needs {name} as a single tensor, got {type(value).__name__}"
            raise TypeError(msg)
        if value.shape[-1] != self.action_dim:
            msg = f"{name} must have {self.action_dim} channels, got {tuple(value.shape)}"
            raise ValueError(msg)
        return value.float()

    # ------------------------------------------------------------------ training windows
    def training_inputs(self, batch: Observation) -> dict[str, Any]:
        """Normalize a dataset window batch for the training / eval loss.

        Returns:
            ``cameras`` ``(B, n_cams, H + chunk, C, h, w)``, normalized ``state`` ``(B, H, D)``,
            ``past_actions`` ``(B, H, D)`` or ``None``, ``actions`` ``(B, chunk, D)``, ``captions`` and
            the ``valid`` window mask.

        Raises:
            ValueError: If the windows do not match the policy's delta indices.
        """
        h, k = self.n_obs_steps, self.chunk_size
        state = self._vector(batch.state, "state")
        commands = self._vector(batch.action, "action")
        if state.ndim != _WINDOW_NDIM or state.shape[1] != h:
            msg = f"training needs a state window (B, {h}, D), got {tuple(state.shape)}"
            raise ValueError(msg)
        if commands.ndim != _WINDOW_NDIM or commands.shape[1] != h + k:
            msg = f"training needs the absolute command window (B, {h + k}, D), got {tuple(commands.shape)}"
            raise ValueError(msg)
        cams = self._cameras(batch)
        frames = h + k
        for cam in cams:
            if cam.ndim != _WINDOW_IMAGE_NDIM or cam.shape[1] != frames:
                msg = f"training needs {frames} frames per camera, got {tuple(cam.shape)}"
                raise ValueError(msg)
        normalized = self._normalized_commands(commands)
        past = None
        if self.config.condition_on_past_actions:
            past = torch.cat([torch.zeros_like(commands[:, :1]), normalized[:, : h - 1]], 1)
        return {
            "cameras": _stack_cameras(cams),
            "state": self.scale(state, "state"),
            "past_actions": past,
            "actions": normalized[:, h - 1 :],
            "captions": _captions(batch.task, state.shape[0]),
            "valid": _valid_windows(batch.extra, state.shape[0], state.device),
        }

    # ------------------------------------------------------------------ inference
    def reset(self) -> None:
        """Clear the observation/command history (call at every episode start)."""
        self._frames: deque[list[torch.Tensor]] = deque(maxlen=self.n_obs_steps)
        self._states: deque[torch.Tensor] = deque(maxlen=self.n_obs_steps)
        self._commands: deque[torch.Tensor] = deque(maxlen=self.n_obs_steps)
        self._last_command: torch.Tensor | None = None
        self._anchor: torch.Tensor | None = None

    def observe(self, batch: Observation) -> dict[str, Any]:
        """Record the current observation and build the model inputs from the history.

        A single-step observation (``state`` ``(B, D)``) updates the stateful history; the first call of an
        episode replicates the observation ``n_obs_steps`` times and uses the measured state as the
        previous command, exactly like LeRobot. A windowed observation (``state`` ``(B, H, D)``) is
        processed offline without touching the history; its command history comes from
        ``extra["command_history"]`` or the first ``H`` entries of ``action``.

        Returns:
            ``cameras`` ``(B, n_cams, H, C, h, w)``, normalized ``state``, ``past_actions`` and ``captions``.

        Raises:
            ValueError: If the observation shapes are inconsistent.
        """
        h = self.n_obs_steps
        state = self._vector(batch.state, "state")
        cams = self._cameras(batch)
        if state.ndim == _VECTOR_NDIM:
            if any(cam.ndim != _IMAGE_NDIM for cam in cams):
                msg = "stateful inference needs one image per camera per tick"
                raise ValueError(msg)
            if self._last_command is None:
                self._last_command = state.clone()
            repeats = h if not self._states else 1
            for _ in range(repeats):
                self._frames.append([cam.clone() for cam in cams])
                self._states.append(state.clone())
                self._commands.append(self._last_command.clone())
            cams = [torch.stack([frames[i] for frames in self._frames], 1) for i in range(len(cams))]
            state = torch.stack(list(self._states), 1)
            commands = torch.stack(list(self._commands), 1)
            self._anchor = commands[:, -1].clone()
        elif state.ndim == _WINDOW_NDIM and state.shape[1] == h:
            commands = _offline_commands(batch, h, self.action_dim)
            cams = [cam[:, :h] for cam in cams]
            if any(cam.ndim != _WINDOW_IMAGE_NDIM or cam.shape[1] != h for cam in cams):
                msg = f"windowed inference needs at least {h} frames per camera"
                raise ValueError(msg)
            self._anchor = commands[:, -1].clone()
        else:
            msg = f"expected state (B, D) or (B, {h}, D), got {tuple(state.shape)}"
            raise ValueError(msg)
        return {
            "cameras": _stack_cameras(cams),
            "state": self.scale(state, "state"),
            "past_actions": self._past(commands) if self.config.condition_on_past_actions else None,
            "captions": _captions(batch.task, state.shape[0]),
        }

    def _absolute(self, values: torch.Tensor, anchor: torch.Tensor, *, cumulative: bool) -> torch.Tensor:
        commands = values.clone()
        if self.config.action_representation == "delta" and self.relative_dims:
            dims = self.relative_dims
            if cumulative:
                commands[..., dims] = values[..., dims].cumsum(1) + anchor[:, None, dims]
            else:
                commands[..., dims] = values[..., dims] + anchor[..., dims]
        return commands

    def decode_step(self, action: torch.Tensor) -> torch.Tensor:
        """Normalized ``(B, D)`` action of the current tick -> absolute command, recorded as history.

        Returns:
            Absolute command ``(B, D)``.

        Raises:
            RuntimeError: If :meth:`observe` was not called for this tick.
        """
        if self._anchor is None:
            msg = "decode_step needs observe() on the same tick"
            raise RuntimeError(msg)
        values = self.scale(action.float(), "action", inverse=True)
        command = self._absolute(values, self._anchor.to(values), cumulative=False)
        self._last_command = command.clone()
        return command

    def decode_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalized ``(B, T, D)`` chunk -> absolute commands, integrated from the last command.

        Returns:
            Absolute commands ``(B, T, D)``. The history is not modified.

        Raises:
            RuntimeError: If :meth:`observe` has not been called.
        """
        if self._anchor is None:
            msg = "decode_chunk needs observe() first"
            raise RuntimeError(msg)
        values = self.scale(actions.float(), "action", inverse=True)
        return self._absolute(values, self._anchor.to(values), cumulative=True)


def _channels_first(image: torch.Tensor) -> torch.Tensor:
    """Accept ``(..., H, W, 3)`` camera tensors by moving channels before height/width.

    Returns:
        The image with channels at dim -3.
    """
    if image.shape[-1] == _RGB and image.shape[-3] != _RGB:
        return image.movedim(-1, -3)
    return image


def _stack_cameras(cams: list[torch.Tensor]) -> torch.Tensor:
    """Stack ``(B, T, C, h, w)`` cameras to ``(B, n_cams, T, C, h, w)``, resizing to a common size.

    Returns:
        Stacked camera tensor.
    """
    hw = (max(c.shape[-2] for c in cams), max(c.shape[-1] for c in cams))
    resized = []
    for cam in cams:
        if tuple(cam.shape[-2:]) != hw:
            flat = cam.reshape(-1, *cam.shape[-3:]).float()
            out = F.interpolate(flat, size=hw, mode="bilinear", align_corners=False, antialias=True)
            if cam.dtype == torch.uint8:
                out = out.round_().clamp_(0, 255).to(torch.uint8)
            cam = out.reshape(*cam.shape[:-2], *hw)  # noqa: PLW2901
        resized.append(cam)
    return torch.stack(resized, 1)


def _captions(task: Any, batch_size: int) -> list[str]:  # noqa: ANN401
    if task is None:
        return [""] * batch_size
    if isinstance(task, str):
        return [task] * batch_size
    tasks = [str(t) if t is not None else "" for t in task]
    if len(tasks) == 1 and batch_size > 1:
        tasks *= batch_size
    if len(tasks) != batch_size:
        msg = f"got {len(tasks)} task strings for a batch of {batch_size}"
        raise ValueError(msg)
    return tasks


def _valid_windows(extra: dict[str, Any] | None, batch_size: int, device: torch.device) -> torch.Tensor:
    """Windows reaching past an episode boundary carry ``*_is_pad`` flags; those samples are excluded.

    Returns:
        Boolean ``(B,)`` mask of fully valid windows.
    """
    keep = torch.ones(batch_size, dtype=torch.bool, device=device)
    for key, value in (extra or {}).items():
        if (
            key.endswith("_is_pad")
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.bool
            and value.ndim >= 1
            and value.shape[0] == batch_size
        ):
            keep &= ~value.reshape(batch_size, -1).any(-1).to(device)
    return keep


def _offline_commands(batch: Observation, n_obs_steps: int, action_dim: int) -> torch.Tensor:
    commands = (batch.extra or {}).get("command_history")
    if commands is None and isinstance(batch.action, torch.Tensor) and batch.action.ndim == _WINDOW_NDIM:
        commands = batch.action[:, :n_obs_steps]
    if not isinstance(commands, torch.Tensor) or commands.shape[1:] != (n_obs_steps, action_dim):
        msg = (
            "windowed inference needs the absolute command history (B, H, D) in extra['command_history'] "
            "or as the first H steps of action"
        )
        raise ValueError(msg)
    return commands.float()
