# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the FLUX 3 Action (``flux3``) policy.

Defaults reproduce the released SO-101 checkpoint (``black-forest-labs/flux-3-action-so101``) and the
LeRobot LoRA task-adaptation recipe shipped with it (``lora.json``): history conditioning on 8
observations with two visual snapshots, past-command conditioning, command-delta actions with an absolute
gripper, 42 predicted / 32 executed actions at 30 Hz, two cameras composed side by side on a 256x512
canvas, and rank-32 LoRA with fully trained action heads.

Example (CLI):
    physicalai fit --config configs/physicalai/flux3/so101/lora.yaml

Example (API):
    >>> from physicalai.policies.flux3 import Flux3Config
    >>> config = Flux3Config(camera_keys=("top", "wrist"))
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from physicalai.config import Config

SO101_REPO_ID = "black-forest-labs/flux-3-action-so101"
SO101_REVISION = "1442796cef14eb35ede8b86da0f013bcebae5d49"
ENCODERS_REPO_ID = "black-forest-labs/flux-3-action-base"
ENCODERS_REVISION = "62878e2925e59b7a89ec14463ce89932624c490d"

# LoRA on every attention / MLP projection of the DiT, as in LeRobot's ``_get_default_peft_targets``.
DEFAULT_LORA_TARGET_MODULES = r".*\.(q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)"

# Fields that define the checkpoint's architecture and observation/action contract. They are checked
# against the pretrained ``config.json`` and must match it.
CONTRACT_FIELDS = (
    "n_obs_steps",
    "chunk_size",
    "fps",
    "video_position_fps",
    "history_snapshots",
    "condition_on_past_actions",
    "action_representation",
    "delta_absolute_dims",
    "normalization_clip",
    "camera_layout",
    "canvas_hw",
    "action_modality",
    "action_scale",
    "text_fixed_length",
    "dit_config",
)

_SAMPLERS = ("euler", "cosmos_unipc")
_ATTN_MODES = ("torch", "flash", "cudnn")


@dataclass(frozen=True)
class Flux3Config(Config):
    """Configuration of the FLUX 3 Action policy (history conditioning).

    Attributes:
        pretrained_name_or_path: Hugging Face repo id or local directory of a LeRobot-format FLUX3
            policy (``config.json``, ``model.safetensors`` and saved processors). ``None`` builds a
            randomly initialized DiT (tests and wiring checks); ``normalization_stats`` is then required.
        revision: Pinned commit SHA of ``pretrained_name_or_path`` (ignored for local directories).
        encoders_repo_id: Repository holding the frozen video VAE (``video_vae.safetensors``) and the
            Qwen3-VL text encoder (``text_encoder/``).
        encoders_revision: Pinned commit SHA of ``encoders_repo_id``.
        camera_keys: Dataset camera names (``Observation.images`` keys) in checkpoint slot order. ``None``
            uses ``checkpoint_camera_keys`` unchanged. E.g. ``("top", "wrist")`` feeds a dataset's ``top``
            camera into the SO-101 ``scene`` slot.
        checkpoint_camera_keys: Camera slot names of the checkpoint, in composition order.
        action_dim: State and action width (they share the conditioning channels).
        normalization_stats: ``{"state"|"action": {"q01": [...], "q99": [...]}}`` for randomly initialized
            models. Pretrained checkpoints always use their own saved quantiles; setting both is an error.
        n_obs_steps: Observation history length (states, past commands and image history).
        chunk_size: Actions predicted per call.
        n_action_steps: Actions executed per prediction before replanning.
        fps: Control and video frame rate of the checkpoint.
        video_position_fps: Frame rate of the video RoPE time ids.
        history_snapshots: Image snapshots taken from the history, evenly spaced, newest included.
        condition_on_past_actions: Concatenate past command deltas with the state history.
        action_representation: ``delta`` (consecutive command deltas) or ``absolute`` commands.
        delta_absolute_dims: Channels kept absolute in delta mode (the gripper).
        normalization_clip: Clip of quantile-normalized values.
        camera_layout: Canvas composition of the cameras (``side_by_side`` for SO-101).
        canvas_hw: Pixel canvas encoded by the video VAE.
        action_modality: Name of the action streams inside the DiT (part of checkpoint keys).
        action_scale: Scale of action tokens (property of the pretrained trunk).
        text_fixed_length: Padded caption length in tokens.
        dit_config: Overrides of ``JointSingleSeqParams`` (tiny values give CPU-testable models).
        head_init_seed: Seed of fresh embodiment heads (used only when a checkpoint lacks them).
        separate_timesteps: Sample video and action noise levels independently during training.
        video_logit_mean: Mean of the logit-normal video timestep distribution.
        video_logit_std: Std of the logit-normal video timestep distribution.
        conditioning_noise_max: Maximum noise level applied to conditioning video latents in training.
        loss_reduction: ``modalities`` (per-modality means) or ``joint_tokens``.
        action_channel_weights: Per-channel action loss weights.
        train_timestep_width: Logit-logistic scale of the action timestep distribution.
        train_timestep_shift: Rational shift of the action timestep distribution.
        action_loss_weight: Weight of the action flow-matching loss.
        video_loss_weight: Weight of the video flow-matching loss.
        caption_dropout: Probability of training on the empty caption.
        augment: Apply crop + color-jitter augmentation in training.
        gradient_checkpointing: Recompute DiT blocks in the backward pass.
        sampler: ``euler`` or ``cosmos_unipc``.
        num_inference_steps: Denoising steps.
        guidance_scale: Classifier-free guidance of the video stream.
        guidance_scale_action: Guidance of the action stream; ``None`` reuses ``guidance_scale``.
        sampler_shift: Timestep shift of the sampler schedule.
        inference_seed: Seed of the CPU generator drawing the initial noise of every prediction.
        dtype: DiT weight dtype (``bfloat16`` or ``float32``).
        attn_mode: DiT attention backend (``torch`` SDPA, ``flash`` or ``cudnn``).
        compile_model: ``torch.compile`` the DiT forward used by inference.
        text_encoder_device: Pin the frozen text encoder to a device (e.g. ``"cpu"`` to save accelerator
            memory; captions are cached). ``None`` follows the policy device.
        lora_enabled: Train LoRA adapters on the DiT plus the four action heads in full. ``False`` trains
            the whole DiT.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha (scaling ``alpha / rank``).
        lora_dropout: Dropout on LoRA inputs.
        lora_target_modules: Regex of DiT linear layers that get adapters.
        optimizer_lr: AdamW learning rate of the trunk / adapters.
        optimizer_lr_heads_multiplier: Learning-rate multiplier of the action heads.
        optimizer_betas: AdamW betas.
        optimizer_eps: AdamW epsilon.
        optimizer_weight_decay: AdamW weight decay.
        optimizer_grad_clip_norm: Gradient-norm clip (``0`` disables).
        scheduler_freeze_backbone_steps: Optimizer steps with a zero trunk learning rate.
        scheduler_warmup_steps: Linear trunk warmup after the frozen phase.
        scheduler_warmup_steps_heads: Linear warmup of the head learning rate.
        scheduler_decay_steps: Linear decay to zero after the frozen phase (``0`` = constant).
        scheduler_cooldown_steps: Linear decay to zero over the final steps (``0`` = off).
    """

    pretrained_name_or_path: str | None = SO101_REPO_ID
    revision: str | None = SO101_REVISION
    encoders_repo_id: str = ENCODERS_REPO_ID
    encoders_revision: str = ENCODERS_REVISION
    camera_keys: tuple[str, ...] | None = None
    checkpoint_camera_keys: tuple[str, ...] = ("scene", "wrist")
    action_dim: int = 6
    normalization_stats: dict[str, dict[str, list[float]]] | None = None

    # --- observation / action contract (checked against the checkpoint) ---
    n_obs_steps: int = 8
    chunk_size: int = 42
    n_action_steps: int = 32
    fps: float = 30.0
    video_position_fps: float = 24.0
    history_snapshots: int = 2
    condition_on_past_actions: bool = True
    action_representation: Literal["delta", "absolute"] = "delta"
    delta_absolute_dims: tuple[int, ...] = (-1,)
    normalization_clip: float = 6.0
    camera_layout: Literal["side_by_side", "single", "grid"] = "side_by_side"
    canvas_hw: tuple[int, int] = (256, 512)
    action_modality: str = "action"
    action_scale: float = 2.0
    text_fixed_length: int | None = 320
    dit_config: dict[str, Any] | None = None
    head_init_seed: int = 0

    # --- training objective ---
    separate_timesteps: bool = True
    video_logit_mean: float = 1.08
    video_logit_std: float = 1.0
    conditioning_noise_max: float = 0.2
    loss_reduction: Literal["modalities", "joint_tokens"] = "modalities"
    action_channel_weights: tuple[float, ...] | None = (1.0, 1.0, 1.0, 1.0, 1.0, 2.0)
    train_timestep_width: float = 0.75
    train_timestep_shift: float = 42.0
    action_loss_weight: float = 0.5
    video_loss_weight: float = 1.0
    caption_dropout: float = 0.1
    augment: bool = False
    gradient_checkpointing: bool = True

    # --- sampling ---
    sampler: Literal["euler", "cosmos_unipc"] = "euler"
    num_inference_steps: int = 4
    guidance_scale: float = 3.0
    guidance_scale_action: float | None = None
    sampler_shift: float = 6.93
    inference_seed: int = 42

    # --- runtime ---
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    attn_mode: Literal["torch", "flash", "cudnn"] = "torch"
    compile_model: bool = False
    text_encoder_device: str | None = None

    # --- LoRA recipe ---
    lora_enabled: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES

    # --- optimizer / scheduler ---
    optimizer_lr: float = 1e-4
    optimizer_lr_heads_multiplier: float = 5.0
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_freeze_backbone_steps: int = 0
    scheduler_warmup_steps: int = 0
    scheduler_warmup_steps_heads: int = 0
    scheduler_decay_steps: int = 0
    scheduler_cooldown_steps: int = 0

    def _set_frozen(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)  # ruff: ignore[unnecessary-dunder-call]

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Normalize sequence fields and validate the configuration.

        Raises:
            ValueError: If a field is out of range or inconsistent.
        """
        for name in ("camera_keys", "checkpoint_camera_keys", "delta_absolute_dims", "action_channel_weights"):
            value = getattr(self, name)
            if value is not None:
                self._set_frozen(name, tuple(value))
        self._set_frozen("canvas_hw", (int(self.canvas_hw[0]), int(self.canvas_hw[1])))
        self._set_frozen("optimizer_betas", (float(self.optimizer_betas[0]), float(self.optimizer_betas[1])))

        errors: list[str] = []
        if self.pretrained_name_or_path is None and self.normalization_stats is None:
            errors.append("a model without pretrained_name_or_path needs normalization_stats")
        if self.pretrained_name_or_path is not None and self.normalization_stats is not None:
            errors.append("pretrained checkpoints own their normalization; do not set normalization_stats")
        for name in ("revision", "encoders_revision"):
            value = getattr(self, name)
            if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
                errors.append(f"{name} must be a 40-character commit SHA, got {value!r}")
        if self.camera_keys is not None and len(self.camera_keys) != len(self.checkpoint_camera_keys):
            errors.append(
                f"camera_keys {self.camera_keys} must map onto the checkpoint slots {self.checkpoint_camera_keys}",
            )
        if self.camera_layout == "side_by_side" and len(self.checkpoint_camera_keys) != 2:  # noqa: PLR2004
            errors.append("side_by_side needs exactly two cameras")
        if self.action_dim < 1:
            errors.append("action_dim must be positive")
        if not 1 <= self.history_snapshots <= self.n_obs_steps:
            errors.append("history conditioning requires 1 <= history_snapshots <= n_obs_steps")
        if self.chunk_size < 17:  # noqa: PLR2004
            errors.append("independent future-video encoding needs chunk_size >= 17")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            errors.append("need 1 <= n_action_steps <= chunk_size")
        if any(not -self.action_dim <= d < self.action_dim for d in self.delta_absolute_dims):
            errors.append("delta_absolute_dims must be within the action dimension")
        if self.action_representation != "delta" and self.delta_absolute_dims:
            errors.append("delta_absolute_dims requires action_representation='delta'")
        if self.action_channel_weights is not None and (
            len(self.action_channel_weights) != self.action_dim
            or any(not math.isfinite(w) or w <= 0 for w in self.action_channel_weights)
        ):
            errors.append("action_channel_weights needs one positive finite weight per action channel")
        if self.sampler not in _SAMPLERS:
            errors.append(f"sampler must be one of {_SAMPLERS}")
        if self.attn_mode not in _ATTN_MODES:
            errors.append(f"attn_mode must be one of {_ATTN_MODES}")
        if self.dtype not in {"bfloat16", "float32"}:
            errors.append("dtype must be 'bfloat16' or 'float32'")
        if not 0.0 <= self.caption_dropout <= 1.0 or not 0.0 <= self.conditioning_noise_max <= 1.0:
            errors.append("caption_dropout and conditioning_noise_max must be in [0, 1]")
        if self.num_inference_steps < 1:
            errors.append("num_inference_steps must be >= 1")
        if self.lora_enabled and (self.lora_rank < 1 or self.lora_alpha < 1):
            errors.append("lora_rank and lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            errors.append("lora_dropout must be in [0, 1)")
        if errors:
            raise ValueError("Invalid Flux3Config: " + "; ".join(errors))

    # ---- derived (names follow the vendored packing helpers) ----
    @property
    def dataset_camera_keys(self) -> tuple[str, ...]:
        """Dataset camera names in checkpoint slot order."""
        return self.camera_keys if self.camera_keys is not None else self.checkpoint_camera_keys

    @property
    def latent_hw(self) -> tuple[int, int]:
        """Latent grid of the canvas (``ceil(canvas / 32)``)."""
        return (-(-self.canvas_hw[0] // 32), -(-self.canvas_hw[1] // 32))

    @property
    def image_observation_delta_indices(self) -> list[int]:
        """Image window: history, current frame and ``chunk_size`` future frames."""
        return list(range(1 - self.n_obs_steps, self.chunk_size + 1))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        """State window: ``n_obs_steps`` states ending at the current tick."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Command window: one extra preceding command establishes the first history delta."""
        return list(range(-self.n_obs_steps, self.chunk_size))
