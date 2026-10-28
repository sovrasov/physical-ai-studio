# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Action policy (Lightning wrapper): LoRA task adaptation and torch inference.

History handling follows LeRobot:

* **Training** loads full temporal windows from the dataset (per-stream delta indices, episode-boundary
  frames dropped) and :class:`~physicalai.policies.flux3.processor.Flux3Processor` normalizes them
  statelessly.
* **Inference** is stateful: :meth:`Flux3.select_action` must be called on *every* control tick. It
  records the observation, predicts a new chunk of command deltas when the queue is empty, and integrates
  the executed delta into an absolute command that becomes part of the next tick's history. Call
  :meth:`Flux3.reset` at every episode start.

Export (ONNX/OpenVINO/ExecuTorch) is not supported: deployment uses native PyTorch inference.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from physicalai.data import Dataset, Observation
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import DEFAULT_LORA_TARGET_MODULES as _DEFAULT_LORA_TARGETS
from .config import ENCODERS_REPO_ID, ENCODERS_REVISION, SO101_REPO_ID, SO101_REVISION, Flux3Config
from .model import Flux3Model
from .processor import Flux3Processor

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch import nn

    from physicalai.data import DataModule

logger = logging.getLogger(__name__)

_FPS_TOLERANCE = 1e-6


class Flux3(Policy):
    """FLUX 3 Action world-action policy (history conditioning), fine-tuned with LoRA.

    Every constructor argument is a :class:`~physicalai.policies.flux3.config.Flux3Config` field; see its
    docstring. Defaults reproduce the SO-101 checkpoint and LeRobot's ``lora.json`` recipe. The 7B DiT and
    the frozen encoders are loaded lazily (``setup``, checkpoint loading or the first prediction).

    Examples:
        >>> from physicalai.policies import Flux3
        >>> policy = Flux3(camera_keys=("top", "wrist"))  # dataset "top" camera feeds the "scene" slot
    """

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | None = SO101_REPO_ID,
        *,
        revision: str | None = SO101_REVISION,
        encoders_repo_id: str = ENCODERS_REPO_ID,
        encoders_revision: str = ENCODERS_REVISION,
        camera_keys: tuple[str, ...] | None = None,
        checkpoint_camera_keys: tuple[str, ...] = ("scene", "wrist"),
        action_dim: int = 6,
        normalization_stats: dict[str, dict[str, list[float]]] | None = None,
        n_obs_steps: int = 8,
        chunk_size: int = 42,
        n_action_steps: int = 32,
        fps: float = 30.0,
        video_position_fps: float = 24.0,
        history_snapshots: int = 2,
        condition_on_past_actions: bool = True,
        action_representation: Literal["delta", "absolute"] = "delta",
        delta_absolute_dims: tuple[int, ...] = (-1,),
        normalization_clip: float = 6.0,
        camera_layout: Literal["side_by_side", "single", "grid"] = "side_by_side",
        canvas_hw: tuple[int, int] = (256, 512),
        action_modality: str = "action",
        action_scale: float = 2.0,
        text_fixed_length: int | None = 320,
        dit_config: dict[str, Any] | None = None,
        head_init_seed: int = 0,
        separate_timesteps: bool = True,
        video_logit_mean: float = 1.08,
        video_logit_std: float = 1.0,
        conditioning_noise_max: float = 0.2,
        loss_reduction: Literal["modalities", "joint_tokens"] = "modalities",
        action_channel_weights: tuple[float, ...] | None = (1.0, 1.0, 1.0, 1.0, 1.0, 2.0),
        train_timestep_width: float = 0.75,
        train_timestep_shift: float = 42.0,
        action_loss_weight: float = 0.5,
        video_loss_weight: float = 1.0,
        caption_dropout: float = 0.1,
        augment: bool = False,
        gradient_checkpointing: bool = True,
        sampler: Literal["euler", "cosmos_unipc"] = "euler",
        num_inference_steps: int = 4,
        guidance_scale: float = 3.0,
        guidance_scale_action: float | None = None,
        sampler_shift: float = 6.93,
        inference_seed: int = 42,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        attn_mode: Literal["torch", "flash", "cudnn"] = "torch",
        compile_model: bool = False,
        text_encoder_device: str | None = None,
        lora_enabled: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: str = _DEFAULT_LORA_TARGETS,
        optimizer_lr: float = 1e-4,
        optimizer_lr_heads_multiplier: float = 5.0,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_freeze_backbone_steps: int = 0,
        scheduler_warmup_steps: int = 0,
        scheduler_warmup_steps_heads: int = 0,
        scheduler_decay_steps: int = 0,
        scheduler_cooldown_steps: int = 0,
    ) -> None:
        """Initialize the policy configuration; heavy components load lazily."""
        values = locals()
        self.config = Flux3Config(**{f.name: values[f.name] for f in dataclasses.fields(Flux3Config)})
        super().__init__(n_action_steps=self.config.n_action_steps)
        # Checkpoints hold only trainable weights and normalization buffers (see on_save_checkpoint).
        self.strict_loading = False
        self.save_hyperparameters()
        self.model = None
        self.processor: Flux3Processor | None = None

    # ------------------------------------------------------------------ construction
    def build_model(self, *, video_vae: Any = None, text_encoder: nn.Module | None = None) -> None:  # noqa: ANN401
        """Load the DiT, frozen encoders and checkpoint quantiles, then attach LoRA adapters.

        Args:
            video_vae: Optional pre-built frozen video VAE (tests); loaded from ``encoders_repo_id`` if None.
            text_encoder: Optional pre-built frozen text encoder (tests).
        """
        from .f3.text_encoder import load_text_encoder  # noqa: PLC0415
        from .f3.video_vae import load_video_vae  # noqa: PLC0415
        from .lora import inject_lora  # noqa: PLC0415
        from .pretrained import (  # noqa: PLC0415
            WEIGHTS_FILENAME,
            check_contract,
            load_dit_weights,
            read_policy_config,
            read_quantiles,
            resolve_encoders,
            resolve_policy_dir,
        )

        cfg = self.config
        policy_dir = None
        if cfg.pretrained_name_or_path is not None:
            policy_dir = resolve_policy_dir(cfg.pretrained_name_or_path, cfg.revision)
            check_contract(cfg, read_policy_config(policy_dir))
            stats = read_quantiles(policy_dir, cfg.action_dim)
        else:
            stats = cfg.normalization_stats or {}
        if video_vae is None or text_encoder is None:
            vae_path, text_dir = resolve_encoders(cfg.encoders_repo_id, cfg.encoders_revision)
            if video_vae is None:
                video_vae = load_video_vae(str(vae_path), compile_model=cfg.compile_model)
            if text_encoder is None:
                text_encoder = load_text_encoder(str(text_dir), compile_model=cfg.compile_model)

        model = Flux3Model(
            cfg,
            video_vae=video_vae,
            text_encoder=text_encoder,
            device="meta" if policy_dir is not None else "cpu",
        )
        if policy_dir is not None:
            load_dit_weights(model, policy_dir / WEIGHTS_FILENAME)
        if cfg.lora_enabled:
            inject_lora(
                model.dit,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                trainable_modules=model.head_module_names(),
            )
        else:
            model.dit.requires_grad_(requires_grad=True)
        self.model = model.to(self.device)
        self.processor = Flux3Processor(cfg, stats).to(self.device)

    def _require_model(self) -> tuple[Flux3Model, Flux3Processor]:
        if self.model is None or self.processor is None:
            self.build_model()
        return cast("Flux3Model", self.model), cast("Flux3Processor", self.processor)

    def setup(self, stage: str) -> None:
        """Build the model and request the history windows from the datamodule's datasets.

        Raises:
            ValueError: If the dataset frame rate differs from the checkpoint's.
        """
        del stage
        self._require_model()
        datamodule: DataModule = self.trainer.datamodule  # type: ignore[attr-defined]
        dataset = datamodule.train_dataset
        if isinstance(dataset, Dataset) and abs(float(dataset.fps) - self.config.fps) > _FPS_TOLERANCE:
            msg = f"dataset fps {dataset.fps} differs from the policy fps {self.config.fps}"
            raise ValueError(msg)
        reformat_dataset_to_match_policy(self, datamodule)

    # ------------------------------------------------------------------ training
    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Training loss in train mode, absolute action chunk in eval mode.

        Returns:
            ``(loss, loss_dict)`` or ``(B, chunk_size, action_dim)``.
        """
        if not self.training:
            return self.predict_action_chunk(batch)
        model, processor = self._require_model()
        return model.compute_loss(processor.training_inputs(batch.to(self.device)))

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Evaluation-mode flow loss with a fixed noise draw.

        Returns:
            ``(loss, loss_dict)``.
        """
        model, processor = self._require_model()
        return model.compute_val_loss(processor.training_inputs(batch.to(self.device)))

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss.
        """
        del batch_idx
        loss, loss_dict = self(batch)  # type: ignore[misc]
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        self.log("train/video_mse", loss_dict["video_mse"])
        self.log("train/action_mse", loss_dict["action_mse"])
        return loss

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict absolute commands ``(B, chunk_size, action_dim)``.

        A single-step observation is recorded into the stateful history first (so do not mix this with
        :meth:`select_action` on the same tick). A windowed observation is predicted offline.

        Returns:
            Absolute command chunk.
        """
        model, processor = self._require_model()
        inputs = processor.observe(batch.to(self.device))
        return processor.decode_chunk(model.predict_action_chunk(inputs))

    @torch.no_grad()
    def select_action(self, batch: Observation) -> torch.Tensor:
        """Record this tick's observation and return the next absolute command ``(B, action_dim)``.

        Returns:
            Absolute command for this tick.
        """
        model, processor = self._require_model()
        inputs = processor.observe(batch.to(self.device))
        if not self._action_queue:
            deltas = model.predict_action_chunk(inputs)[:, : self.config.n_action_steps]
            self._action_queue.extend(deltas.transpose(0, 1))
        return processor.decode_step(self._action_queue.popleft())

    def reset(self) -> None:
        """Clear the action queue and the observation/command history (episode start)."""
        super().reset()
        if self.processor is not None:
            self.processor.reset()

    # ------------------------------------------------------------------ optimization
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """AdamW with a trunk/adapter group and an action-head group (``x optimizer_lr_heads_multiplier``).

        Returns:
            Optimizer and per-step LR scheduler.
        """
        model, _ = self._require_model()
        cfg = self.config
        head_prefixes = tuple(f"{name}." for name in model.head_module_names())
        trunk: list[nn.Parameter] = []
        heads: list[nn.Parameter] = []
        for name, param in model.dit.named_parameters():
            if param.requires_grad:
                (heads if name.startswith(head_prefixes) else trunk).append(param)
        groups: list[dict[str, Any]] = [{"params": trunk, "lr": cfg.optimizer_lr}]
        if heads:
            groups.append({"params": heads, "lr": cfg.optimizer_lr * cfg.optimizer_lr_heads_multiplier})
        optimizer = torch.optim.AdamW(
            groups,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.optimizer_weight_decay,
        )
        total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0) if self._trainer else 0
        lambdas = [_lr_lambda(cfg, total, heads=False)] + ([_lr_lambda(cfg, total, heads=True)] if heads else [])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip the gradient norm to ``optimizer_grad_clip_norm`` unless the trainer overrides it."""
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    # ------------------------------------------------------------------ checkpoints
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep only trainable weights (LoRA adapters, action heads) and normalization buffers."""
        if "state_dict" not in checkpoint:
            return
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if k in trainable or k.startswith("processor.")
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Rebuild the pretrained model (with adapters) before Lightning restores the trained weights.

        Raises:
            RuntimeError: If the checkpoint holds weights this policy does not define.
        """
        self._require_model()
        unknown = set(checkpoint.get("state_dict", {})) - set(self.state_dict())
        if unknown:
            msg = f"flux3 checkpoint has {len(unknown)} unknown keys, e.g. {sorted(unknown)[:4]}"
            raise RuntimeError(msg)


def _lr_lambda(cfg: Flux3Config, total_steps: int, *, heads: bool) -> Callable[[int], float]:
    """Port of LeRobot's ``FrozenWarmupConstantSchedulerConfig.lr_lambda`` in optimizer steps.

    Trunk: zero through step ``freeze`` inclusive, linear warmup, then constant. Heads: warmup from step 0.
    Optional linear decay (from the end of the frozen phase) and final cooldown.

    Returns:
        LR multiplier as a function of the zero-based optimizer step.
    """
    frozen = 0 if heads else cfg.scheduler_freeze_backbone_steps
    warmup = cfg.scheduler_warmup_steps_heads if heads else cfg.scheduler_warmup_steps
    decay_steps, cooldown = cfg.scheduler_decay_steps, cfg.scheduler_cooldown_steps

    def factor(step: int) -> float:
        if heads:
            s = step
            warm = min(1.0, (s + 1) / (warmup + 1)) if warmup > 0 else 1.0
        else:
            if (frozen > 0 or warmup > 0) and step <= frozen:
                return 0.0
            s = step - frozen
            warm = min(1.0, s / warmup) if warmup > 0 else 1.0
        decay = max(0.0, 1.0 - s / decay_steps) if decay_steps > 0 else 1.0
        cool = 1.0
        if cooldown > 0 and total_steps > 0 and step >= total_steps - cooldown:
            cool = max(0.0, 1.0 - (step - (total_steps - cooldown)) / cooldown)
        return warm * decay * cool

    return factor
