# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Action model: history-conditioned joint video + action denoising on the FLUX 3 DiT.

Port of LeRobot's ``Flux3Policy`` (``modeling_flux3.py``) for the history packer. The trainable DiT is a
registered submodule; the frozen video VAE and Qwen3-VL text encoder live in :class:`FrozenComponents`,
outside ``state_dict()``/``parameters()``, and follow device moves through :meth:`Flux3Model._apply`.
Inputs are the normalized tensors produced by :class:`~physicalai.policies.flux3.processor.Flux3Processor`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from physicalai.policies.base import Model

from .f3 import packing, sampling
from .f3.positional import batched_prc_audio, batched_prc_vid, times_to_ids
from .f3.text_encoder import VEC_DIM, text_context
from .f3.transformer import JointSingleSeq, JointSingleSeqParams
from .f3.wiring import (
    REQUIRED_CONTENT_STREAMS,
    action_dit_params,
    build_action_dit,
    fresh_module_names,
    restrict_content_streams,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .config import Flux3Config

logger = logging.getLogger(__name__)

_CTX_CACHE_SIZE = 256
_VAL_SEED = 0


@dataclass
class FrozenComponents:
    """Video VAE and text encoder: eval-only, never saved, never trained.

    Deliberately not an ``nn.Module`` so nothing registers on the model. ``text_encoder_device`` pins the
    text encoder (captions are cached, so it may live on the CPU).
    """

    video_vae: Any
    text_encoder: nn.Module
    text_encoder_device: str | None = None

    def __post_init__(self) -> None:
        """Freeze both components."""
        for module in self._modules(include_text=True):
            module.eval()
            module.requires_grad_(requires_grad=False)
        if self.text_encoder_device is not None:
            self.text_encoder.to(self.text_encoder_device)

    def _modules(self, *, include_text: bool) -> Iterator[nn.Module]:
        vae = getattr(self.video_vae, "module", self.video_vae)
        if isinstance(vae, nn.Module):
            yield vae
        if include_text and isinstance(self.text_encoder, nn.Module):
            yield self.text_encoder

    def apply(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Forward a device/dtype move (``nn.Module._apply``) to the frozen modules."""
        for module in self._modules(include_text=self.text_encoder_device is None):
            module._apply(fn)  # noqa: SLF001


class Flux3Model(Model):
    """FLUX3 DiT plus frozen encoders; training loss and joint video/action sampling."""

    def __init__(
        self,
        config: Flux3Config,
        *,
        video_vae: Any,  # noqa: ANN401
        text_encoder: nn.Module,
        device: str | torch.device = "cpu",
    ) -> None:
        """Build the DiT (without weights when ``device="meta"``) and attach the frozen encoders.

        Args:
            config: Policy configuration.
            video_vae: Frozen video VAE exposing ``encode_task``.
            text_encoder: Frozen text encoder exposing ``forward_bucketed``.
            device: Construction device of the DiT; ``"meta"`` defers allocation to weight loading.
        """
        super().__init__()
        self.config = config
        self.modality = config.action_modality
        self.dtype_ = cast("torch.dtype", getattr(torch, config.dtype))
        base = restrict_content_streams(JointSingleSeqParams(**(config.dit_config or {})), REQUIRED_CONTENT_STREAMS)
        params = action_dit_params(
            base,
            config.action_modality,
            config.action_dim,
            attn_mode=config.attn_mode,
            conditioning_channels=packing.history_conditioning_channels(config),
        )
        self.dit: JointSingleSeq = build_action_dit(
            params,
            None,
            modality=self.modality,
            device=device,
            dtype=self.dtype_,
            head_seed=config.head_init_seed,
        )
        self.dit.gradient_checkpointing = config.gradient_checkpointing  # type: ignore[assignment]
        # build_action_dit returns an eval-mode DiT; Lightning preserves submodule modes, so restore the
        # default train mode (gradient checkpointing only runs in training mode).
        self.dit.train()
        self.frozen = FrozenComponents(video_vae, text_encoder, config.text_encoder_device)
        self._ctx_cache: dict[str, tuple[Tensor, Tensor]] = {}
        self._compiled_dit: Callable[..., dict[str, Tensor]] | None = None

    # ------------------------------------------------------------------ module plumbing
    def _apply(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        """Move the unregistered frozen components together with the DiT.

        Returns:
            This model.
        """
        super()._apply(fn, *args, **kwargs)
        self.frozen.apply(fn)
        self._ctx_cache.clear()
        self._compiled_dit = None
        return self

    def head_module_names(self) -> list[str]:
        """DiT submodules of the embodiment heads (trained in full, own learning rate).

        Returns:
            Names relative to ``self.dit``.
        """
        return fresh_module_names(self.modality)

    def _inference_dit(self) -> Callable[..., dict[str, Tensor]]:
        if not self.config.compile_model:
            return self.dit
        if self._compiled_dit is None:
            self._compiled_dit = torch.compile(self.dit.forward)
        return self._compiled_dit

    # ------------------------------------------------------------------ data contract
    @property
    def image_observation_delta_indices(self) -> list[int]:
        """Camera window offsets: history, current frame and the future clip."""
        return self.config.image_observation_delta_indices

    @property
    def state_observation_delta_indices(self) -> list[int]:
        """State window offsets."""
        return self.config.state_observation_delta_indices

    @property
    def observation_delta_indices(self) -> list[int]:
        """Window of any other observation stream (follows the state window)."""
        return self.config.state_observation_delta_indices

    @property
    def action_delta_indices(self) -> list[int]:
        """Absolute command window: ``n_obs_steps`` past commands plus the predicted chunk."""
        return self.config.action_delta_indices

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not used."""
        return

    @property
    def drop_n_first_frames(self) -> int:
        """Episode frames skipped at the start so history windows are never padded."""
        return self.config.n_obs_steps

    @property
    def drop_n_last_frames(self) -> int:
        """Episode frames skipped at the end so future windows are never padded."""
        return self.config.chunk_size

    # ------------------------------------------------------------------ helpers
    def _context(self, caption: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """Text context ``(1, L, ctx_dim)`` in the DiT dtype plus its ids, cached per caption.

        Returns:
            ``(ctx, ctx_ids)``.
        """
        hit = self._ctx_cache.get(caption)
        if hit is None or hit[0].device != device:
            if len(self._ctx_cache) > _CTX_CACHE_SIZE:
                self._ctx_cache.clear()
            ctx = text_context(self.frozen.text_encoder, caption, device, fixed_length=self.config.text_fixed_length)
            hit = (ctx.to(self.dtype_), packing.pack_text(ctx, VEC_DIM)["ctx_ids"])
            self._ctx_cache[caption] = hit
        return hit

    def _cast_inputs(self, kwargs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Streams/context/vector in the DiT dtype; timesteps stay fp32; ids stay int.

        Returns:
            Model keyword arguments.
        """
        out = {}
        for k, v in kwargs.items():
            out[k] = (
                v.to(self.dtype_)
                if v.is_floating_point() and not (k.endswith("_timesteps") or k == "timesteps_ctx")
                else v
            )
        return out

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:  # noqa: PLR0914
        """Flow-matching loss on one batch of normalized history windows.

        Args:
            batch: Output of ``Flux3Processor.training_inputs``.

        Returns:
            ``(loss, {"loss", "video_mse", "action_mse", "n_valid_windows"})``.

        Raises:
            ValueError: If the window shapes are inconsistent.
        """
        cfg = self.config
        cams = batch["cameras"]
        b, _, t = cams.shape[:3]
        if t != cfg.n_obs_steps + cfg.chunk_size:
            msg = f"training needs {cfg.n_obs_steps + cfg.chunk_size} frames per camera, got {t}"
            raise ValueError(msg)
        device = cams.device
        state = batch["state"].to(device)
        past = batch["past_actions"]
        past = past.to(device) if past is not None else None
        actions = batch["actions"].float().to(device)
        if actions.shape[1:] != (cfg.chunk_size, cfg.action_dim):
            msg = f"actions must be (B, {cfg.chunk_size}, {cfg.action_dim}), got {tuple(actions.shape)}"
            raise ValueError(msg)
        keep = batch["valid"].to(device)
        captions = [packing.select_caption(c, self.training) for c in batch["captions"]]
        if self.training and cfg.caption_dropout > 0:
            drop = (torch.rand(b) < cfg.caption_dropout).tolist()
            captions = ["" if d else c for c, d in zip(captions, drop, strict=True)]
        idx_keep = [i for i, valid in enumerate(keep.tolist()) if valid]
        if not idx_keep:
            zero = next(p for p in self.parameters() if p.requires_grad).sum() * 0.0
            return zero, {"loss": zero.detach(), "video_mse": 0.0, "action_mse": 0.0, "n_valid_windows": 0}
        ctxs = {i: self._context(captions[i], device)[0] for i in idx_keep}
        # The DiT has no text attention mask: batch only captions of equal length together.
        groups: dict[int, list[int]] = defaultdict(list)
        for i in idx_keep:
            groups[ctxs[i].shape[1]].append(i)

        total = len(idx_keep)
        loss: Tensor | float = 0.0
        mse = torch.zeros(2, device=device)
        augment = self.training and cfg.augment
        camera_hw = (int(cams.shape[-2]), int(cams.shape[-1]))
        for idxs in groups.values():
            nb = len(idxs)
            videos = torch.stack([
                packing.materialize_video(
                    cams[i],
                    packing.sample_augmentation(camera_hw=camera_hw) if augment else None,
                    device,
                    layout=cfg.camera_layout,
                    canvas_hw=cfg.canvas_hw,
                )
                for i in idxs
            ])
            video = packing.history_pack_video(cfg, self.frozen.video_vae, videos, targets=True)
            action = packing.history_pack_actions(
                cfg,
                state[idxs],
                past[idxs] if past is not None else None,
                actions[idxs],
            )
            text = packing.pack_text(torch.cat([ctxs[i] for i in idxs]), VEC_DIM)
            timesteps = packing.sample_timesteps(nb, width=cfg.train_timestep_width, shift=cfg.train_timestep_shift)
            action_timesteps = timesteps
            if cfg.separate_timesteps:
                timesteps = torch.sigmoid(torch.randn(nb) * cfg.video_logit_std + cfg.video_logit_mean)
            kwargs, targets = packing.build_forward_kwargs(
                video,
                action,
                text,
                timesteps,
                self.modality,
                action_timesteps=action_timesteps,
                conditioning_noise_max=cfg.conditioning_noise_max if self.training else 0.0,
            )
            pred = self.dit(**self._cast_inputs(kwargs))
            losses = packing.flow_loss(
                pred,
                targets,
                self.modality,
                cfg.action_loss_weight,
                cfg.video_loss_weight,
                reduction=cfg.loss_reduction,
                channel_weights=list(cfg.action_channel_weights) if cfg.action_channel_weights else None,
            )
            w = nb / total
            loss += losses["loss"] * w
            mse += torch.stack((losses["video_mse"], losses["action_mse"])).detach() * w
        loss_tensor = cast("Tensor", loss)
        return loss_tensor, {
            "loss": loss_tensor.detach(),
            "video_mse": mse[0],
            "action_mse": mse[1],
            "n_valid_windows": total,
        }

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Evaluation-mode flow loss with a fixed noise/timestep draw (comparable across validations).

        Returns:
            ``(loss, loss_dict)``.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_VAL_SEED)
                return self.compute_loss(batch)
        finally:
            self.train(was_training)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]] | Tensor:
        """Training loss in train mode, normalized action chunk otherwise.

        Returns:
            ``(loss, loss_dict)`` or ``(B, chunk_size, action_dim)``.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _sample(self, cond: dict[str, Tensor], caption: str, seed: int) -> Tensor:  # noqa: PLR0914
        """Joint video + action denoising from noise -> ``(chunk, D)`` normalized actions.

        Returns:
            Normalized action chunk of one sample.
        """
        cfg, m, mdt = self.config, self.modality, self.dtype_
        ak, ck = f"x_{m}", f"x_{m}_cond"
        device = cond["x_video_cond"].device
        n_pred = packing.history_predicted_latent_frames(cfg)
        rng = torch.Generator().manual_seed(seed)
        video_noise = torch.randn(1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng)
        action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
        x_video, x_video_ids = batched_prc_vid(video_noise, packing.history_predicted_video_times(cfg, 1))
        x_action, x_action_ids = batched_prc_audio(action_noise, times_to_ids(packing.history_action_times(cfg, 1)))
        # The solver state stays fp32; inputs are cast at the model boundary.
        flow = {"x_video": x_video.to(device), ak: x_action.to(device)}
        fixed = {
            "x_video_ids": x_video_ids.to(device),
            f"{ak}_ids": x_action_ids.to(device),
            "x_video_cond": cond["x_video_cond"].to(device, mdt),
            "x_video_cond_ids": cond["x_video_cond_ids"].to(device),
            "x_video_cond_timesteps": torch.zeros(1, cond["x_video_cond"].shape[1], device=device),
            ck: cond[ck].to(device, mdt),
            f"{ck}_ids": cond[f"{ck}_ids"].to(device),
            f"{ck}_timesteps": torch.zeros(1, cond[ck].shape[1], device=device),
            "vector": torch.zeros(1, VEC_DIM, device=device, dtype=mdt),
        }
        guidance = {
            "x_video": cfg.guidance_scale,
            ak: cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action,
        }
        ctx_c = self._context(caption, device)
        needs_uncond = any(g != 1.0 for g in guidance.values())  # noqa: RUF069 (exact 1.0 disables CFG)
        ctx_uc = self._context("", device) if needs_uncond else None
        dit = self._inference_dit()

        def predict(samples: dict[str, Tensor], t: Tensor | float) -> dict[str, Tensor]:
            t = float(t) / 1000.0 if isinstance(t, Tensor) and not t.is_floating_point() else float(t)
            timesteps = {
                "x_video_timesteps": torch.full((1, samples["x_video"].shape[1]), t, device=device),
                f"{ak}_timesteps": torch.full((1, cfg.chunk_size), t, device=device),
            }
            model_in = {k: v.to(mdt) for k, v in samples.items()}
            if ctx_uc is None:
                ctx, ctx_ids = ctx_c
                pred = dit(
                    **model_in,
                    **fixed,
                    **timesteps,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    timesteps_ctx=torch.zeros(ctx.shape[:2], device=device),
                )
                pred = {k: pred[k] for k in model_in}
            else:
                pred = sampling.cfg_two_pass(dit, model_in, fixed, timesteps, ctx_uc, ctx_c, guidance)
            return {k: v.float() for k, v in pred.items()}

        if cfg.sampler == "cosmos_unipc":
            out = sampling.cosmos_unipc_order2(flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift)
        else:
            out = sampling.euler(flow, predict, n_steps=cfg.num_inference_steps, alpha=cfg.sampler_shift)
        return out[ak][0].float() / cfg.action_scale

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Normalized history inputs -> ``(B, chunk_size, action_dim)`` normalized actions.

        Args:
            batch: Output of ``Flux3Processor.observe``.

        Returns:
            Normalized action chunk (command deltas for ``action_representation="delta"``).

        Raises:
            ValueError: If the history length does not match ``n_obs_steps``.
        """
        cfg = self.config
        cams = batch["cameras"]
        if cams.shape[2] != cfg.n_obs_steps:
            msg = f"prediction needs exactly {cfg.n_obs_steps} observations per camera, got {cams.shape[2]}"
            raise ValueError(msg)
        states, past = batch["state"], batch["past_actions"]
        device = cams.device
        chunks = []
        for i, caption in enumerate(batch["captions"]):
            canvas = packing.materialize_video(cams[i], None, device, layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw)
            cond = packing.history_pack_video(cfg, self.frozen.video_vae, canvas[None], targets=False)
            cond.update(
                packing.history_pack_actions(
                    cfg,
                    states[i : i + 1].to(device),
                    past[i : i + 1].to(device) if past is not None else None,
                    targets=False,
                ),
            )
            chunks.append(self._sample(cond, packing.select_caption(caption, training=False), cfg.inference_seed))
        return torch.stack(chunks).float()
