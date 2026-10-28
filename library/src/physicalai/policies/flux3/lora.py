# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA wiring of the FLUX3 DiT, matching LeRobot's PEFT recipe for this policy.

LeRobot wraps the policy with ``peft.get_peft_model``: LoRA adapters on every attention/MLP projection of
the DiT (cast to fp32 by PEFT's default ``autocast_adapter_dtype``) and the four embodiment heads in
``modules_to_save`` (trained in full, in the base dtype). Here adapters are injected in place so module
names and ``state_dict`` keys stay unchanged, and the heads are simply unfrozen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def inject_lora(
    dit: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str,
    trainable_modules: list[str],
) -> None:
    """Freeze ``dit``, inject LoRA adapters (fp32) and unfreeze ``trainable_modules`` in full.

    Args:
        dit: The FLUX3 DiT, mutated in place.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA input dropout.
        target_modules: Regex of adapted ``nn.Linear`` module names (relative to ``dit``).
        trainable_modules: Submodule names trained in full (the embodiment heads).

    Raises:
        RuntimeError: If the regex matches no module.
    """
    import peft  # noqa: PLC0415

    dit.requires_grad_(requires_grad=False)
    config = peft.LoraConfig(  # pyrefly: ignore[missing-attribute]
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    try:
        peft.inject_adapter_in_model(config, dit)  # pyrefly: ignore[missing-attribute]
    except ValueError as e:
        msg = f"LoRA target_modules {target_modules!r} matched no DiT layer"
        raise RuntimeError(msg) from e
    for name, param in dit.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
    for name in trainable_modules:
        dit.get_submodule(name).requires_grad_(requires_grad=True)
    trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    total = sum(p.numel() for p in dit.parameters())
    logger.info("flux3 LoRA: %d / %d trainable DiT parameters (%.3f%%)", trainable, total, 100 * trainable / total)
