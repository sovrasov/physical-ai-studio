# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Action (Black Forest Labs) world-action policy.

History-conditioned joint video + action denoising on the 7B FLUX 3 DiT, fine-tuned with LoRA from the
released SO-101 checkpoint. Requires the ``flux3`` extra; the video VAE uses NATTEN when installed (CUDA)
and a PyTorch neighborhood-attention fallback otherwise (e.g. XPU).

Example:
    >>> from physicalai.policies.flux3 import Flux3
    >>> policy = Flux3(camera_keys=("top", "wrist"))
"""

from .config import Flux3Config
from .model import Flux3Model
from .policy import Flux3

__all__ = ["Flux3", "Flux3Config", "Flux3Model"]
