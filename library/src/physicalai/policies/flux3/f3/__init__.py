# mypy: ignore-errors
# Copyright 2026 Black Forest Labs. All rights reserved.
# Modifications Copyright (C) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FLUX 3 Action model components, vendored from lerobot's ``flux3`` policy (black-forest-labs/flux-action).

Kept close to upstream for diffability. Library-specific changes: no lerobot imports, local-path-only
loaders (downloads are pinned in ``physicalai.policies.flux3.pretrained``), device-generic autocast in the
packers, and a PyTorch neighborhood-attention fallback in ``video_vae`` (NATTEN-free, XPU-capable, tiled for
long clips).
"""
