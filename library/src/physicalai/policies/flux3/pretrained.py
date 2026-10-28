# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned loading of LeRobot-format FLUX3 checkpoints and the shared frozen encoders."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .config import CONTRACT_FIELDS
from .f3.wiring import fresh_head_state_dict, fresh_module_names, stream_of_key, unused_content_key

if TYPE_CHECKING:
    from torch import nn

    from .config import Flux3Config
    from .model import Flux3Model

logger = logging.getLogger(__name__)

POLICY_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]
ENCODER_ALLOW_PATTERNS = [
    "video_vae.safetensors",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "text_encoder/*.txt",
]
WEIGHTS_FILENAME = "model.safetensors"
VIDEO_VAE_FILENAME = "video_vae.safetensors"
TEXT_ENCODER_SUBDIR = "text_encoder"
_HISTORY_NORMALIZER = "flux3_observation_history_normalizer"
_ACTION_UNNORMALIZER = "flux3_action_history_unnormalizer"


def _snapshot(repo_id: str, revision: str | None, allow_patterns: list[str]) -> Path:
    if revision is None:
        msg = f"Downloading {repo_id!r} needs a pinned revision (commit SHA)"
        raise ValueError(msg)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=allow_patterns))


def resolve_policy_dir(name_or_path: str, revision: str | None) -> Path:
    """Return a local directory of the policy checkpoint, downloading it at a pinned revision.

    Returns:
        Directory with ``config.json``, ``model.safetensors`` and the saved processors.
    """
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    return _snapshot(name_or_path, revision, POLICY_ALLOW_PATTERNS)


def resolve_encoders(repo_id_or_path: str, revision: str | None) -> tuple[Path, Path]:
    """Return local paths of the video VAE weights and the text-encoder directory.

    Returns:
        ``(video_vae.safetensors, text_encoder/)``.

    Raises:
        FileNotFoundError: If the encoders are missing.
    """
    path = Path(repo_id_or_path).expanduser()
    root = path.resolve() if path.is_dir() else _snapshot(repo_id_or_path, revision, ENCODER_ALLOW_PATTERNS)
    vae, text = root / VIDEO_VAE_FILENAME, root / TEXT_ENCODER_SUBDIR
    if not vae.is_file() or not text.is_dir():
        msg = f"{repo_id_or_path!r} lacks {VIDEO_VAE_FILENAME} or {TEXT_ENCODER_SUBDIR}/"
        raise FileNotFoundError(msg)
    return vae, text


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise TypeError(msg)
    return data


def _normalize(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def read_policy_config(policy_dir: Path) -> dict[str, Any]:
    """Read and validate the checkpoint's ``config.json`` (LeRobot ``Flux3Config``).

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the checkpoint is not a history-conditioned FLUX3 policy.
        TypeError: If a required field is malformed.
    """
    cfg = _read_json(policy_dir / "config.json")
    if cfg.get("type") != "flux3":
        msg = f"{policy_dir} is not a flux3 policy (type={cfg.get('type')!r})"
        raise ValueError(msg)
    if cfg.get("conditioning") != "history" or cfg.get("packer") is not None:
        msg = "only history-conditioned flux3 checkpoints with the built-in packer are supported"
        raise ValueError(msg)
    for name in ("input_features", "output_features", "camera_keys"):
        if not isinstance(cfg.get(name), (dict, list)):
            msg = f"config.json field {name!r} is missing or malformed"
            raise TypeError(msg)
    return cfg


def checkpoint_camera_keys(cfg: dict[str, Any]) -> tuple[str, ...]:
    """Short camera names (``observation.images.`` stripped) in checkpoint slot order.

    Returns:
        Camera slot names.

    Raises:
        ValueError: If a camera key is malformed.
    """
    keys = []
    for key in cfg["camera_keys"]:
        if not isinstance(key, str) or not key.startswith("observation.images."):
            msg = f"unexpected camera key {key!r} in config.json"
            raise ValueError(msg)
        keys.append(key.removeprefix("observation.images."))
    return tuple(keys)


def checkpoint_action_dim(cfg: dict[str, Any]) -> int:
    """Action width declared by ``output_features.action``.

    Returns:
        Action dimension.

    Raises:
        ValueError: If the action feature is missing or malformed.
    """
    shape = cfg["output_features"].get("action", {}).get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        msg = "config.json lacks a 1-D action feature"
        raise ValueError(msg)
    return shape[0]


def check_contract(config: Flux3Config, cfg: dict[str, Any]) -> None:
    """Ensure architecture/contract fields of ``config`` match the checkpoint.

    Raises:
        ValueError: Listing every mismatching field.
    """
    mismatches = []
    for name in CONTRACT_FIELDS:
        ours, theirs = _normalize(getattr(config, name)), _normalize(cfg.get(name))
        if ours != theirs:
            mismatches.append(f"{name}: config={ours!r} checkpoint={theirs!r}")
    if config.action_dim != checkpoint_action_dim(cfg):
        mismatches.append(f"action_dim: config={config.action_dim} checkpoint={checkpoint_action_dim(cfg)}")
    if config.checkpoint_camera_keys != checkpoint_camera_keys(cfg):
        mismatches.append(
            f"checkpoint_camera_keys: config={config.checkpoint_camera_keys} checkpoint={checkpoint_camera_keys(cfg)}",
        )
    if mismatches:
        msg = "Flux3Config does not match the pretrained checkpoint:\n  " + "\n  ".join(mismatches)
        raise ValueError(msg)


def _step_state(policy_dir: Path, processor_file: str, registry_name: str) -> dict[str, torch.Tensor]:
    steps = _read_json(policy_dir / processor_file).get("steps")
    if not isinstance(steps, list):
        msg = f"{processor_file} has no steps"
        raise TypeError(msg)
    matches = [s for s in steps if isinstance(s, dict) and s.get("registry_name") == registry_name]
    if len(matches) != 1 or not isinstance(matches[0].get("state_file"), str):
        msg = f"{processor_file} needs exactly one {registry_name!r} step with a state file"
        raise ValueError(msg)
    name = matches[0]["state_file"]
    if Path(name).name != name or not name.endswith(".safetensors"):
        msg = f"unsafe processor state file name {name!r}"
        raise ValueError(msg)
    return load_file(str(policy_dir / name))


def read_quantiles(policy_dir: Path, action_dim: int) -> dict[str, dict[str, list[float]]]:
    """Read the checkpoint-owned state/action quantiles from the saved LeRobot processors.

    Returns:
        ``{"state"|"action": {"q01": [...], "q99": [...]}}``.

    Raises:
        ValueError: If quantiles are missing, malformed or disagree between pre- and post-processor.
    """
    pre = _step_state(policy_dir, "policy_preprocessor.json", _HISTORY_NORMALIZER)
    post = _step_state(policy_dir, "policy_postprocessor.json", _ACTION_UNNORMALIZER)
    stats: dict[str, dict[str, list[float]]] = {}
    for stream in ("state", "action"):
        stats[stream] = {}
        for q in ("q01", "q99"):
            value = pre.get(f"{stream}.{q}")
            if value is None or value.shape != (action_dim,):
                msg = f"saved processor lacks {stream}.{q} with {action_dim} values"
                raise ValueError(msg)
            stats[stream][q] = [float(v) for v in value.float()]
            if not all(math.isfinite(v) for v in stats[stream][q]):
                msg = f"{stream}.{q} has non-finite values"
                raise ValueError(msg)
    for q in ("q01", "q99"):
        key = f"action.{q}"
        if key not in post or not torch.equal(post[key].float(), pre[key].float()):
            msg = "pre/postprocessor action quantiles disagree"
            raise ValueError(msg)
    return stats


def load_dit_weights(model: Flux3Model, weights_file: Path) -> None:  # noqa: PLR0914
    """Strictly load a LeRobot FLUX3 ``model.safetensors`` (``dit.*`` keys) into ``model``.

    Content streams the policy is built without are skipped; embodiment heads missing from the file get
    the reference fresh initialization. Port of LeRobot's ``Flux3Policy._load_as_safetensor``.

    Raises:
        RuntimeError: On missing, unexpected or mis-shaped keys.
    """
    dit = model.dit
    expected = {f"dit.{k}": v for k, v in dit.state_dict().items()}
    head_prefixes = tuple(f"dit.{n}." for n in fresh_module_names(model.modality))
    built = set(dit.in_channels)
    state: dict[str, torch.Tensor] = {}
    foreign: list[str] = []
    dtype = next(dit.parameters()).dtype
    with safe_open(str(weights_file), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open handles are not dicts)
            if unused_content_key(key, built):
                foreign.append(key)
                continue
            tensor = f.get_tensor(key)
            if key in expected and tuple(expected[key].shape) != tuple(tensor.shape):
                msg = f"shape mismatch in {key}: checkpoint {tuple(tensor.shape)} vs model {tuple(expected[key].shape)}"
                raise RuntimeError(msg)
            state[key] = tensor.to(dtype)
    if foreign:
        logger.info(
            "flux3: ignoring %d tensors of content streams this policy is built without: %s",
            len(foreign),
            sorted(filter(None, {stream_of_key(k) for k in foreign})),
        )
    missing = [k for k in expected.keys() - state.keys() if not k.startswith(head_prefixes)]
    if missing:
        msg = f"flux3 checkpoint missing {len(missing)} keys, e.g. {sorted(missing)[:6]}"
        raise RuntimeError(msg)
    unexpected = state.keys() - expected.keys()
    if unexpected:
        msg = f"flux3 checkpoint has {len(unexpected)} unexpected keys, e.g. {sorted(unexpected)[:6]}"
        raise RuntimeError(msg)
    fresh_missing = expected.keys() - state.keys()
    if fresh_missing:
        emb_in = cast("nn.ModuleDict", dit.emb_in)
        emb = cast("torch.Tensor", emb_in[model.modality].weight)
        emb_cond = cast("torch.Tensor", emb_in[f"{model.modality}_cond"].weight)
        fresh = fresh_head_state_dict(
            int(emb.shape[0]),
            model.modality,
            int(emb.shape[1]),
            model.config.head_init_seed,
            int(emb_cond.shape[1]),
        )
        state.update({k: fresh[k.removeprefix("dit.")].to(dtype) for k in fresh_missing})
    dit.load_state_dict({k.removeprefix("dit."): v for k, v in state.items()}, strict=True, assign=True)
