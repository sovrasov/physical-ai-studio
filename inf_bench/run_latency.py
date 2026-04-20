"""Latency: avg +/- std over 100 timed passes (after 10 warmup)."""

import copy
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.runners import SinglePass

log = logging.getLogger(__name__)

CONFIGS = [
    {"name": "ACT",     "export_dir": "./exports/act_openvino",     "backend": "openvino"},
    {"name": "ACT",     "export_dir": "./exports/act_torch",        "backend": "torch"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_openvino", "backend": "openvino"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_torch",    "backend": "torch"},
    {"name": "Pi0.5",   "export_dir": "./exports/pi05_openvino",    "backend": "openvino"},
    {"name": "Pi0.5",   "export_dir": "./exports/pi05_torch",       "backend": "torch"},
]

DEVICES = ["CPU", "GPU"]
DEV_MAPPING = {
    "torch":    {"CPU": "cpu", "GPU": "xpu" if torch.xpu.is_available() else "cuda"},
    "openvino": {"CPU": "CPU", "GPU": "GPU"},
}
WARMUP_STEPS = 10
BENCHMARK_STEPS = 100
CSV_OUT = "latency.csv"


def get_observation():
    dm = LeRobotDataModule(repo_id="HuggingFaceVLA/libero", train_batch_size=1, episodes=[0])
    obs = next(iter(dm.train_dataloader()))
    return obs.to_numpy().to_dict(flatten=False)


def benchmark(cfg, device, obs):
    name, export_dir, backend = cfg["name"], cfg["export_dir"], cfg["backend"]
    try:
        model = InferenceModel.load(
            export_dir, backend=backend,
            device=DEV_MAPPING[backend][device], runner=SinglePass(),
        )
    except Exception as e:
        log.warning("SKIP %s [%s] [%s]: %s", name, backend, device, e)
        return None

    try:
        for _ in range(WARMUP_STEPS):
            model.select_action(copy.deepcopy(obs))
        timings = []
        for _ in range(BENCHMARK_STEPS):
            t0 = time.perf_counter()
            model.select_action(copy.deepcopy(obs))
            timings.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        log.warning("SKIP %s [%s] [%s] infer failed: %s", name, backend, device, e)
        return None

    arr = np.array(timings)
    return {
        "name": name, "backend": backend, "device": device,
        "avg_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)),
    }


def main():
    obs = get_observation()
    results = [
        r for cfg in CONFIGS for device in DEVICES
        if (r := benchmark(cfg, device, obs)) is not None
    ]
    if not results:
        log.warning("No results.")
        return

    for r in results:
        log.info("%-10s %-10s %-4s %8.2f +/- %6.2f ms",
                 r["name"], r["backend"], r["device"], r["avg_ms"], r["std_ms"])

    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info("Saved to %s", CSV_OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
