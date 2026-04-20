"""Memory footprint: on-disk size + peak RSS in a fresh subprocess per config."""

import csv
import logging
import multiprocessing as mp
import os
from pathlib import Path

log = logging.getLogger(__name__)

CONFIGS = [
    {"name": "ACT",     "export_dir": "./exports/act_openvino",     "backend": "openvino"},
    {"name": "ACT",     "export_dir": "./exports/act_torch",        "backend": "torch"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_openvino", "backend": "openvino"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_torch",    "backend": "torch"},
    {"name": "Pi0.5",   "export_dir": "./exports/pi05_openvino",    "backend": "openvino"},
    {"name": "Pi0.5",   "export_dir": "./exports/pi05_torch",       "backend": "torch"},
]

DEVICE = "CPU"
CSV_OUT = "memory.csv"


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _child(export_dir, backend, device, q):
    import resource
    import sys
    from physicalai.inference import InferenceModel
    from physicalai.inference.runners import SinglePass

    dev_map = {"torch": {"CPU": "cpu"}, "openvino": {"CPU": "CPU"}}
    InferenceModel.load(export_dir, backend=backend,
                        device=dev_map[backend][device], runner=SinglePass())

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = maxrss / 1024 if sys.platform.startswith("linux") else maxrss / (1024 * 1024)
    q.put(peak_mb)


def measure(cfg):
    name, export_dir, backend = cfg["name"], cfg["export_dir"], cfg["backend"]
    path = Path(export_dir)
    if not path.exists():
        log.warning("SKIP %s [%s]: %s missing", name, backend, export_dir)
        return None

    disk_mb = dir_size_mb(path)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(export_dir, backend, DEVICE, q))
    p.start()
    p.join()

    if p.exitcode != 0 or q.empty():
        log.warning("SKIP %s [%s]: child exit %s", name, backend, p.exitcode)
        return None

    return {"name": name, "backend": backend,
            "disk_mb": disk_mb, "peak_rss_mb": q.get()}


def main():
    results = [r for cfg in CONFIGS if (r := measure(cfg)) is not None]
    if not results:
        log.warning("No results.")
        return

    for r in results:
        log.info("%-10s %-10s disk=%8.1f MB  peak=%8.1f MB",
                 r["name"], r["backend"], r["disk_mb"], r["peak_rss_mb"])

    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info("Saved to %s", CSV_OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
