"""Export ACT, SmolVLA, Pi0.5 to OpenVINO and Torch backends."""

import logging

from physicalai.data import LeRobotDataModule
from physicalai.policies.act import ACT, Pi05, SmolVLA

log = logging.getLogger(__name__)

DATAMODULE_REPO = "HuggingFaceVLA/libero"

POLICY_EXPORTS = [
    (ACT, {"chunk_size": 100, "n_obs_steps": 1}, [
        ("openvino", "./exports/act_openvino"),
        ("torch",    "./exports/act_torch"),
    ]),
    (SmolVLA, {"chunk_size": 100, "n_obs_steps": 1}, [
        ("openvino", "./exports/smolvla_openvino"),
        ("torch",    "./exports/smolvla_torch"),
    ]),
    (Pi05, {"chunk_size": 100, "n_obs_steps": 1}, [
        ("openvino", "./exports/pi05_openvino"),
        ("torch",    "./exports/pi05_torch"),
    ]),
]


def main():
    dm = LeRobotDataModule(repo_id=DATAMODULE_REPO, num_workers=0, episodes=[0])
    for policy_cls, kwargs, backends in POLICY_EXPORTS:
        name = policy_cls.__name__
        log.info("=== %s ===", name)
        policy = policy_cls(**kwargs, dataset_stats=dm.train_dataset.stats)
        for backend, out in backends:
            log.info("Exporting %s -> %s at %s", name, backend, out)
            try:
                policy.export(out, backend=backend)
            except Exception as e:
                log.warning("SKIP %s [%s]: %s", name, backend, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
