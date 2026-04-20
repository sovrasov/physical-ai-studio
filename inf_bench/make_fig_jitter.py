"""Boxplot of per-pass latency distribution from jitter_raw.csv."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE.parent / "fig_jitter.pdf"

COLORS = {
    "openvino/CPU": "#4c72b0",
    "openvino/GPU": "#55a868",
    "torch/CPU":    "#c44e52",
    "torch/GPU":    "#dd8452",
}

PLOT_FONT = 7
plt.rcParams.update({
    "font.size": PLOT_FONT,
    "axes.titlesize": PLOT_FONT + 1,
    "axes.labelsize": PLOT_FONT,
    "xtick.labelsize": PLOT_FONT - 1,
    "ytick.labelsize": PLOT_FONT - 1,
    "pdf.fonttype": 42,
})


def load_raw(path):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["name"], f"{row['backend']}/{row['device']}")
            groups[key].append(float(row["latency_ms"]))
    return groups


def main():
    groups = load_raw(HERE / "jitter_raw.csv")

    policies = sorted({k[0] for k in groups})
    configs  = sorted({k[1] for k in groups})

    # one subplot per policy
    fig, axes = plt.subplots(1, len(policies), figsize=(3.4 * len(policies), 2.4),
                             sharey=False, squeeze=False)

    for ax, policy in zip(axes[0], policies):
        data   = [groups[(policy, c)] for c in configs if (policy, c) in groups]
        labels = [c for c in configs if (policy, c) in groups]
        colors = [COLORS.get(c, "#888888") for c in labels]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
                        whiskerprops={"linewidth": 0.8},
                        capprops={"linewidth": 0.8},
                        boxprops={"linewidth": 0.8})

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(policy)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
