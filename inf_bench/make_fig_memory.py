"""Grouped bar chart of disk size and peak RSS from memory.csv."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent / "fig_memory.pdf"

PLOT_FONT = 7
plt.rcParams.update({
    "font.size": PLOT_FONT,
    "axes.titlesize": PLOT_FONT + 1,
    "axes.labelsize": PLOT_FONT,
    "xtick.labelsize": PLOT_FONT - 1,
    "ytick.labelsize": PLOT_FONT - 1,
    "pdf.fonttype": 42,
})


def main():
    rows = []
    with open(HERE / "memory.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    labels   = [f"{r['name']}\n{r['backend']}" for r in rows]
    disk_mb  = [float(r["disk_mb"])     for r in rows]
    peak_mb  = [float(r["peak_rss_mb"]) for r in rows]

    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(x - width / 2, disk_mb, width, label="Disk",     color="#4c72b0")
    ax.bar(x + width / 2, peak_mb, width, label="Peak RSS", color="#c44e52")

    ax.set_yscale("log")
    ax.set_ylabel("MB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", frameon=False, ncol=2,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
