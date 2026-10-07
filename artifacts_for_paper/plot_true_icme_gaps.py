"""
Generates publication-quality figures of True ICME Gaps across all catalogs 
(Cane & Richardson, Jian, Lepping, Chi, Nieves) compared against C&R alone,
with duration measured in Patches of 80 minutes and Hours.
Exports headlessly to PNG without any matplotlib popups.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent popups
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.loaders import get_icme_intervals_from_lists

# 1 patch = 16 timesteps * 5 minutes = 80 minutes = 4/3 hours
PATCH_MINUTES = 80.0
PATCH_HOURS = PATCH_MINUTES / 60.0  # 1.3333... hours


def compute_inter_event_gaps_seconds(intervals: list[tuple[np.datetime64, np.datetime64]]) -> np.ndarray:
    """
    Computes time gaps in seconds between consecutive unioned ICME intervals:
    gap_i = start_{i+1} - end_i
    """
    gaps_sec = []
    for i in range(len(intervals) - 1):
        end_curr = pd.to_datetime(intervals[i][1])
        start_next = pd.to_datetime(intervals[i + 1][0])
        gap = (start_next - end_curr).total_seconds()
        if gap > 0:
            gaps_sec.append(gap)
    return np.array(gaps_sec, dtype=np.float64)


def generate_true_icme_gaps_patch_plot(output_png: Path, max_display_patches: float = 400.0, bin_width_patches: float = 10.0, dpi: int = 300) -> Path:
    """
    Generates a publication-quality figure of True ICME Gaps in 80-min patches
    with a synchronized secondary top x-axis in Hours.
    """
    all_catalogs = ["cr", "jian", "lepping", "chi", "nieves"]
    all_intervals = get_icme_intervals_from_lists(all_catalogs)
    cr_intervals = get_icme_intervals_from_lists(["cr"])

    all_sec = compute_inter_event_gaps_seconds(all_intervals)
    cr_sec = compute_inter_event_gaps_seconds(cr_intervals)

    # Convert to 80-minute patches and hours
    all_patches = all_sec / (PATCH_MINUTES * 60.0)
    all_hours = all_sec / 3600.0
    cr_patches = cr_sec / (PATCH_MINUTES * 60.0)
    cr_hours = cr_sec / 3600.0

    n_all = len(all_patches)
    mean_patches = float(np.mean(all_patches))
    median_patches = float(np.median(all_patches))
    std_patches = float(np.std(all_patches))
    q25_patches = float(np.quantile(all_patches, 0.25))
    q75_patches = float(np.quantile(all_patches, 0.75))

    mean_hours = float(np.mean(all_hours))
    median_hours = float(np.median(all_hours))

    n_cr = len(cr_patches)
    mean_cr_patches = float(np.mean(cr_patches))
    median_cr_patches = float(np.median(cr_patches))
    mean_cr_hours = float(np.mean(cr_hours))
    median_cr_hours = float(np.median(cr_hours))

    # Binning in 80-minute patches
    bins = np.arange(0, max_display_patches + bin_width_patches + 1e-5, bin_width_patches)
    all_clipped = np.clip(all_patches, 0, max_display_patches)
    cr_clipped = np.clip(cr_patches, 0, max_display_patches)

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=dpi)

    # Histogram for All Lists
    counts_all, edges, _ = ax.hist(
        all_clipped,
        bins=bins,
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.75,
        label=f"All Lists Union (CR, Jian, Lepping, Chi, Nieves; N = {n_all})"
    )

    # Step outline for C&R alone for direct comparison
    counts_cr, _ = np.histogram(cr_clipped, bins=bins)
    ax.step(
        edges[:-1],
        counts_cr,
        where="post",
        color="crimson",
        linewidth=2.2,
        linestyle="-",
        label=f"C&R List Only (N = {n_cr}, for comparison)"
    )

    # Vertical markers for Mean and Median
    ax.axvline(
        mean_patches,
        color="darkgreen",
        linestyle="--",
        linewidth=2.4,
        label=f"All Lists Mean: {mean_patches:.1f} patches ({mean_hours:.1f} h)"
    )
    ax.axvline(
        median_patches,
        color="purple",
        linestyle=":",
        linewidth=2.4,
        label=f"All Lists Median: {median_patches:.1f} patches ({median_hours:.1f} h)"
    )

    # Summary box
    stats_box = (
        f"True ICME Gaps (All Catalogs)\n"
        f"─────────────────────────────\n"
        f"Total Gaps (N):   {n_all}\n"
        f"Mean Gap:         {mean_patches:.1f} patches ({mean_hours:.1f} h)\n"
        f"Std Dev:          {std_patches:.1f} patches\n"
        f"Median Gap:       {median_patches:.1f} patches ({median_hours:.1f} h)\n"
        f"IQR (Q1–Q3):      {q25_patches:.1f}–{q75_patches:.1f} patches\n"
        f"C&R Mean Gap:     {mean_cr_patches:.1f} patches ({mean_cr_hours:.1f} h)\n"
        f"C&R Median Gap:   {median_cr_patches:.1f} patches ({median_cr_hours:.1f} h)\n"
        f"1 Patch =         80 min (1.33 h)\n"
        f"Bin Width:        {bin_width_patches:.0f} patches ({bin_width_patches * PATCH_HOURS:.1f} h)"
    )
    ax.text(
        0.97, 0.95,
        stats_box,
        transform=ax.transAxes,
        fontsize=10.0,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.92)
    )

    ax.set_title(
        f"True ICME Gaps Distribution (All Catalogs vs. C&R Only)\n"
        f"Inter-event quiet solar wind intervals ($\\geq$ {int(max_display_patches)} patches binned in final bin)",
        fontsize=13.0,
        fontweight="bold",
        pad=16
    )
    ax.set_xlabel("Gap Duration in Patches of 80 min (1 patch = 80 min = 1.33 h)", fontsize=11.5)
    ax.set_ylabel("Number of Gaps", fontsize=11.5)
    ax.set_xlim(0, max_display_patches + bin_width_patches * 0.5)
    ax.set_ylim(0, 195)
    ax.set_xticks(np.arange(0, max_display_patches + 1, 50))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Synchronized top x-axis in Hours
    def patches_to_hours(p):
        return p * PATCH_HOURS

    def hours_to_patches(h):
        return h / PATCH_HOURS

    secax = ax.secondary_xaxis("top", functions=(patches_to_hours, hours_to_patches))
    secax.set_xlabel("Gap Duration in Hours", fontsize=11.5, labelpad=8)
    secax.set_xticks(np.arange(0, int(max_display_patches * PATCH_HOURS) + 1, 50))

    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 0.97), fontsize=9.5, frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    return output_png


def generate_true_icme_gaps_hours_plot(output_png: Path, max_display_hours: float = 480.0, bin_width_hours: float = 12.0, dpi: int = 300) -> Path:
    """
    Generates a dedicated figure binned in Hours with top axis in 80-min patches.
    """
    all_catalogs = ["cr", "jian", "lepping", "chi", "nieves"]
    all_intervals = get_icme_intervals_from_lists(all_catalogs)
    cr_intervals = get_icme_intervals_from_lists(["cr"])

    all_sec = compute_inter_event_gaps_seconds(all_intervals)
    cr_sec = compute_inter_event_gaps_seconds(cr_intervals)

    all_hours = all_sec / 3600.0
    cr_hours = cr_sec / 3600.0

    bins = np.arange(0, max_display_hours + bin_width_hours + 1e-5, bin_width_hours)
    all_clipped = np.clip(all_hours, 0, max_display_hours)
    cr_clipped = np.clip(cr_hours, 0, max_display_hours)

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=dpi)

    ax.hist(
        all_clipped,
        bins=bins,
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.75,
        label=f"All Lists Union (N = {len(all_hours)})"
    )

    counts_cr, edges = np.histogram(cr_clipped, bins=bins)
    ax.step(
        edges[:-1],
        counts_cr,
        where="post",
        color="crimson",
        linewidth=2.2,
        label=f"C&R List Only (N = {len(cr_hours)})"
    )

    mean_h = float(np.mean(all_hours))
    median_h = float(np.median(all_hours))

    ax.axvline(mean_h, color="darkgreen", linestyle="--", linewidth=2.4, label=f"Mean: {mean_h:.1f} h ({mean_h / PATCH_HOURS:.1f} patches)")
    ax.axvline(median_h, color="purple", linestyle=":", linewidth=2.4, label=f"Median: {median_h:.1f} h ({median_h / PATCH_HOURS:.1f} patches)")

    ax.set_title(
        f"True ICME Gaps Distribution in Hours (All Catalogs vs. C&R Only)\n"
        f"Inter-event quiet solar wind intervals ($\\geq$ {int(max_display_hours)} h binned in final bin)",
        fontsize=13.0,
        fontweight="bold",
        pad=16
    )
    ax.set_xlabel("Gap Duration Between Consecutive ICMEs (Hours)", fontsize=11.5)
    ax.set_ylabel("Number of Gaps", fontsize=11.5)
    ax.set_xlim(0, max_display_hours + bin_width_hours * 0.5)
    ax.set_xticks(np.arange(0, max_display_hours + 1, 48))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    secax = ax.secondary_xaxis("top", functions=(lambda h: h / PATCH_HOURS, lambda p: p * PATCH_HOURS))
    secax.set_xlabel("Gap Duration in Patches of 80 min", fontsize=11.5, labelpad=8)
    secax.set_xticks(np.arange(0, int(max_display_hours / PATCH_HOURS) + 1, 36))

    ax.legend(loc="upper center", bbox_to_anchor=(0.48, 0.96), fontsize=10.0, frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    return output_png


def generate_patch_gaps_plot(output_png: Path) -> Path:
    """
    Copies the model evaluation patch gap figure.
    """
    source_true_gaps = REPO_ROOT / "results" / "full" / "patchtsmixer_backbone_all_lists" / "misc" / "icme_pred_gaps" / "true_gaps.png"
    if source_true_gaps.exists():
        import shutil
        shutil.copy2(source_true_gaps, output_png)
    return output_png


def main():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out_main = SCRIPT_DIR / "true_icme_gaps.png"
    out_hours = SCRIPT_DIR / "true_icme_gaps_hours.png"
    out_patch = SCRIPT_DIR / "true_icme_patch_gaps.png"

    print("Generating True ICME Gaps in 80-min Patches with synchronized Hours axis...")
    generate_true_icme_gaps_patch_plot(out_main)
    print(f"[Success] Saved: {out_main}")

    print("Generating True ICME Gaps in Hours...")
    generate_true_icme_gaps_hours_plot(out_hours)
    print(f"[Success] Saved: {out_hours}")

    print("Copying/Generating patch-level True ICME Gaps from model evaluation...")
    generate_patch_gaps_plot(out_patch)
    print(f"[Success] Saved: {out_patch}")


if __name__ == "__main__":
    main()
