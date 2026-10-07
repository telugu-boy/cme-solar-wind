"""
Statistical Investigation of ICME Gap Definitions:
1. Edge-to-Edge    : start_{i+1} - end_i    (Quiet/ambient solar wind duration between events)
2. Start-to-Start  : start_{i+1} - start_i  (Onset-to-onset waiting time)
3. Middle-to-Middle: mid_{i+1} - mid_i      (Center-to-center waiting time)
4. End-to-End      : end_{i+1} - end_i      (Recovery-to-recovery waiting time)

All units are in Patches of 80 minutes (1 patch = 16 timesteps * 5 min = 80 min = 1.333 h).

================================================================================
STATISTICAL HYPOTHESIS TESTING SUMMARY
================================================================================

1. Kolmogorov-Smirnov Pairwise Test Statistic (D) Matrix:
--------------------------------------------------------------------------------
                  Edge-to-Edge  Start-to-Start  Middle-to-Middle  End-to-End
Edge-to-Edge          0.000000        0.190564          0.200740    0.172988
Start-to-Start        0.190564        0.000000          0.016651    0.024977
Middle-to-Middle      0.200740        0.016651          0.000000    0.035153
End-to-End            0.172988        0.024977          0.035153    0.000000

2. Kolmogorov-Smirnov Pairwise p-value Matrix:
--------------------------------------------------------------------------------
                  Edge-to-Edge  Start-to-Start  Middle-to-Middle    End-to-End
Edge-to-Edge      1.000000e+00    1.430784e-17      1.830984e-19  1.541063e-14
Start-to-Start    1.430784e-17    1.000000e+00      9.982958e-01  8.890025e-01
Middle-to-Middle  1.830984e-19    9.982958e-01      1.000000e+00  5.165449e-01
End-to-End        1.541063e-14    8.890025e-01      5.165449e-01  1.000000e+00

3. Anderson-Darling 4-Sample Omnibus Test:
--------------------------------------------------------------------------------
Statistic A^2: 55.2123
Critical values: [0.499, 1.324, 1.916, 2.493, 3.246, 3.823, 5.121]
p-value: <= 0.001 (Highly significant rejection of H0; distributions differ)

4. Kruskal-Wallis H-Test:
--------------------------------------------------------------------------------
H-Statistic: 73.5060 (df = 3)
p-value: 7.573150e-16 (Extremely significant difference across groups)

Key Takeaways:
- Edge-to-Edge is fundamentally distinct from the other three definitions
  (p < 1e-14 against all three). It subtracts the preceding event duration,
  compressing the median by ~29.3 patches (-29%) and creating near-zero artifacts.
- Start-to-Start, Middle-to-Middle, and End-to-End are statistically indistinguishable
  from one another (all pairwise KS p-values > 0.50, S2S vs M2M p = 0.998).
- Middle-to-Middle (or Start-to-Start) is the physically consistent indicator for
  event arrival / recurrence intervals.
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
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
PATCH_SECONDS = PATCH_MINUTES * 60.0


def compute_all_gap_definitions() -> pd.DataFrame:
    """
    Computes all four gap definitions across the union of all 5 catalogs
    (C&R, Jian, Lepping, Chi, Nieves) in units of 80-min patches.
    """
    all_catalogs = ["cr", "jian", "lepping", "chi", "nieves"]
    all_intervals = get_icme_intervals_from_lists(all_catalogs)

    edge_to_edge = []
    start_to_start = []
    middle_to_middle = []
    end_to_end = []

    for i in range(len(all_intervals) - 1):
        s_curr = pd.to_datetime(all_intervals[i][0])
        e_curr = pd.to_datetime(all_intervals[i][1])
        m_curr = s_curr + (e_curr - s_curr) / 2

        s_next = pd.to_datetime(all_intervals[i + 1][0])
        e_next = pd.to_datetime(all_intervals[i + 1][1])
        m_next = s_next + (e_next - s_next) / 2

        edge = (s_next - e_curr).total_seconds() / PATCH_SECONDS
        s2s = (s_next - s_curr).total_seconds() / PATCH_SECONDS
        m2m = (m_next - m_curr).total_seconds() / PATCH_SECONDS
        e2e = (e_next - e_curr).total_seconds() / PATCH_SECONDS

        edge_to_edge.append(edge)
        start_to_start.append(s2s)
        middle_to_middle.append(m2m)
        end_to_end.append(e2e)

    return pd.DataFrame({
        "Edge-to-Edge": edge_to_edge,
        "Start-to-Start": start_to_start,
        "Middle-to-Middle": middle_to_middle,
        "End-to-End": end_to_end
    })


def run_statistical_tests(df_gaps: pd.DataFrame, out_dir: Path):
    """
    Executes KS pairwise matrix, Anderson-Darling 4-sample test,
    and Kruskal-Wallis test. Prints results and saves reports.
    """
    names = list(df_gaps.columns)

    # 1. Pairwise Kolmogorov-Smirnov test matrix
    ks_stat_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    ks_pval_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for n1 in names:
        for n2 in names:
            res = stats.ks_2samp(df_gaps[n1], df_gaps[n2])
            ks_stat_matrix.loc[n1, n2] = res.statistic
            ks_pval_matrix.loc[n1, n2] = res.pvalue

    ks_stat_matrix.to_csv(out_dir / "ks_test_statistic_matrix.csv")
    ks_pval_matrix.to_csv(out_dir / "ks_test_pvalue_matrix.csv")

    # 2. Anderson-Darling 4-sample test
    ad_res = stats.anderson_ksamp([df_gaps[n].values for n in names])

    # 3. Kruskal-Wallis test
    kw_res = stats.kruskal(*[df_gaps[n].values for n in names])

    report = (
        "================================================================================\n"
        "STATISTICAL TESTS REPORT: 4 ICME GAP DEFINITIONS\n"
        "================================================================================\n\n"
        "1. Kolmogorov-Smirnov Test Statistic (D) Matrix:\n"
        f"{ks_stat_matrix.to_string()}\n\n"
        "2. Kolmogorov-Smirnov Test p-value Matrix:\n"
        f"{ks_pval_matrix.to_string()}\n\n"
        "3. Anderson-Darling 4-Sample Test:\n"
        f"   Statistic A^2   : {ad_res.statistic:.4f}\n"
        f"   Critical Values : {ad_res.critical_values}\n"
        f"   p-value         : {ad_res.pvalue:.6e} (floored at 0.001)\n\n"
        "4. Kruskal-Wallis H-Test:\n"
        f"   H-Statistic     : {kw_res.statistic:.4f} (df = 3)\n"
        f"   p-value         : {kw_res.pvalue:.6e}\n\n"
        "Conclusion:\n"
        "   - Edge-to-Edge is fundamentally distinct from the other three metrics (p < 1e-14).\n"
        "   - Start-to-Start, Middle-to-Middle, and End-to-End are statistically identical (p > 0.5).\n"
        "================================================================================\n"
    )
    with open(out_dir / "statistical_tests_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    return ks_stat_matrix, ks_pval_matrix, ad_res, kw_res


def plot_single_gap_histogram(
    series: pd.Series,
    title: str,
    subtitle: str,
    color: str,
    output_png: Path,
    max_display_patches: float = 400.0,
    bin_width_patches: float = 10.0,
    dpi: int = 300
) -> None:
    """
    Plots a standalone publication-quality histogram for a single gap definition.
    """
    data = series.values
    count = len(data)
    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    std_val = float(np.std(data))
    q25 = float(np.quantile(data, 0.25))
    q75 = float(np.quantile(data, 0.75))
    min_val = float(np.min(data))

    bins = np.arange(0, max_display_patches + bin_width_patches + 1e-5, bin_width_patches)
    clipped = np.clip(data, 0, max_display_patches)

    fig, ax = plt.subplots(figsize=(11.0, 6.5), dpi=dpi)

    ax.hist(
        clipped,
        bins=bins,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.75,
        label=f"{title} (N = {count})"
    )

    # Vertical markers
    ax.axvline(mean_val, color="darkgreen", linestyle="--", linewidth=2.4, label=f"Mean: {mean_val:.1f} patches ({mean_val * PATCH_HOURS:.1f} h)")
    ax.axvline(median_val, color="purple", linestyle=":", linewidth=2.4, label=f"Median: {median_val:.1f} patches ({median_val * PATCH_HOURS:.1f} h)")

    # Summary box
    stats_box = (
        f"{title}\n"
        f"─────────────────────────────\n"
        f"Gaps Count (N): {count}\n"
        f"Mean:           {mean_val:.1f} patches ({mean_val * PATCH_HOURS:.1f} h)\n"
        f"Std Dev:        {std_val:.1f} patches\n"
        f"Median:         {median_val:.1f} patches ({median_val * PATCH_HOURS:.1f} h)\n"
        f"IQR (Q1–Q3):    {q25:.1f}–{q75:.1f} patches\n"
        f"Min Gap:        {min_val:.2f} patches ({min_val * PATCH_HOURS:.2f} h)\n"
        f"1 Patch =       80 min (1.33 h)\n"
        f"Bin Width:      {bin_width_patches:.0f} patches ({bin_width_patches * PATCH_HOURS:.1f} h)"
    )
    ax.text(
        0.97, 0.95,
        stats_box,
        transform=ax.transAxes,
        fontsize=9.5,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.92)
    )

    ax.set_title(
        f"{title} ICME Gaps\n{subtitle} ($\\geq$ {int(max_display_patches)} patches binned in final bin)",
        fontsize=12.5,
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
    secax = ax.secondary_xaxis("top", functions=(lambda p: p * PATCH_HOURS, lambda h: h / PATCH_HOURS))
    secax.set_xlabel("Gap Duration in Hours", fontsize=11.5, labelpad=8)
    secax.set_xticks(np.arange(0, int(max_display_patches * PATCH_HOURS) + 1, 50))

    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 0.97), fontsize=9.5, frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def plot_combined_4panel_comparison(
    df: pd.DataFrame,
    output_png: Path,
    max_display_patches: float = 400.0,
    bin_width_patches: float = 10.0,
    dpi: int = 300
) -> None:
    """
    Plots a 2x2 grid comparing all 4 definitions with shared axes and binning.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True, dpi=dpi)

    configs = [
        ("Edge-to-Edge", r"start$_{i+1}$ - end$_i$ (Ambient solar wind between events)", "#1f77b4", axes[0, 0]),
        ("Start-to-Start", r"start$_{i+1}$ - start$_i$ (Onset-to-onset waiting time)", "#ff7f0e", axes[0, 1]),
        ("Middle-to-Middle", r"mid$_{i+1}$ - mid$_i$ (Center-to-center waiting time)", "#2ca02c", axes[1, 0]),
        ("End-to-End", r"end$_{i+1}$ - end$_i$ (Recovery-to-recovery waiting time)", "#d62728", axes[1, 1]),
    ]

    bins = np.arange(0, max_display_patches + bin_width_patches + 1e-5, bin_width_patches)

    for col_name, formula, color, ax in configs:
        data = df[col_name].values
        clipped = np.clip(data, 0, max_display_patches)
        mean_v = float(np.mean(data))
        median_v = float(np.median(data))

        ax.hist(clipped, bins=bins, color=color, edgecolor="black", linewidth=0.8, alpha=0.75, label=f"{col_name}")
        ax.axvline(mean_v, color="black", linestyle="--", linewidth=1.8, label=f"Mean: {mean_v:.1f} p ({mean_v * PATCH_HOURS:.1f} h)")
        ax.axvline(median_v, color="darkmagenta", linestyle=":", linewidth=2.0, label=f"Median: {median_v:.1f} p ({median_v * PATCH_HOURS:.1f} h)")

        ax.set_title(f"{col_name}\n[{formula}]", fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlim(0, max_display_patches + bin_width_patches * 0.5)
        ax.set_ylim(0, 195)
        ax.set_xticks(np.arange(0, max_display_patches + 1, 50))
        ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="lightgray")

    axes[1, 0].set_xlabel("Gap Duration (Patches of 80 min)", fontsize=11.5)
    axes[1, 1].set_xlabel("Gap Duration (Patches of 80 min)", fontsize=11.5)
    axes[0, 0].set_ylabel("Number of Gaps", fontsize=11.5)
    axes[1, 0].set_ylabel("Number of Gaps", fontsize=11.5)

    fig.suptitle(
        "Comparison of 4 ICME Gap Definitions (All Catalogs Union; N = 1081)\n"
        r"1 Patch = 80 min (1.33 h) | Final bin includes all gaps $\geq$ 400 patches",
        fontsize=13.5,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main():
    out_dir = SCRIPT_DIR / "gap_definitions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing all 4 ICME gap definitions across unioned catalogs...")
    df_gaps = compute_all_gap_definitions()

    # Save summary table
    summary_path = out_dir / "gap_statistics_summary.csv"
    summary_df = pd.DataFrame({
        "Definition": ["Edge-to-Edge", "Start-to-Start", "Middle-to-Middle", "End-to-End"],
        "Formula": ["start_{i+1} - end_i", "start_{i+1} - start_i", "mid_{i+1} - mid_i", "end_{i+1} - end_i"],
        "Mean (Patches)": [df_gaps[c].mean() for c in df_gaps.columns],
        "Mean (Hours)": [df_gaps[c].mean() * PATCH_HOURS for c in df_gaps.columns],
        "Median (Patches)": [df_gaps[c].median() for c in df_gaps.columns],
        "Median (Hours)": [df_gaps[c].median() * PATCH_HOURS for c in df_gaps.columns],
        "Std Dev (Patches)": [df_gaps[c].std() for c in df_gaps.columns],
        "Min (Patches)": [df_gaps[c].min() for c in df_gaps.columns],
        "Min (Hours)": [df_gaps[c].min() * PATCH_HOURS for c in df_gaps.columns],
        "Q25 (Patches)": [df_gaps[c].quantile(0.25) for c in df_gaps.columns],
        "Q75 (Patches)": [df_gaps[c].quantile(0.75) for c in df_gaps.columns],
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")

    # Run statistical tests (KS matrix, Anderson-Darling, Kruskal-Wallis)
    print("Running statistical tests...")
    run_statistical_tests(df_gaps, out_dir)

    # Generate the 4 individual plots
    definitions = [
        ("Edge-to-Edge", "Ambient solar wind between consecutive events (start_{i+1} - end_i)", "#1f77b4", "edge_to_edge_gaps.png"),
        ("Start-to-Start", "Onset-to-onset waiting time (start_{i+1} - start_i)", "#ff7f0e", "start_to_start_gaps.png"),
        ("Middle-to-Middle", "Center-to-center waiting time (mid_{i+1} - mid_i)", "#2ca02c", "middle_to_middle_gaps.png"),
        ("End-to-End", "Recovery-to-recovery waiting time (end_{i+1} - end_i)", "#d62728", "end_to_end_gaps.png"),
    ]

    for title, subtitle, color, filename in definitions:
        out_png = out_dir / filename
        plot_single_gap_histogram(df_gaps[title], title, subtitle, color, out_png)
        print(f"[Saved] {out_png}")


    # Generate combined 4-panel comparison
    combined_png = out_dir / "comparison_all_definitions.png"
    plot_combined_4panel_comparison(df_gaps, combined_png)
    print(f"[Saved] {combined_png}")


if __name__ == "__main__":
    main()
