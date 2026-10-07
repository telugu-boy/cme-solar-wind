"""
Generates a publication-quality histogram of ICME durations (average/distribution of ICME times)
from the Cane & Richardson (C&R) catalog headlessly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent popups
import matplotlib.pyplot as plt


def load_cr_icme_durations(csv_path: Path) -> pd.Series:
    """
    Load the Cane & Richardson ICME catalog and compute event durations in hours.
    """
    df = pd.read_csv(csv_path)

    time_cols = ['disturbance_datetime_ut', 'icme_plasma_field_start_ut', 'icme_plasma_field_end_ut']
    for col in time_cols:
        # Strip footnotes (e.g., '(A)', '(B)') and parse datetime
        df[col] = pd.to_datetime(
            df[col].astype(str).str.replace(r'\(.*\)', '', regex=True).str.strip(),
            format="%Y/%m/%d %H%M",
            utc=True,
            errors='coerce'
        )

    # Calculate duration between plasma field start and end in hours
    durations = (df['icme_plasma_field_end_ut'] - df['icme_plasma_field_start_ut']).dt.total_seconds() / 3600.0
    durations = durations.dropna()
    return durations


def plot_cr_icme_histogram(
    durations: pd.Series,
    output_path: Path,
    bin_width: float = 5.0,
    dpi: int = 300
) -> Path:
    """
    Plot and save the histogram of ICME durations with Mean and Median markers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_val = float(durations.mean())
    median_val = float(durations.median())
    std_val = float(durations.std())
    q25 = float(durations.quantile(0.25))
    q75 = float(durations.quantile(0.75))
    count = len(durations)

    # Bin appropriately: 5-hour intervals spanning from 0 to max duration
    max_val = np.ceil(durations.max())
    bins = np.arange(0, max_val + bin_width + 1e-5, bin_width)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    counts, edges, patches = ax.hist(
        durations,
        bins=bins,
        color='skyblue',
        edgecolor='black',
        linewidth=1.0,
        alpha=0.75,
        label=f'ICME Events (N = {count})'
    )

    # Markers for Mean (Average ICME Time) and Median
    ax.axvline(
        mean_val,
        color='forestgreen',
        linestyle='--',
        linewidth=2.2,
        label=f'Mean (Average): {mean_val:.2f} h'
    )
    ax.axvline(
        median_val,
        color='crimson',
        linestyle=':',
        linewidth=2.2,
        label=f'Median: {median_val:.2f} h'
    )

    # Summary statistics box
    stats_text = (
        f"C&R ICME Summary\n"
        f"────────────────\n"
        f"Events (N):   {count}\n"
        f"Mean:         {mean_val:.2f} h\n"
        f"Std Dev:      {std_val:.2f} h\n"
        f"Median:       {median_val:.2f} h\n"
        f"IQR (Q1–Q3):  {q25:.1f}–{q75:.1f} h\n"
        f"Range:        {durations.min():.1f}–{durations.max():.1f} h\n"
        f"Bin Width:    {bin_width:.0f} h"
    )
    ax.text(
        0.97, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10.5,
        family='monospace',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    ax.set_title("Cane & Richardson ICME Duration Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("ICME Duration (Hours)", fontsize=12)
    ax.set_ylabel("Number of Events", fontsize=12)

    # Format ticks cleanly
    ax.set_xlim(left=0, right=max(edges[-1], 95))
    ax.set_xticks(np.arange(0, max(edges[-1], 95) + 1, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(0.52, 0.96), fontsize=11, frameon=True, facecolor='white', edgecolor='lightgray')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return output_path


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    csv_path = repo_root / "data" / "cr_icme_catalogue.csv"
    output_png = script_dir / "cr_icme_duration_histogram.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Catalogue not found at: {csv_path}")

    print(f"Loading C&R ICME catalogue from: {csv_path}")
    durations = load_cr_icme_durations(csv_path)

    print(f"--- C&R ICME Duration Statistics ---")
    print(f"Count:    {len(durations)}")
    print(f"Min:      {durations.min():.2f} h")
    print(f"Median:   {durations.median():.2f} h")
    print(f"Mean:     {durations.mean():.2f} h")
    print(f"Max:      {durations.max():.2f} h")
    print(f"Std:      {durations.std():.2f} h")

    out_file = plot_cr_icme_histogram(durations, output_png, bin_width=5.0, dpi=300)
    print(f"\n[Success] Headless plot generated and saved to: {out_file}")


if __name__ == "__main__":
    main()
