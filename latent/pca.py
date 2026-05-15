import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import xarray as xr
from datetime import date as Date
import random
from typing import Literal, Optional

import vis.omni_data_vis as omni_data_vis
import vis.cr_icme_vis as cr_icme_vis


def create_feature_matrix(
    event_dataframes: list,
    trace_res: Optional[Literal["5min", "1min"]] = "5min",
    use_asinh: bool = False,
):
    window_sizes = {}
    if trace_res:
        points_per_minute = {"5min": 1/5, "1min": 1}[trace_res]
        window_sizes = {
            #"30min": int(30 * points_per_minute),
            "1h":    int(60 * points_per_minute),
            #"3h":    int(180 * points_per_minute),
        }

    features = []
    target_columns = {
        'V_total': 'flow_speed', 'Vx': 'Vx', 'Vy': 'Vy', 'Vz': 'Vz',
        'B_total': 'F', 'Bx': 'BX_GSE', 'By': 'BY_GSE', 'Bz': 'BZ_GSE',
        'Np': 'proton_density'
    }

    for df in event_dataframes:
        if df.empty:
            continue

        row = {}

        for label, col in target_columns.items():
            data = np.arcsinh(df[col]) if use_asinh else df[col]
            data = data.reset_index(drop=True)

            row[f'{label}_mean'] = data.mean()
            row[f'{label}_var']  = data.var()
            row[f'{label}_min']  = data.min()
            row[f'{label}_max']  = data.max()

            if trace_res:
                for win_name, win_points in window_sizes.items():
                    if win_points < 1:
                        continue

                    rolled = data.rolling(window=win_points, min_periods=1).mean()

                    row[f'{label}_{win_name}_trace_mean'] = rolled.mean()
                    row[f'{label}_{win_name}_trace_var']  = rolled.var()
                    row[f'{label}_{win_name}_trace_min']  = rolled.min()
                    row[f'{label}_{win_name}_trace_max']  = rolled.max()

        features.append(row)

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)


def get_icme_windows(omni_ds: xr.Dataset, icme_df: pd.DataFrame):
    """
    Returns ICME windows and the sorted list of (start, end) intervals
    so the caller can reuse them for ambient extraction without recomputing.
    """
    omni_df = omni_ds.to_dataframe().reset_index()
    omni_df['time'] = pd.to_datetime(omni_df['time'], utc=True)

    icme_windows   = []
    icme_intervals = []

    for _, row in icme_df.iterrows():
        start, end = row['icme_plasma_field_start_ut'], row['icme_plasma_field_end_ut']
        if pd.isna(start) or pd.isna(end):
            continue
        mask   = (omni_df['time'] >= start) & (omni_df['time'] <= end)
        window = omni_df[mask]
        if not window.empty:
            icme_windows.append(window)
            icme_intervals.append((start, end))

    icme_intervals.sort(key=lambda x: x[0])
    return icme_windows, icme_intervals


def get_ambient_windows(omni_ds: xr.Dataset, icme_intervals: list, chunk_hours: int = 30):
    """
    Tiles all OMNI time that does NOT overlap any ICME interval into
    fixed-length chunks of `chunk_hours` hours.

    Parameters
    ----------
    omni_ds        : xr.Dataset   — full OMNI dataset
    icme_intervals : list of (start, end) Timestamps, sorted ascending
    chunk_hours    : width of each ambient tile in hours (default 30,
                     matching the typical ICME duration)
    """
    omni_df = omni_ds.to_dataframe().reset_index()
    omni_df['time'] = pd.to_datetime(omni_df['time'], utc=True)

    chunk     = pd.Timedelta(hours=chunk_hours)
    curr_time = omni_df['time'].min()
    max_time  = omni_df['time'].max()

    ambient_windows = []

    for i_start, i_end in icme_intervals:
        # Tile the gap before this ICME
        while curr_time + chunk <= i_start:
            window = omni_df[
                (omni_df['time'] >= curr_time) &
                (omni_df['time'] <  curr_time + chunk)
            ]
            if len(window) > 10:
                ambient_windows.append(window)
            curr_time += chunk
        # Jump past the ICME
        curr_time = max(curr_time, i_end)

    # Tile any remaining tail after the last ICME
    while curr_time + chunk <= max_time:
        window = omni_df[
            (omni_df['time'] >= curr_time) &
            (omni_df['time'] <  curr_time + chunk)
        ]
        if len(window) > 10:
            ambient_windows.append(window)
        curr_time += chunk

    return ambient_windows


def _fit_pca(feature_df: pd.DataFrame, n_components: int = 2):
    """Scale, clip, fit PCA; return (pca, scores, loading_df)."""
    scaler = RobustScaler()
    X = np.clip(scaler.fit_transform(feature_df), -3, 3)

    pca    = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)

    loading_df = pd.DataFrame(
        pca.components_.T,
        index=feature_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loading_df["_abs_pc1"] = loading_df["PC1"].abs()
    loading_df = loading_df.sort_values("_abs_pc1", ascending=False).drop(columns="_abs_pc1")

    return pca, scores, loading_df


def _print_combined_loading_matrix(icme_loading, ambient_loading, title):
    # Rename columns to distinguish between the two datasets
    icme_renamed = icme_loading.rename(columns=lambda x: f"ICME_{x}")
    amb_renamed = ambient_loading.rename(columns=lambda x: f"Amb_{x}")
    
    # Join on the feature index
    combined = pd.concat([icme_renamed, amb_renamed], axis=1)
    
    # Sort by the absolute value of ICME PC1 for consistent readability
    combined["_sort"] = combined["ICME_PC1"].abs()
    combined = combined.sort_values("_sort", ascending=False).drop(columns="_sort")

    width = 100
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    print(combined.to_string(float_format="{:+.4f}".format))


LABEL_MAP = {
    "V_total_mean": "⟨|V|⟩",        "V_total_var":  "Var(|V|)",
    "V_total_min":  "|V| min",        "V_total_max":  "|V| max",
    "Vx_mean":      "⟨Vx⟩",          "Vy_mean":      "⟨Vy⟩",       "Vz_mean":  "⟨Vz⟩",
    "B_total_mean": "⟨|B|⟩",         "B_total_var":  "Var(|B|)",
    "B_total_min":  "|B| min",        "B_total_max":  "|B| max",
    "Bx_mean":      "⟨Bx⟩",          "By_mean":      "⟨By⟩",       "Bz_mean":  "⟨Bz⟩",
    "Bz_var":       "Var(Bz)",        "Bz_min":       "Bz min",     "Bz_max":   "Bz max",
    "Np_mean":      "⟨Np⟩",          "Np_var":       "Var(Np)",
    "Np_min":       "Np min",         "Np_max":       "Np max",
    "V_total_30min_trace_mean": "|V| 30m",  "V_total_1h_trace_mean": "|V| 1h",
    "V_total_3h_trace_mean":   "|V| 3h",   "V_total_30min_trace_var": "Var(|V|) 30m",
    "B_total_30min_trace_mean": "|B| 30m",  "B_total_1h_trace_mean": "|B| 1h",
    "B_total_3h_trace_mean":   "|B| 3h",
    "Bz_30min_trace_mean":     "Bz 30m",   "Bz_1h_trace_mean":      "Bz 1h",
    "Bz_3h_trace_mean":        "Bz 3h",    "Bz_30min_trace_var":    "Var(Bz) 30m",
    "Np_30min_trace_mean":     "Np 30m",   "Np_1h_trace_mean":      "Np 1h",
    "Np_3h_trace_mean":        "Np 3h",    "Np_30min_trace_var":    "Var(Np) 30m",
}


def _draw_biplot(ax, pca, scores, loading_df, title, n_arrows, color):
    """Draws a single PCA biplot onto `ax`."""
    explained = pca.explained_variance_ratio_
    feature_names = loading_df.index.tolist()
    loadings = pca.components_.T  # (n_features, 2)

    ax.scatter(
        scores[:, 0], scores[:, 1],
        color=color, alpha=0.55, s=45,
        edgecolor='white', linewidth=0.4, zorder=3
    )

    score_radius = np.percentile(np.sqrt(scores[:, 0]**2 + scores[:, 1]**2), 90)
    arrow_scale  = score_radius * 0.8

    top_idx = np.argsort(np.linalg.norm(loadings, axis=1))[::-1][:n_arrows]

    for idx in top_idx:
        fx = loadings[idx, 0] * arrow_scale
        fy = loadings[idx, 1] * arrow_scale
        flabel = LABEL_MAP.get(feature_names[idx], feature_names[idx])

        ax.annotate(
            "", xy=(fx, fy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.4, mutation_scale=13),
            zorder=5,
        )
        nudge = 1.14
        ax.text(
            fx * nudge, fy * nudge, flabel,
            fontsize=8, ha='center', va='center',
            color='#111111', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none')
        )

    ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.7, ls='--', alpha=0.5)
    ax.set_xlabel(f"PC1  ({explained[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2  ({explained[1]*100:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)


def perform_pca_comparison(
    icme_feature_df: pd.DataFrame,
    ambient_feature_df: pd.DataFrame,
    n_components: int = 2,
    n_arrows: int = 12,
):
    """
    Fits PCA independently on ICME and ambient feature matrices,
    prints both loading matrices, and plots biplots side by side.

    Parameters
    ----------
    icme_feature_df    : create_feature_matrix output for ICME windows
    ambient_feature_df : create_feature_matrix output for ambient windows
    n_components       : PCA dimensionality (2 for biplot)
    n_arrows           : top-N features to annotate per plot
    """
    icme_pca,    icme_scores,    icme_loading    = _fit_pca(icme_feature_df,    n_components)
    ambient_pca, ambient_scores, ambient_loading = _fit_pca(ambient_feature_df, n_components)

    # --- Print loading matrices ---
    title_str = (
        f"Side-by-Side Loading Matrix Comparison\n"
        f"  ICME (n={len(icme_feature_df)}) | Ambient (n={len(ambient_feature_df)})"
    )
    _print_combined_loading_matrix(icme_loading, ambient_loading, title_str)

    # --- Side-by-side biplots ---
    fig, (ax_icme, ax_amb) = plt.subplots(1, 2, figsize=(22, 10))

    _draw_biplot(
        ax_icme, icme_pca, icme_scores, icme_loading,
        title=f"ICME Events  (n={len(icme_feature_df)})",
        n_arrows=n_arrows, color='#EE6677'
    )
    _draw_biplot(
        ax_amb, ambient_pca, ambient_scores, ambient_loading,
        title=f"Ambient Solar Wind  (n={len(ambient_feature_df)})",
        n_arrows=n_arrows, color='#4477AA'
    )

    fig.suptitle(
        "PCA Biplot Comparison — ICME vs Ambient Solar Wind (Year 2000)\n"
        "Arrows = directions of maximum feature loading",
        fontsize=14, weight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()

    return {
        "icme":    {"pca": icme_pca,    "scores": icme_scores,    "loading": icme_loading},
        "ambient": {"pca": ambient_pca, "scores": ambient_scores, "loading": ambient_loading},
    }


def main():
    start_date, end_date = Date(2000, 1, 1), Date(2001, 12, 31)
    omni_ds = omni_data_vis.get_omni_dataset(start_date, end_date, "5min")
    icme_df = cr_icme_vis.get_cr_icme_dataframe(start=start_date, end=end_date)

    icme_wins, icme_intervals = get_icme_windows(omni_ds, icme_df)
    ambient_wins = get_ambient_windows(omni_ds, icme_intervals, chunk_hours=30)

    print(f"Extracted {len(icme_wins)} ICME windows.")
    print(f"Extracted {len(ambient_wins)} ambient windows.")

    icme_feature_df    = create_feature_matrix(icme_wins,    trace_res="5min", use_asinh=False)
    ambient_feature_df = create_feature_matrix(ambient_wins, trace_res="5min", use_asinh=False)

    results = perform_pca_comparison(icme_feature_df, ambient_feature_df, n_components=2, n_arrows=12)


if __name__ == "__main__":
    main()