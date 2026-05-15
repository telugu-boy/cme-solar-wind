import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, RobustScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import xarray as xr
from datetime import date as Date
import random

import vis.omni_data_vis as omni_data_vis
import vis.cr_icme_vis as cr_icme_vis

from typing import Literal, Optional

def create_feature_matrix(
    event_dataframes: list,
    trace_res: Optional[Literal["5min", "1min"]] = "5min",
    use_asinh: bool = False,
):
    # Map resolution to points-per-window
    window_sizes = {}
    if trace_res:
        points_per_minute = {"5min": 1/5, "1min": 1}[trace_res]
        window_sizes = {
            "30min": int(30 * points_per_minute),
            "1h":    int(60 * points_per_minute),
            "3h":    int(180 * points_per_minute),
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
            # Apply transformation
            data = np.arcsinh(df[col]) if use_asinh else df[col]
            data = data.reset_index(drop=True)

            # 1. Global statistics
            row[f'{label}_mean'] = data.mean()
            row[f'{label}_var']  = data.var()
            row[f'{label}_min']  = data.min()
            row[f'{label}_max']  = data.max()

            if trace_res:
                trace_means = {}
                
                # 2. Rolling window trace features
                for win_name, win_points in window_sizes.items():
                    if win_points < 1:
                        continue  

                    rolled = data.rolling(window=win_points, min_periods=1).mean()
                    
                    # Store mean for gradient calculation
                    m = rolled.mean()
                    trace_means[win_name] = m

                    row[f'{label}_{win_name}_trace_mean'] = m
                    row[f'{label}_{win_name}_trace_var']  = rolled.var()
                    row[f'{label}_{win_name}_trace_min']  = rolled.min()
                    row[f'{label}_{win_name}_trace_max']  = rolled.max()

                # 3. Gradient (Slope) Features
                # Captures the transition/evolution of the plasma.
                # Significant delta pulls transition points away from the ambient cluster.
                if "30min" in trace_means and "3h" in trace_means:
                    row[f'{label}_slope_30m_3h'] = trace_means["30min"] - trace_means["3h"]
                
                if "30min" in trace_means and "1h" in trace_means:
                    row[f'{label}_slope_30m_1h'] = trace_means["30min"] - trace_means["1h"]

        features.append(row)

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)

def perform_mds(feature_df: pd.DataFrame, ndim: int):
    # nguyen/rudisser use mean 0 stddev 1 scaling so we should use  a robustscaler
    # which uses the median and iqr instead .. this will help.
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(feature_df)
    
    # Clip extreme outliers in scaled space to prevent MDS collapse (3 sigma)
    scaled_data = np.clip(scaled_data, -3, 3)
    
    print(f"Computing {ndim}D MDS manifold...")
    mds = MDS(n_components=ndim, dissimilarity='euclidean', random_state=42, n_init=4, metric=False)
    mds_coords = mds.fit_transform(scaled_data)

    # Stress calculation
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(scaled_data)
    sum_sq_dist = np.sum(np.triu(dist_matrix**2))
    normalized_stress = np.sqrt(mds.stress_ / sum_sq_dist) if sum_sq_dist != 0 else 0
    print(f"Raw stress: {mds.stress_}. Normalized Stress: {normalized_stress:.4f}")
    
    return mds_coords

def get_event_and_control_windows(omni_ds: xr.Dataset, icme_df: pd.DataFrame, 
                                 n_controls_factor=1, include_all_ambient=False):
    """
    Slices OMNI into ICME intervals and Ambient windows.
    If include_all_ambient=True, it tiles all non-ICME time into 30h chunks.
    """
    omni_df = omni_ds.to_dataframe().reset_index()
    omni_df['time'] = pd.to_datetime(omni_df['time'], utc=True)
    typical_duration = pd.Timedelta(hours=30)
    
    icme_windows = []
    icme_intervals = []
    
    # 1. Identify ICME windows
    for _, row in icme_df.iterrows():
        start, end = row['icme_plasma_field_start_ut'], row['icme_plasma_field_end_ut']
        if pd.isna(start) or pd.isna(end): continue
        mask = (omni_df['time'] >= start) & (omni_df['time'] <= end)
        window = omni_df[mask]
        if not window.empty:
            icme_windows.append(window)
            icme_intervals.append((start, end))

    # Sort intervals for gap finding
    icme_intervals.sort(key=lambda x: x[0])
    
    ambient_windows = []
    
    if include_all_ambient:
        print("Exhaustive sampling: Tiling all solar wind between ICMEs...")
        curr_time = omni_df['time'].min()
        max_time = omni_df['time'].max()
        
        for i_start, i_end in icme_intervals:
            # While we haven't hit the next ICME, chop gaps into 30h blocks
            while curr_time + typical_duration < i_start:
                window = omni_df[(omni_df['time'] >= curr_time) & (omni_df['time'] < curr_time + typical_duration)]
                if len(window) > 10: ambient_windows.append(window)
                curr_time += typical_duration
            curr_time = i_end # Jump past the ICME
            
        # Final tail after last ICME
        while curr_time + typical_duration < max_time:
            window = omni_df[(omni_df['time'] >= curr_time) & (omni_df['time'] < curr_time + typical_duration)]
            if len(window) > 10: ambient_windows.append(window)
            curr_time += typical_duration
            
    else:
        # Random sampling logic (your original)
        n_controls = int(len(icme_windows) * n_controls_factor)
        min_time, max_time = omni_df['time'].min(), omni_df['time'].max() - typical_duration
        attempts = 0
        while len(ambient_windows) < n_controls and attempts < (n_controls * 100):
            attempts += 1
            r_start = min_time + (max_time - min_time) * random.random()
            r_end = r_start + typical_duration
            if not any(r_start < i[1] and r_end > i[0] for i in icme_intervals):
                window = omni_df[(omni_df['time'] >= r_start) & (omni_df['time'] <= r_end)]
                if len(window) > 10: ambient_windows.append(window)
                
    return icme_windows, ambient_windows

def plot_mds_3d(mds_coords: np.ndarray, labels=None):
    """
    Renders the 3D latent space projection with a fixed, discrete color bar.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        # Define a discrete colormap: Blue for 0 (Ambient), Red for 1 (ICME)
        cmap = mcolors.ListedColormap(['#4477AA', '#EE6677'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        scatter = ax.scatter(
            mds_coords[:, 0], mds_coords[:, 1], mds_coords[:, 2], 
            c=labels, 
            cmap=cmap, 
            norm=norm,
            s=60, 
            alpha=0.7, 
            edgecolor='white',
            linewidth=0.5
        )

        # Create a discrete colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Ambient (0)', 'ICME (1)'])
        cbar.set_label('Solar Wind Classification', rotation=270, labelpad=15)
    else:
        ax.scatter(mds_coords[:, 0], mds_coords[:, 1], mds_coords[:, 2], s=40, alpha=0.6)
        
    ax.set_title("3D MDS: ICME vs Exhaustive Ambient Wind", weight='bold', fontsize=16)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")
    
    # Adjust view to see the spread better
    ax.view_init(elev=25, azim=30)
    
    plt.tight_layout()
    plt.show()

def plot_mds_2d(mds_coords: np.ndarray, labels=None):
    """
    Renders the 2D latent space projection.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if labels is not None:
        cmap = mcolors.ListedColormap(['#4477AA', '#EE6677'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        scatter = ax.scatter(
            mds_coords[:, 0], mds_coords[:, 1], 
            c=labels, 
            cmap=cmap, 
            norm=norm,
            s=70, 
            alpha=0.7, 
            edgecolor='white',
            linewidth=0.5
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Ambient (0)', 'ICME (1)'])
        cbar.set_label('Solar Wind Classification', rotation=270, labelpad=15)
    else:
        ax.scatter(mds_coords[:, 0], mds_coords[:, 1], s=50, alpha=0.6, color='gray')
        
    ax.set_title("2D MDS: ICME vs Exhaustive Ambient Wind (30min-1h-3h trace + gradient features)", weight='bold', fontsize=16)
    ax.set_xlabel("Latent Dim 1", fontsize=12)
    ax.set_ylabel("Latent Dim 2", fontsize=12)
    
    # Add a grid to help see the relative spacing of the interspersed points
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def main():
    start_date, end_date = Date(2000, 1, 1), Date(2000, 12, 31)
    omni_ds = omni_data_vis.get_omni_dataset(start_date, end_date, "5min")
    icme_df = cr_icme_vis.get_cr_icme_dataframe(start=start_date, end=end_date)
    
    # Toggle include_all_ambient=True for the exhaustive sampling you requested
    icme_wins, amb_wins = get_event_and_control_windows(omni_ds, icme_df, include_all_ambient=True)
    
    print(f"Extracted {len(icme_wins)} ICMEs and {len(amb_wins)} Ambient samples.")
    
    all_windows = amb_wins + icme_wins
    labels = np.array([0] * len(amb_wins) + [1] * len(icme_wins))
    
    # 5min or 1min to compute trace resolution
    feature_df = create_feature_matrix(all_windows, "5min")

    coords = perform_mds(feature_df, 2)
    plot_mds_2d(coords, labels)

if __name__ == "__main__":
    main()