import xarray as xr
import cdflib.xarray as cdf_xr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from datetime import date as Date
from typing import Literal

import cr_icme_vis
import omni_data_vis

# CLAUDE
def helper_centred_diff(values: np.ndarray, dt_seconds: float, kernel_radius: int = 1) -> np.ndarray:
    """
    Compute dB/dt via a centred-difference kernel of adjustable half-width.

    kernel_radius=1  →  [-1, 0, 1] / (2 * dt)      standard centred difference
    kernel_radius=2  →  convolve with [-1, 0, 0, 0, 1] / (4 * dt)   wider smoothing
    kernel_radius=k  →  [-1, 0, …, 0, 1] / (2k * dt)

    Edges are set to NaN so they don't create artificial spikes.
    """
    k = kernel_radius
    kernel = np.zeros(2 * k + 1)
    kernel[0]  = -1.0
    kernel[-1] =  1.0
    denom = 2.0 * k * dt_seconds

    result = np.full_like(values, np.nan, dtype=float)
    # Valid interior range
    result[k:-k] = np.convolve(values, kernel, mode="valid") / denom
    return result

# CLAUDE
def helper_dt_seconds(omni_ds: xr.Dataset) -> float:
    """Infer the cadence in seconds from the time coordinate."""
    times = omni_ds.time.values
    if len(times) < 2:
        return 60.0  # fallback
    delta = pd.Timestamp(times[1]) - pd.Timestamp(times[0])
    return delta.total_seconds()


def helper_overlay_icmes(axes: list, icme_df: pd.DataFrame, ds_start, ds_end) -> None:
    """Stamp ICME boundaries (shock line + shaded body) onto every axis."""
    for i, row in icme_df.iterrows():
        t_start = row["icme_plasma_field_start_ut"]
        if pd.isna(t_start) or t_start < ds_start or t_start > ds_end:
            continue

        t_dist = row["disturbance_datetime_ut"]
        t_end  = row["icme_plasma_field_end_ut"]

        for ax in axes:
            ax.axvspan(
                t_start, t_end,
                color="limegreen", alpha=0.15,
                label="ICME Body" if (i == 0 and ax is axes[0]) else "",
            )
            if not pd.isna(t_dist):
                ax.axvline(
                    t_dist, color="indigo", linestyle="dotted", alpha=0.6, linewidth=0.75,
                    label="Shock" if (i == 0 and ax is axes[0]) else "",
                )

            # Event ID at the top of every panel
            ylim = ax.get_ylim()
            ax.text(
                t_start, ylim[1], str(i),
                rotation=45, fontsize=5.5, fontweight="bold", alpha=0.7,
                va="top", ha="left", clip_on=True,
            )

def plot_icmes_B(omni_ds: xr.Dataset, icme_df: pd.DataFrame):
    """
    Plots the Total Magnetic Field (F) from OMNI and overlays 
    the ICME boundaries from the catalog.
    """
    plt.figure(figsize=(15, 6))

    # 1. Plot the OMNI Total B-field
    plt.plot(omni_ds.time, omni_ds['F'], color='black', linewidth=1, label='Total |B| (OMNI)')

    # 2. Iterate through the catalog to draw boundaries
    # We only plot if the event overlaps with the OMNI dataset's time range
    # this needs to be as a pd.Timestamp or numpy datetime compatible
    ds_start = pd.Timestamp(omni_ds.time.values[0]).tz_localize("UTC")
    ds_end   = pd.Timestamp(omni_ds.time.values[-1]).tz_localize("UTC")
    helper_overlay_icmes([plt.gca()], icme_df, ds_start, ds_end)

    # 3. Formatting
    plt.title(f"OMNI Magnetic Field with ICME Catalog Overlays ({ds_start.strftime('%Y-%m-%d')} to {ds_end.strftime('%Y-%m-%d')})")
    plt.ylabel("B Total (nT)")
    plt.xlabel("Time (UTC)")
    
    # Adaptive X-axis (Daily/Monthly)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_icmes_B_density_velocity(omni_ds: xr.Dataset, icme_df: pd.DataFrame):
    """
    Plots a 3-panel stack (B, Density, Velocity) with ICME catalog overlays.
    """
    # Create subplots sharing the same time axis
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Plot the OMNI Data Variables
    ax[0].plot(omni_ds.time, omni_ds['F'], color='black', linewidth=1,
               label=r'$\langle |B| \rangle$ (Scalar Avg)')
    b_vec_mag = np.sqrt(omni_ds['BX_GSE']**2 + omni_ds['BY_GSE']**2 + omni_ds['BZ_GSE']**2)
    ax[0].plot(omni_ds.time, b_vec_mag, color='magenta', linewidth=1,
               linestyle='--', label=r'$|\langle B \rangle|$ (Vector Avg)', alpha=0.5)
    # plot dst
    ax[1].plot(omni_ds.time, omni_ds['SYM_H'], color = "orange", linewidth=1, label="Dst (nT)")

    ax[2].plot(omni_ds.time, omni_ds['proton_density'], color='darkgreen', linewidth=1, label='N$_p$ (cm$^{-3}$)')
    ax[3].plot(omni_ds.time, omni_ds['flow_speed'], color='blue', linewidth=1, label='V$_{sw}$ (km/s)')

    # Set labels
    ax[0].set_ylabel('Magnetic Field (nT)')
    ax[1].set_ylabel('SYM_H (Dst) Index (nT)')
    ax[2].set_ylabel('Density (cm$^{-3}$)')
    ax[3].set_ylabel('Speed (km/s)')

    ax[0].legend(loc='upper right', fontsize='small')

    ds_start = pd.Timestamp(omni_ds.time.values[0]).tz_localize("UTC")
    ds_end = pd.Timestamp(omni_ds.time.values[-1]).tz_localize("UTC")
    helper_overlay_icmes([plt.gca()], icme_df, ds_start, ds_end)
    
    # 3. Formatting and Ticks
    fig.suptitle("OMNI Magnetic Field & Plasma Parameters with C&R ICME Overlays"
                 f" ({ds_start.strftime('%Y-%m-%d')} to {ds_end.strftime('%Y-%m-%d')})",
                 y=0.99, fontsize=16)
    
    # Configure the shared X-axis (bottom plot)
    ax[3].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax[3].get_xticklabels(), rotation=45)
    
    for axis in ax:
        axis.grid(True, which='both', linestyle=':', alpha=0.4)

    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    plt.show()

# CLAUDE
def plot_icmes_B_components(
    omni_ds: xr.Dataset,
    icme_df: pd.DataFrame,
    show_derivative: bool = False,
    kernel_radius: int = 1,
) -> None:
    """
    Magnetic field component plot with ICME boundaries.

    Layout (show_derivative=False)
    ──────────────────────────────
    1 row × 5 columns, all sharing one time axis:
      Col 0 : Scalar |B|          (black)
      Col 1 : Vector |<B>|        (magenta)
      Col 2 : Bx_GSE              (dark red)
      Col 3 : By_GSE              (dark green)
      Col 4 : Bz_GSE              (dark blue)

    Layout (show_derivative=True)
    ──────────────────────────────
    2 rows × 5 columns — signal on top, derivative directly below,
    signal row taller than derivative row (height ratio 3:2):

      ┌──────────┬──────────┬──────────┬──────────┬──────────┐
      │  |B| sc  │  |B| vec │    Bx    │    By    │    Bz    │
      ├──────────┼──────────┼──────────┼──────────┼──────────┤
      │ d|B|sc/dt│d|B|vc/dt │  dBx/dt  │  dBy/dt  │  dBz/dt  │
      └──────────┴──────────┴──────────┴──────────┴──────────┘

    Parameters
    ----------
    omni_ds         : xarray Dataset with OMNI variables on a 'time' coordinate.
    icme_df         : DataFrame with columns:
                        disturbance_datetime_ut, icme_plasma_field_start_ut,
                        icme_plasma_field_end_ut
    show_derivative : If True, add a derivative row beneath each signal panel.
    kernel_radius   : Half-width of the centred-difference kernel (default 1 →
                      standard [-1, 0, 1] / 2Δt). Increase for smoother derivatives.
    """

    # ── Colour scheme ──────────────────────────────────────────────────────────
    # (signal colour, derivative colour)
    _PANEL = [
        ("black",   "#999999",  "$|B|$ scalar",      r"$\langle|B|\rangle$ sc",  r"d$\langle|B|\rangle$/dt"),
        ("chocolate", "#EB7B2C",  "$|B|$ vector",      r"$|\langle B\rangle|$ vec", r"d$|\langle B\rangle|$/dt"),
        ("#8B0000", "#E08080",  "$B_X$ GSE (nT)",    r"$B_X$ GSE",               r"d$B_X$/dt"),
        ("#006400", "#80C080",  "$B_Y$ GSE (nT)",    r"$B_Y$ GSE",               r"d$B_Y$/dt"),
        ("#00008B", "#8080E0",  "$B_Z$ GSE (nT)",    r"$B_Z$ GSE",               r"d$B_Z$/dt"),
    ]
    # columns: col_sig, col_d, ylabel_sig, legend_sig, legend_d

    LW = 0.6   # line width for all signal lines

    # ── Data ───────────────────────────────────────────────────────────────────

    time   = omni_ds.time.values
    dt_sec = helper_dt_seconds(omni_ds)

    scalar_B = omni_ds["F"].values.astype(float)
    vector_B = np.sqrt(
        omni_ds["BX_GSE"].values ** 2 +
        omni_ds["BY_GSE"].values ** 2 +
        omni_ds["BZ_GSE"].values ** 2
    ).astype(float)

    sig_data = [
        scalar_B,
        vector_B,
        omni_ds["BX_GSE"].values.astype(float),
        omni_ds["BY_GSE"].values.astype(float),
        omni_ds["BZ_GSE"].values.astype(float),
    ]

    # ── Figure layout ──────────────────────────────────────────────────────────
    n_cols = 5
    n_rows = 2 if show_derivative else 1

    fig, axes_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(22, 8 if show_derivative else 4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]} if show_derivative else {},
    )

    # Normalise to 2-D: axes_grid[row, col]
    if n_rows == 1:
        axes_grid = axes_grid[np.newaxis, :]

    sig_row   = axes_grid[0, :]
    deriv_row = axes_grid[1, :] if show_derivative else [None] * n_cols

    # ── Signal row ─────────────────────────────────────────────────────────────
    for col, (vals, (col_sig, col_d, ylabel, leg_sig, leg_d)) in enumerate(
        zip(sig_data, _PANEL)
    ):
        ax = sig_row[col]

        ax.plot(time, vals, color=col_sig, linewidth=LW, label=leg_sig)

        # Zero line for components; not useful for magnitudes
        if col >= 2:
            ax.axhline(0, color="#aaaaaa", linewidth=0.4, linestyle=":")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(leg_sig, fontsize=9, pad=3)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        if not show_derivative:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ── Derivative row ─────────────────────────────────────────────────────────
    if show_derivative:
        for col, (vals, (col_sig, col_d, ylabel, leg_sig, leg_d)) in enumerate(
            zip(sig_data, _PANEL)
        ):
            ax     = deriv_row[col]
            d_vals = helper_centred_diff(vals, dt_sec, kernel_radius)

            ax.plot(time, d_vals, color=col_d, linewidth=LW, label=leg_d)
            ax.axhline(0, color="#cccccc", linewidth=0.4, linestyle=":")
            ax.set_ylabel("d/dt (nT/s)", fontsize=8, color=col_d)
            ax.tick_params(axis="y", labelcolor=col_d, labelsize=7)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ── ICME overlays ─────────────────────────────────────────────────────────
    ds_start = pd.Timestamp(time[0]).tz_localize("UTC")
    ds_end   = pd.Timestamp(time[-1]).tz_localize("UTC")
    helper_overlay_icmes(list(axes_grid.flat), icme_df, ds_start, ds_end)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    deriv_note = (
        f" | derivative kernel radius={kernel_radius}"
        if show_derivative else ""
    )
    fig.suptitle(
        f"OMNI GSE Magnetic Field Components with C&R ICME Overlays{deriv_note}\n"
        f"({ds_start.strftime('%Y-%m-%d')} to {ds_end.strftime('%Y-%m-%d')})",
        fontsize=13,
    )

    fig.supxlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()

def main():
    omni_start = Date(2000, 3, 1)
    omni_end = Date(2000, 6, 30)

    omni_ds = omni_data_vis.get_omni_dataset(omni_start, omni_end, "5min")
    icme_df = cr_icme_vis.get_cr_icme_dataframe(start=omni_start, end=omni_end)

    plot_icmes_B_components(omni_ds, icme_df, show_derivative=True, kernel_radius=3)

if __name__ == "__main__":
    main()
