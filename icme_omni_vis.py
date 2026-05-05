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
    ds_start = omni_ds.time.values[0]
    ds_end = omni_ds.time.values[-1]

    for i, row in icme_df.iterrows():
        # Check if the event is within our current OMNI dataset window
        if pd.isna(row['icme_plasma_field_start_ut']):
            continue
            
        t_dist = row['disturbance_datetime_ut']
        t_start = row['icme_plasma_field_start_ut']
        t_end = row['icme_plasma_field_end_ut']

        # Draw a vertical line for the Disturbance (Shock)
        if not pd.isna(t_dist):
            plt.axvline(t_dist, color='blue', linestyle='--', alpha=0.7, label='Disturbance' if i == 0 else "")

        # Highlight the ICME body with a shaded region (Plasma/Field region)
        plt.axvspan(t_start, t_end, color='limegreen', alpha=0.2, label='ICME Body' if i == 0 else "")
        
        # Add a text label above the peak for ID (optional)
        plt.text(t_start, plt.gca().get_ylim()[1]*0.9, f"Event {i}", rotation=75, fontsize=8, alpha=0.6)

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
    b_vec_mag = np.sqrt(omni_ds['BX_GSE']**2 + omni_ds['BY_GSM']**2 + omni_ds['BZ_GSM']**2)
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

    # 2. Iterate through catalog and draw on ALL panels
    for i, row in icme_df.iterrows():
        t_start = row['icme_plasma_field_start_ut']
        if pd.isna(t_start) or t_start < ds_start or t_start > ds_end:
            continue
            
        t_dist = row['disturbance_datetime_ut']
        t_end = row['icme_plasma_field_end_ut']

        # Apply boundaries to every axis in the stack
        for axis in ax:
            # Shaded ICME Body
            axis.axvspan(t_start, t_end, color='limegreen', alpha=0.15, 
                         label='ICME Body' if i == 0 and axis == ax[0] else "")
            
            # Disturbance/Shock Line
            if not pd.isna(t_dist):
                axis.axvline(t_dist, color='red', linestyle='--', alpha=0.6, 
                             label='Shock' if i == 0 and axis == ax[0] else "")

        # Add event ID text only on the top panel
        ax[0].text(t_start, ax[0].get_ylim()[1]*0.85, f"Event {i}", 
                   rotation=45, fontsize=9, fontweight='bold', alpha=0.7)

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

def main():
    omni_start = Date(2000, 3, 1)
    omni_end = Date(2000, 6, 30)

    omni_ds = omni_data_vis.get_omni_dataset(omni_start, omni_end, "5min")
    icme_df = cr_icme_vis.get_cr_icme_dataframe(start=omni_start, end=omni_end)

    plot_icmes_B_density_velocity(omni_ds, icme_df)

if __name__ == "__main__":
    main()
