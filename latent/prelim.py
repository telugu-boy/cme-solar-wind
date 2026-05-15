import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch, spectrogram
from datetime import date as Date

import vis.omni_data_vis as omni_data_vis
import vis.cr_icme_vis as cr_icme_vis
import vis.icme_omni_vis as icme_vis

def strip_tz(ds: xr.Dataset) -> xr.Dataset:
    """Safely removes timezone info from the 'time' coordinate for xarray math."""
    # Convert to pandas index, strip TZ, and re-assign
    naive_times = pd.DatetimeIndex(ds.time.values).tz_localize(None)
    return ds.assign_coords(time=naive_times)

def plot_psd_group(ds: xr.Dataset, variables: list, titles: list, suptitle: str):
    """Plots Welch PSD on a log-log scale against Period (Hours)."""
    dt = icme_vis.helper_dt_seconds(ds)
    fs = 1.0 / dt
    
    ds_naive = strip_tz(ds)
    
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)
    if len(variables) == 1: axes = [axes]

    for ax, var, title in zip(axes, variables, titles):
        signal = ds_naive[var].interpolate_na(dim="time", method="linear").values
        f, Pxx = welch(signal, fs=fs, nperseg=1024)
        
        mask = f > 0
        period_hours = (1.0 / f[mask]) / 3600.0
        
        # LOG-LOG plot
        ax.loglog(period_hours, Pxx[mask], color='black', lw=1.2)
        
        ax.set_title(title, weight='bold')
        ax.set_ylabel(r'PSD [Units$^2$/Hz]')
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Period [Hours]')
    
    # Set useful ticks for the log scale
    tick_locs = [0.1, 0.5, 1, 3, 6, 12, 24, 48]
    axes[-1].set_xticks(tick_locs)
    axes[-1].get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.suptitle(suptitle, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def plot_spectrogram_with_icmes(ds: xr.Dataset, icme_df: pd.DataFrame, var='F'):
    dt = icme_vis.helper_dt_seconds(ds)
    fs = 1.0 / dt
    
    ds_naive = strip_tz(ds)
    signal = ds_naive[var].interpolate_na(dim="time", method="linear").values
    
    orig_times = ds.time.values 
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
    
    start_time = pd.Timestamp(orig_times[0]).tz_localize(None)
    t_dates = [start_time + pd.Timedelta(seconds=sec) for sec in t]

    fig, ax = plt.subplots(figsize=(15, 7))
    # Using 'magma' for high contrast with white overlays
    pcm = ax.pcolormesh(t_dates, f, np.log10(Sxx), shading='gouraud', cmap='magma', vmin=-2, vmax=4)
    plt.colorbar(pcm, ax=ax, label='Log$_{10}$(Power)', pad=0.01)

    # Use original ds for time range logic
    ds_start = pd.Timestamp(orig_times[0])
    ds_end = pd.Timestamp(orig_times[-1])
    if ds_start.tzinfo is None:
        ds_start, ds_end = ds_start.tz_localize("UTC"), ds_end.tz_localize("UTC")

    # Pass the white color and alpha to the helper
    icme_vis.helper_overlay_icmes([ax], icme_df, ds_start, ds_end, color='white', alpha=0.3)

    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(f'Spectrogram: {var} (ICMEs in Transparent White)', weight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def fft_main():
    # Example range: looking at the Bastille Day event era or similar
    start, end = Date(2000, 5, 1), Date(2000, 7, 31)
    
    omni_ds = omni_data_vis.get_omni_dataset(start, end, "5min")
    icme_df = cr_icme_vis.get_cr_icme_dataframe(start=start, end=end)

    # 1. Magnetic Field PSD
    plot_psd_group(
        omni_ds, 
        ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE'],
        ['Total |B|', '$B_x$ GSE', '$B_y$ GSE', '$B_z$ GSE'],
        'Power Spectral Density: Magnetic Field'
    )

    # 2. Solar Wind Velocity PSD
    plot_psd_group(
        omni_ds, 
        ['flow_speed', 'Vx', 'Vy', 'Vz'],
        ['Total Speed ($V_{sw}$)', '$V_x$', '$V_y$', '$V_z$'],
        'Power Spectral Density: Solar Wind Velocity'
    )

    # 3. Plasma Density PSD
    plot_psd_group(
        omni_ds, 
        ['proton_density'], 
        ['Proton Density ($N_p$)'],
        'Power Spectral Density: Proton Density'
    )

    # 4. Spectrogram
    plot_spectrogram_with_icmes(omni_ds, icme_df, var='F')

if __name__ == "__main__":
    fft_main()