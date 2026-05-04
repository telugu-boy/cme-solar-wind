import xarray as xr
import cdflib.xarray as cdf_xr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from datetime import date as Date
from typing import Literal

def get_omni_filepath(date: Date, res: Literal["1min"] | Literal["5min"]):
    omni_folder_name = None
    omni_folder = None
    if res == "1min":
        omni_folder_name = "omni_hro2_1min"
    elif res == "5min":
        omni_folder_name = "omni_hro2_5min"
    omni_folder = Path(omni_folder_name)
    
    year_str = date.strftime("%Y")
    cdf_filename = f"{omni_folder_name}_{date.strftime('%Y%m01')}_v01.cdf"
    return omni_folder / year_str / cdf_filename

def get_omni_dataset_in_range(start: Date, end: Date, res: Literal["1min"] | Literal["5min"]):
    """
    `start` and `end` should contain the year and month; day is ignored.

    Returns: xarray dataset, with the pd.datetime `time` coordinate aligned to Epoch
    """
    # Create a range of months between start and end
    date_range = pd.date_range(start=start, end=end, freq='MS')
    datasets = []

    for d in date_range:
        path = get_omni_filepath(d.date(), res)
        if path.exists():
            # Load file; to_unixtime=True facilitates easier datetime conversion later
            month_ds = cdf_xr.cdf_to_xarray(str(path), to_unixtime=True, fillval_to_nan=True)
            datasets.append(month_ds)
        else:
            print(f"Warning: File not found for {d.strftime('%Y-%m')}")

    if not datasets:
        raise FileNotFoundError("No OMNI files found in the specified range.")

    # Combine all months along the Epoch dimension
    ds: xr.Dataset = xr.concat(datasets, dim='Epoch')

    # Assign coordinates to satisfy Pylance
    datetime_vals = pd.to_datetime(ds['Epoch'].values, unit='s', utc=True)
    ds = ds.assign_coords(time=('Epoch', datetime_vals))
    ds = ds.swap_dims({'Epoch': 'time'})

    return ds

# we now want to plot this data for icmes
# we are looking at the magnetic field B, particle density, particle velocity
def plot_B_density_velocity(ds: xr.Dataset):
    datetime_cols = (
        "YR", "Day", "HR", "Minute"
    )

    magnetic_cols = (
        "F", "BX_GSE", "BY_GSE", "BZ_GSE", "RMS_SD_B", "RMS_SD_fld_vec"
    )

    vel_cols = (
        "flow_speed", "Vx", "Vy", "Vz"
    )

    proton_density = "proton_density"

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Total Magnetic Field (F)
    ax[0].plot(ds.time, ds['F'], color='black', label='|B|')
    ax[0].set_ylabel('B (nT)')

    # 2. Proton Density
    ax[1].plot(ds.time, ds['proton_density'], color='green')
    ax[1].set_ylabel('Proton Density N (cm$^{-3}$)')

    # 3. Flow Speed
    ax[2].plot(ds.time, ds['flow_speed'], color='blue')
    ax[2].set_ylabel('V (km/s)')

    delta = pd.to_datetime(ds.time.values[-1]) - pd.to_datetime(ds.time.values[0])
    duration_days = delta.days

    if duration_days <= 31:
        # Weekly or less: Show daily ticks
        ax[2].xaxis.set_major_locator(mdates.DayLocator())
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif duration_days <= 120:
        # 1-2 Months: Show weekly ticks
        ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)) # type: ignore
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        # Longer ranges: Show monthly ticks
        ax[2].xaxis.set_major_locator(mdates.MonthLocator())
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)

    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    plt.show()

def main():
    omni_start = Date(2000, 1, 1)
    omni_end = Date(2000, 12, 31)
    ds = get_omni_dataset_in_range(omni_start, omni_end, "5min")

    plot_B_density_velocity(ds)

if __name__ == "__main__":
    main()