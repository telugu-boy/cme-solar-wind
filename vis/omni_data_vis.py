import xarray as xr
import cdflib.xarray as cdf_xr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from datetime import date as Date
from typing import Literal

import vis.cr_icme_vis as cr_icme_vis

def get_omni_filepath(date: Date, res: Literal["1min"] | Literal["5min"]):
    omni_folder_name = None
    omni_folder = None
    if res == "1min":
        omni_folder_name = "omni_hro2_1min"
    elif res == "5min":
        omni_folder_name = "omni_hro2_5min"
    omni_folder = Path("data/"+omni_folder_name)
    
    year_str = date.strftime("%Y")
    cdf_filename = f"{omni_folder_name}_{date.strftime('%Y%m01')}_v01.cdf"
    return omni_folder / year_str / cdf_filename

def get_omni_dataset(start: Date, end: Date, res: Literal["1min"] | Literal["5min"]):
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
def plot_omni_B_density_velocity(ds: xr.Dataset):
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
    """
    OMNI Dataset Column Descriptions:

    Time and Data Quality Identifiers
    YR: Year
    Day: Day of year
    HR: Hour
    Minute: Minute
    IMF: IMF spacecraft ID
    PLS: Plasma spacecraft ID
    IMF_PTS: Number of points in the IMF average
    PLS_PTS: Number of points in the plasma average
    percent_interp: Percent of interpolation
    Timeshift: Time shift (seconds)
    RMS_Timeshift: RMS of the time shift
    Time_btwn_obs: Time between observations

    Magnetic Field (IMF) Data
    F: Scalar Magnitude of the Magnetic Field (B)
    BX_GSE: X-component of B (GSE coordinate system)
    BY_GSE: Y-component of B (GSE)
    BZ_GSE: Z-component of B (GSE)
    BY_GSM: Y-component of B (GSM coordinate system)
    BZ_GSM: Z-component of B (GSM)
    RMS_SD_B: RMS Standard Deviation of B magnitude
    RMS_SD_fld_vec: RMS Standard Deviation of the field vector

    Plasma and Solar Wind Parameters
    flow_speed: Solar wind speed (V)
    Vx: Velocity X-component
    Vy: Velocity Y-component
    Vz: Velocity Z-component
    proton_density: Proton density (n)
    T: Proton temperature
    NaNp_Ratio: Alpha/Proton ratio
    Pressure: Flow pressure
    E: Electric field
    Beta: Plasma Beta
    Mach_num: Alfven Mach number
    Mgs_mach_num: Magnetosonic Mach number

    Spacecraft Position and Bow Shock
    x: Spacecraft X-position
    y: Spacecraft Y-position
    z: Spacecraft Z-position
    BSN_x: Bow Shock Nose X-position
    BSN_y: Bow Shock Nose Y-position
    BSN_z: Bow Shock Nose Z-position

    Geomagnetic Indices
    AE_INDEX: Auroral Electrojet index
    AL_INDEX: Lower Auroral Electrojet index
    AU_INDEX: Upper Auroral Electrojet index
    SYM_D: Symmetric D-component
    SYM_H: Symmetric H-component
    ASY_D: Asymmetric D-component
    ASY_H: Asymmetric H-component

    Proton Flux
    PR-FLX_10: Proton flux (>10 MeV)
    PR-FLX_30: Proton flux (>30 MeV)
    PR-FLX_60: Proton flux (>60 MeV)
    """

    omni_start = Date(2000, 1, 1)
    omni_end = Date(2000, 2, 1)

    omni_ds = get_omni_dataset(omni_start, omni_end, "5min")
    print(omni_ds.data_vars)

    plot_omni_B_density_velocity(omni_ds)

if __name__ == "__main__":
    main()