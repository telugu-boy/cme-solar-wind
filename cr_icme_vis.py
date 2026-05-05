import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date as Date
from typing import Optional

def get_cr_icme_dataframe(start:Optional[Date]=None, end:Optional[Date]=None):
    cr_icmes = pd.read_csv("data/icme_catalog.csv")

    time_cols = ['disturbance_datetime_ut', 'icme_plasma_field_start_ut', 'icme_plasma_field_end_ut']

    # we trim footnotes stuff like (A), (B)
    for col in time_cols:
        cr_icmes[col] = pd.to_datetime(cr_icmes[col].astype(str).str.replace(r'\(.*\)', '', regex=True).str.strip(), 
                                    format="%Y/%m/%d %H%M", 
                                    utc=True, 
                                    errors='coerce')

    # Convert numeric columns
    cols_to_int = ['comp_start_hrs', 'comp_end_hrs', 'mc_start_hrs', 'mc_end_hrs']
    cr_icmes[cols_to_int] = cr_icmes[cols_to_int].apply(pd.to_numeric, errors='coerce').astype('Int64')

    if start is not None and end is not None:
        start_ts = pd.to_datetime(start).tz_localize('UTC')
        end_ts = pd.to_datetime(end).tz_localize('UTC')

        cr_icmes = cr_icmes[cr_icmes['disturbance_datetime_ut'].between(start_ts, end_ts)]

    return cr_icmes

def main():
    cr_icmes = get_cr_icme_dataframe()

    # 3. Filter to Solar Cycle (May 1996 - Nov 2008)
    cutoff1 = pd.Timestamp("1996-05-01", tz="UTC")
    cutoff2 = pd.Timestamp("2008-11-01", tz="UTC")

    data_subset = cr_icmes[cr_icmes['disturbance_datetime_ut'].between(cutoff1, cutoff2)].copy()

    # 4. Plotting with Matplotlib
    plt.figure(figsize=(12, 4))

    # Create a baseline y=1 for all points
    y_val = 1

    # Add the 'Linking Lines' (segments)
    for i, row in data_subset.iterrows():
        plt.plot([row['icme_plasma_field_start_ut'], row['icme_plasma_field_end_ut']], 
                [y_val, y_val], color='gray', linewidth=2, zorder=1)

    # Add the 'Pings' (points/markers)
    plt.scatter(data_subset['disturbance_datetime_ut'], [y_val]*len(data_subset), 
                marker='|', color='black', s=200, label='Disturbance', zorder=2)

    plt.scatter(data_subset['icme_plasma_field_start_ut'], [y_val]*len(data_subset), 
                marker='|', color='limegreen', s=200, label='Plasma Start', zorder=2)

    plt.scatter(data_subset['icme_plasma_field_end_ut'], [y_val]*len(data_subset), 
                marker='|', color='red', s=200, label='Plasma End', zorder=2)

    # 5. Formatting
    plt.ylim(0.8, 1.2)
    plt.yticks([]) # Hide Y axis ticks like yaxt="n"
    plt.title("ICME Event Timelines (1D)")
    plt.xlabel("Time (UTC)")

    # Format X-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()