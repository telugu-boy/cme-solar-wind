import numpy as np
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

def get_icme_duration_summary(df: pd.DataFrame, log_flag: bool = False):
    """
    Returns 5-point summary plus Mean, Skew, and Kurtosis.
    """
    durations = (df['icme_plasma_field_end_ut'] - df['icme_plasma_field_start_ut']).dt.total_seconds() / 3600.0
    durations = durations.dropna()
    
    if log_flag:
        # Log10 transform for normalizing highly skewed data
        durations = np.log10(durations[durations > 0])
        label = "Log10(Hours)"
    else:
        label = "Hours"

    summary = durations.describe()
    skew_val = durations.skew()
    kurt_val = durations.kurt()
    mean_val = durations.mean() # Re-added the mean
    
    print(f"--- ICME Duration Statistics ({label}) ---")
    print(f"Count:    {summary['count']:.0f}")
    print(f"Min:      {summary['min']:.4f}")
    print(f"25% (Q1): {summary['25%']:.4f}")
    print(f"Median:   {summary['50%']:.4f}")
    print(f"75% (Q3): {summary['75%']:.4f}")
    print(f"Max:      {summary['max']:.4f}")
    print(f"Mean:     {mean_val:.4f}") # Display mean
    print(f"Skew:     {skew_val:.4f}")
    print(f"Kurtosis: {kurt_val:.4f}")
    
    return durations

def plot_icme_duration_histogram(df: pd.DataFrame, log_flag: bool = False):
    """
    Plots a histogram of ICME durations with Median and Mean markers.
    """
    durations = (df['icme_plasma_field_end_ut'] - df['icme_plasma_field_start_ut']).dt.total_seconds() / 3600.0
    durations = durations.dropna()

    if log_flag:
        data_to_plot = np.log10(durations[durations > 0])
        xlabel = "$\log_{10}$(Duration in Hours)"
        title = "Log-Transformed ICME Durations"
    else:
        data_to_plot = durations
        xlabel = "Duration (Hours)"
        title = "Raw ICME Durations"

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
    
    # Red dashed line for Median
    median_val = data_to_plot.median()
    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2, 
                label=f'Median: {median_val:.2f}')
    
    # Green dotted line for Mean
    mean_val = data_to_plot.mean()
    plt.axvline(mean_val, color='green', linestyle='dotted', linewidth=2, 
                label=f'Mean: {mean_val:.2f}')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Events")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

from scipy.stats import entropy

def compare_distribution_entropy(df: pd.DataFrame):
    """
    Calculates and compares the entropy of raw vs log-transformed durations.
    """
    durations = (df['icme_plasma_field_end_ut'] - df['icme_plasma_field_start_ut']).dt.total_seconds() / 3600.0
    durations = durations.dropna()
    
    # We use a fixed number of bins to make the comparison fair
    bins = 50
    
    # Raw Entropy
    hist_raw, _ = np.histogram(durations, bins=bins, density=True)
    ent_raw = entropy(hist_raw)
    
    # Log Entropy
    log_durations = np.log10(durations[durations > 0])
    hist_log, _ = np.histogram(log_durations, bins=bins, density=True)
    ent_log = entropy(hist_log)
    
    print(f"Entropy (Raw Hours): {ent_raw:.4f}")
    print(f"Entropy (Log10 Hours): {ent_log:.4f}")

def main():
    cr_icmes = get_cr_icme_dataframe()

    # can use as hyperparam in the future
    log_flag = True

    compare_distribution_entropy(cr_icmes)
    return
    print(get_icme_duration_summary(cr_icmes, log_flag))
    plot_icme_duration_histogram(cr_icmes, log_flag)
    return

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