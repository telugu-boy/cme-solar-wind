import xarray as xr
import pandas as pd
from pathlib import Path

from omni_data_vis import *
from cr_icme_vis import *

def build_omni_cache(start, end, res="5min", cache_path=None):
    """
    Loads all monthly OMNI CDFs in [start, end], concatenates, and saves
    to a single Parquet file for fast reuse.
    """
    if cache_path is None:
        cache_path = Path(f"data/omni_cache_{res}.parquet")
    else:
        cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading cached OMNI data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("No cache found, building from raw CDF files (this may take a while)...")
    ds = get_omni_dataset(start, end, res)

    # Drop dims/coords we don't need, convert to DataFrame indexed by time
    df = ds.to_dataframe()
    df = df.reset_index()

    # Keep only time + data variables, drop any leftover index columns
    cols_to_drop = [c for c in df.columns if c == "Epoch"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.set_index("time").sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Cached {len(df)} rows to {cache_path}")

    return df


def append_to_omni_cache(new_start, new_end, res="5min", cache_path=None):
    """
    Extend an existing cache with a new date range without reloading
    everything. Deduplicates on time index.
    """
    if cache_path is None:
        cache_path = Path(f"data/omni_cache_{res}.parquet")
    else:
        cache_path = Path(cache_path)

    new_ds = get_omni_dataset(new_start, new_end, res)
    new_df = new_ds.to_dataframe().reset_index()
    new_df = new_df.drop(columns=[c for c in new_df.columns if c == "Epoch"], errors="ignore")
    new_df = new_df.set_index("time")

    if cache_path.exists():
        old_df = pd.read_parquet(cache_path)
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df.sort_index()

    cache_path.to_parquet if False else None  # no-op, just for clarity
    combined.to_parquet(cache_path)
    print(f"Cache now has {len(combined)} rows")
    return combined

def main():
    omni_start = Date(2001, 2, 1)
    omni_end = Date(2012, 12, 1)
    omni_df = build_omni_cache(omni_start, omni_end, res="5min")

if __name__ == "__main__":
    main()