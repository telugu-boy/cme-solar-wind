import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

from datetime import date as Date

def read_omni_cache(cache_path: Path):
    """
    Reads and returns the cached OMNI Parquet file.
    Raises FileNotFoundError if the cache does not exist.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache file found at {cache_path}")

    print(f"Reading OMNI cache from {cache_path}")
    return pd.read_parquet(cache_path)

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


class OmniWindowDataset(Dataset):
    """
    Sliding-window dataset over OMNI solar wind data, with binary
    ICME labels per timestep based on the ICME catalog.

    Each sample is (window, label_window) where:
      - window: (window_size, n_features) tensor of solar wind features
      - label_window: (window_size,) tensor of 0/1 ICME flags
    """

    def __init__(
        self,
        omni_df: pd.DataFrame,
        cr_icmes: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = 288,   # e.g. 24h at 5min resolution
        stride: int = 1,
        normalize: bool = True,
        scaler=None,
    ):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.stride = stride

        df = omni_df[feature_cols].copy()

        # Drop rows where all features are NaN, but keep time index intact
        # for label alignment; interpolate small gaps
        df = df.interpolate(limit=6, limit_direction="both")
        df = df.dropna()

        self.times = df.index
        self.data = df.values.astype(np.float32)

        # Build a binary label array aligned with self.times:
        # 1 if timestamp falls within any ICME plasma field interval
        labels = np.zeros(len(self.times), dtype=np.float32)
        intervals = cr_icmes[
            ["icme_plasma_field_start_ut", "icme_plasma_field_end_ut"]
        ].dropna()

        times_arr = self.times.values  # numpy datetime64 array
        for _, row in intervals.iterrows():
            start = np.datetime64(row["icme_plasma_field_start_ut"])
            end = np.datetime64(row["icme_plasma_field_end_ut"])
            mask = (times_arr >= start) & (times_arr <= end)
            labels[mask] = 1.0

        self.labels = labels

        # Normalization
        if normalize:
            if scaler is None:
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                self.data = self.scaler.fit_transform(self.data).astype(np.float32)
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(self.data).astype(np.float32)
        else:
            self.scaler = None

        # Precompute valid window start indices.
        # Optionally, you may want to skip windows with large time gaps;
        # this checks max gap between consecutive timestamps in the window.
        self.window_starts = []
        time_diffs = np.diff(times_arr).astype("timedelta64[m]").astype(int)

        for start_idx in range(0, len(self.data) - window_size + 1, stride):
            end_idx = start_idx + window_size
            # check for gaps within this window
            window_diffs = time_diffs[start_idx:end_idx - 1]
            if len(window_diffs) == 0 or window_diffs.max() <= 10:  # allow up to 10 min gaps for 5min res
                self.window_starts.append(start_idx)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end = start + self.window_size

        x = torch.from_numpy(self.data[start:end])             # (window_size, n_features)
        y = torch.from_numpy(self.labels[start:end])            # (window_size,)

        return x, y

def main():
    # FULL omni data - Jan 1996 to April 2026
    # Solar Max 1 Omni data - May 1996 to Dec 2008
    # Solar Min 1 Omni data - Feb 2001 to Dec 2012
    # Two full solar cycles - May 1996 to Jan 2020
    omni_start = Date(1996, 5, 1)
    omni_end = Date(2020, 1, 1)

    omni_full_df = read_omni_cache(Path("data/omni_cache_5min_full.parquet"))
    omni_solarmax1_df = omni_full_df.loc[omni_start:omni_end]
    cr_icme_df = get_cr_icme_dataframe(omni_start, omni_end)

    feature_cols = ["F", "BX_GSE", "BY_GSE", "BZ_GSE", "flow_speed", "proton_density", "T", "Pressure"]

    train_dataset = OmniWindowDataset(
        omni_df = omni_solarmax1_df,
        cr_icmes=cr_icme_df,
        feature_cols=feature_cols,
        window_size=(60//5*24), # 288 5-min data points, so 24 hour intervals
        stride=1,
    )

if __name__ == "__main__":
    main()