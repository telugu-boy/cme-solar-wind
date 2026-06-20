from datetime import date as Date
from pathlib import Path
from typing import Optional  # Fixed: Added missing import

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_omni_cache(cache_path: Path):
    """Reads and returns the cached OMNI Parquet file."""
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache file found at {cache_path}")

    print(f"Reading OMNI cache from {cache_path}")
    df = pd.read_parquet(cache_path)

    # Ensure the index is a datetime index and timezone-naïve for uniform numpy manipulation
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:  # Adjust if your time column has a different name
            df = df.set_index(pd.to_datetime(df["Timestamp"]))
        else:
            raise ValueError(
                "OMNI dataframe does not have a DatetimeIndex and no obvious timestamp column was found."
            )

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def get_cr_icme_dataframe(
    start: Optional[Date] = None, end: Optional[Date] = None
):
    cr_icmes = pd.read_csv("data/icme_catalog.csv")

    time_cols = [
        "disturbance_datetime_ut",
        "icme_plasma_field_start_ut",
        "icme_plasma_field_end_ut",
    ]

    # Trim footnotes stuff like (A), (B) and convert to timezone-naive datetime
    for col in time_cols:
        cr_icmes[col] = pd.to_datetime(
            cr_icmes[col]
            .astype(str)
            .str.replace(r"\(.*\)", "", regex=True)
            .str.strip(),
            format="%Y/%m/%d %H%M",
            errors="coerce",
        )
        if cr_icmes[col].dt.tz is not None:
            cr_icmes[col] = cr_icmes[col].dt.tz_localize(None)

    # Convert numeric columns
    cols_to_int = [
        "comp_start_hrs",
        "comp_end_hrs",
        "mc_start_hrs",
        "mc_end_hrs",
    ]
    cr_icmes[cols_to_int] = (
        cr_icmes[cols_to_int].apply(pd.to_numeric, errors="coerce").astype("Int64")
    )

    if start is not None and end is not None:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        cr_icmes = cr_icmes[
            cr_icmes["disturbance_datetime_ut"].between(start_ts, end_ts)
        ]

    return cr_icmes


class OmniWindowDataset(Dataset):
    """Sliding-window dataset over OMNI solar wind data, with binary ICME

    labels per timestep based on the ICME catalog.
    """

    def __init__(
        self,
        omni_df: pd.DataFrame,
        cr_icmes: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = 288,  # 24h at 5min resolution
        stride: int = 1,
        normalize: bool = True,
        scaler=None,
    ):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.stride = stride

        # Extract columns and work with a clean, contiguous copy
        df = omni_df[feature_cols].copy()

        # Handle small missing value gaps without dropping rows (maintaining time step structure)
        df = df.interpolate(limit=6, limit_direction="both")

        # If any rows still contain NaN values after interpolation, fill with 0 or forward fill
        # so matrix operations do not break, or drop carefully if necessary.
        df = df.fillna(0.0)

        self.times = df.index
        self.data = df.values.astype(np.float32)

        # Build a binary label array aligned with self.times (Timezone Naive)
        labels = np.zeros(len(self.times), dtype=np.float32)
        intervals = cr_icmes[
            ["icme_plasma_field_start_ut", "icme_plasma_field_end_ut"]
        ].dropna()

        times_arr = self.times.values  # numpy datetime64 array (naive)
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
                self.data = self.scaler.fit_transform(self.data).astype(
                    np.float32
                )
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(self.data).astype(np.float32)
        else:
            self.scaler = None

        # Precompute valid window start indices
        self.window_starts = []
        time_diffs = np.diff(times_arr).astype("timedelta64[m]").astype(int)

        for start_idx in range(0, len(self.data) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_diffs = time_diffs[start_idx : end_idx - 1]

            # Up to 10-minute gap allowed for a 5-min resolution setup
            if len(window_diffs) == 0 or window_diffs.max() <= 10:
                self.window_starts.append(start_idx)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end = start + self.window_size

        x = torch.from_numpy(
            self.data[start:end]
        )  # (window_size, n_features)
        y = torch.from_numpy(self.labels[start:end])  # (window_size,)

        return x, y


def main():
    omni_start = Date(1996, 5, 1)
    omni_end = Date(2020, 1, 1)

    omni_full_df = read_omni_cache(Path("data/omni_cache_5min_full.parquet"))

    # Explicit string/timestamp-based indexing for safety
    omni_solarmax1_df = omni_full_df.loc[
        str(omni_start) : str(omni_end)
    ].copy()
    cr_icme_df = get_cr_icme_dataframe(omni_start, omni_end)

    feature_cols = [
        "F",
        "BX_GSE",
        "BY_GSE",
        "BZ_GSE",
        "flow_speed",
        "proton_density",
        "T",
        "Pressure",
    ]

    train_dataset = OmniWindowDataset(
        omni_df=omni_solarmax1_df,
        cr_icmes=cr_icme_df,
        feature_cols=feature_cols,
        window_size=(60 * 32 // 5),  # Split into windows of 32h, 36h, 48h
        stride=60 * 1 // 5, # 1 hour = 12 data points stride
    )

    print(f"Dataset successfully created with {len(train_dataset)} windows!")


if __name__ == "__main__":
    main()