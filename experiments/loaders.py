"""
loaders.py
----------
Data loading, feature engineering, and PyTorch dataset modules for PatchTSMixer.
"""

from __future__ import annotations

import warnings
from datetime import date as Date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

CARRINGTON_PERIOD_DAYS = 27.2753   # synodic Carrington rotation period


def read_omni_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache file found at {cache_path}")
    print(f"[data] Reading OMNI cache from {cache_path}")
    df = pd.read_parquet(cache_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["Timestamp"]))
        else:
            raise ValueError("No DatetimeIndex and no Timestamp column found.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def get_cr_icme_dataframe(
    start: Optional[Date] = None,
    end: Optional[Date] = None,
    path: str = "data/icme_catalog.csv",
) -> pd.DataFrame:
    cr = pd.read_csv(path)
    time_cols = [
        "disturbance_datetime_ut",
        "icme_plasma_field_start_ut",
        "icme_plasma_field_end_ut",
    ]
    for col in time_cols:
        cr[col] = pd.to_datetime(
            cr[col].astype(str).str.replace(r"\(.*\)", "", regex=True).str.strip(),
            format="%Y/%m/%d %H%M",
            errors="coerce",
        )
        if cr[col].dt.tz is not None:
            cr[col] = cr[col].dt.tz_localize(None)
    int_cols = ["comp_start_hrs", "comp_end_hrs", "mc_start_hrs", "mc_end_hrs"]
    cr[int_cols] = (
        cr[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
    )
    if start is not None and end is not None:
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        cr = cr[cr["disturbance_datetime_ut"].between(s, e)]
    return cr


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Derive heliophysics features from raw OMNI columns.

    All division operations are guarded by .clip(lower=1e-6) to avoid NaN
    from near-zero denominators (common in fill-value rows in OMNI).

    Raw columns used:  F, BX_GSE, BY_GSE, BZ_GSE
    Optional raw:      f10.7_index (or similar column name)

    Returns a new DataFrame with only the engineered columns added.
    """
    out = df.copy()

    Bx = df["BX_GSE"].values
    By = df["BY_GSE"].values
    Bz = df["BZ_GSE"].values
    B  = np.abs(df["F"].values).clip(1e-6)    # |B| — clip fill values

    # Clock angle  θ_clock = atan2(Bz, By)
    r_perp = np.sqrt(By**2 + Bz**2).clip(1e-6)
    if cfg.get("use_sin_clock"):
        out["sin_clock"] = Bz / r_perp
    if cfg.get("use_cos_clock"):
        out["cos_clock"] = By / r_perp

    # Cone angle  θ_cone = arccos(Bx / |B|)
    cos_cone = np.clip(Bx / B, -1.0, 1.0)
    if cfg.get("use_cos_cone"):
        out["cos_cone"] = cos_cone
    if cfg.get("use_sin_cone"):
        out["sin_cone"] = np.sqrt(np.clip(1.0 - cos_cone**2, 0.0, 1.0))

    # Rolling RMS of |B|  (1-hour window at 5-min resolution = 12 steps)
    if cfg.get("use_rms_B"):
        w = cfg.get("rms_window", 12)
        out["rms_B"] = (
            df["F"].pow(2)
            .rolling(window=w, min_periods=1)
            .mean()
            .pipe(np.sqrt)
        )

    # 27.27-day Carrington rotation (sin + cos for phase continuity)
    if cfg.get("use_carrington_wave"):
        t_sec = (df.index - df.index[0]).total_seconds().values
        omega = 2.0 * np.pi / (CARRINGTON_PERIOD_DAYS * 86400.0)
        out["carr_sin"] = np.sin(omega * t_sec)
        out["carr_cos"] = np.cos(omega * t_sec)

    # f10.7 solar flux index (if present)
    if cfg.get("use_f10_7"):
        f107_col = next(
            (c for c in df.columns if "f10" in c.lower() or "f107" in c.lower()),
            None,
        )
        if f107_col is not None:
            out["f10_7"] = df[f107_col]
        else:
            warnings.warn("use_f10_7=True but no f10.7 column found in OMNI cache; skipping.")

    return out


def select_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """
    Determine which columns to pass to the model based on CFG toggles.
    Order: raw → sin_clock → cos_clock → cos_cone → sin_cone → rms_B
           → f10_7 → carr_sin → carr_cos
    """
    if cfg.get("univariate_test"):
        return ["F"]

    cols = list(cfg["raw_feature_cols"])
    for derived in [
        "sin_clock", "cos_clock", "cos_cone", "sin_cone",
        "rms_B", "f10_7", "carr_sin", "carr_cos",
    ]:
        if derived in df.columns:
            cols.append(derived)
    # Keep only columns actually present
    return [c for c in cols if c in df.columns]


def build_icme_intervals(cr_icmes: pd.DataFrame) -> list[tuple[np.datetime64, np.datetime64]]:
    intervals = []
    for _, row in cr_icmes[
        ["icme_plasma_field_start_ut", "icme_plasma_field_end_ut"]
    ].dropna().iterrows():
        intervals.append((
            np.datetime64(row["icme_plasma_field_start_ut"]),
            np.datetime64(row["icme_plasma_field_end_ut"]),
        ))
    return intervals


class OmniPatchDataset(Dataset):
    """
    Sliding-window dataset for PatchTSMixer pretraining on OMNI solar wind.

    Each sample provides:
      past_values   : FloatTensor (context_length, n_features)
      future_values : FloatTensor (prediction_length, n_features)  — for forecast head
      patch_labels  : FloatTensor (num_patches,) binary             — for anomaly head

    Patch labelling
    ---------------
    patch_labels[p] = 1  iff  (# ICME timesteps in patch p) / patch_length
                              >= overlap_threshold

    Continuity check
    ----------------
    Windows spanning data gaps (> 10 min between 5-min samples) are silently
    excluded, matching the existing OmniWindowDataset convention.
    """

    def __init__(
        self,
        data: np.ndarray,                     # (N, C) float32, already normalised
        times: pd.DatetimeIndex,
        icme_intervals: list[tuple[np.datetime64, np.datetime64]],
        context_length: int,
        prediction_length: int,
        patch_length: int,
        patch_stride: int,
        overlap_threshold: float = 0.10,
        window_stride: int = 12,
    ) -> None:
        self.data = data
        self.times = times
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.overlap_threshold = overlap_threshold
        self.window_stride = window_stride

        total_len = context_length + prediction_length

        # num_patches formula mirrors PatchTSMixerConfig
        self.num_patches: int = (
            max(context_length, patch_length) - patch_length
        ) // patch_stride + 1

        # ── Per-timestep ICME label ───────────────────────────────────────
        times_arr = times.values
        self.timestep_labels = np.zeros(len(times_arr), dtype=np.float32)
        for start_dt, end_dt in icme_intervals:
            mask = (times_arr >= start_dt) & (times_arr <= end_dt)
            self.timestep_labels[mask] = 1.0

        # ── Precompute valid window starts ────────────────────────────────
        diffs = np.diff(times_arr).astype("timedelta64[m]").astype(int)
        self.window_starts: list[int] = []

        for s in range(0, len(self.data) - total_len + 1, window_stride):
            e = s + total_len
            gap = diffs[s : e - 1].max() if e - 1 > s else 0
            if gap <= 10:
                self.window_starts.append(s)

        # ── Precompute patch labels (Vectorized cumulative sum) ───────────
        cumsum_labels = np.zeros(len(self.timestep_labels) + 1, dtype=np.float32)
        cumsum_labels[1:] = np.cumsum(self.timestep_labels)

        self.precomputed_patch_labels = np.zeros(
            (len(self.window_starts), self.num_patches), dtype=np.float32
        )

        if len(self.window_starts) > 0:
            window_starts_arr = np.array(self.window_starts, dtype=np.int32)
            for p in range(self.num_patches):
                p_starts = window_starts_arr + p * self.patch_stride
                p_ends = p_starts + self.patch_length
                sums = cumsum_labels[p_ends] - cumsum_labels[p_starts]
                fracs = sums / self.patch_length
                self.precomputed_patch_labels[:, p] = (fracs >= self.overlap_threshold).astype(np.float32)

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int):
        s = self.window_starts[idx]
        ctx_e = s + self.context_length
        fut_e = ctx_e + self.prediction_length

        past_values   = torch.from_numpy(self.data[s    : ctx_e])   # (L, C)
        future_values = torch.from_numpy(self.data[ctx_e : fut_e])  # (T, C)
        patch_labels  = torch.from_numpy(self.precomputed_patch_labels[idx])     # (P,)

        return past_values, future_values, patch_labels

    def icme_patch_ratio(self) -> float:
        """Fraction of ICME patches across all valid windows (for pos_weight)."""
        pos = int(self.precomputed_patch_labels.sum())
        total = self.precomputed_patch_labels.size
        neg = total - pos
        ratio = neg / max(pos, 1)
        print(f"[dataset] ICME patch ratio  pos={pos}  neg={neg}  pos_weight≈{ratio:.1f}")
        return float(ratio)


class VectorizedGPULoader:
    """
    Direct GPU batch generator using PyTorch multi-dimensional matrix index broadcasting.

    Bypasses standard DataLoader thread loop serialization to deliver 0.000s batch loading.
    """
    def __init__(self, dataset: OmniPatchDataset, cfg: dict, shuffle: bool = True, device="cuda"):
        self.device = device
        self.batch_size = cfg["batch_size"]
        self.context_length = cfg["context_length"]
        self.prediction_length = cfg["prediction_length"]
        self.shuffle = shuffle

        # Move underlying arrays to GPU/Target device upfront (size ~150MB total)
        self.data = torch.from_numpy(dataset.data).float().to(device)
        self.window_starts = torch.tensor(dataset.window_starts, dtype=torch.long, device=device)
        self.patch_labels = torch.from_numpy(dataset.precomputed_patch_labels).float().to(device)

        # Index offsets for vectorization
        self.past_offsets = torch.arange(self.context_length, device=device)
        self.future_offsets = torch.arange(self.context_length, self.context_length + self.prediction_length, device=device)

        self.num_samples = len(self.window_starts)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.device)
            starts = self.window_starts[perm]
            labels = self.patch_labels[perm]
        else:
            starts = self.window_starts
            labels = self.patch_labels

        for i in range(0, self.num_samples, self.batch_size):
            b_starts = starts[i : i + self.batch_size]
            b_labels = labels[i : i + self.batch_size]

            # Broadcast mapping to create batch indices on the GPU
            past_idx = b_starts.unsqueeze(1) + self.past_offsets
            future_idx = b_starts.unsqueeze(1) + self.future_offsets

            past_values = self.data[past_idx]
            future_values = self.data[future_idx]

            yield past_values, future_values, b_labels

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def make_datasets(
    omni_df: pd.DataFrame,
    cr_icmes: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    scaler: Optional[RobustScaler] = None,
) -> tuple[OmniPatchDataset, OmniPatchDataset, OmniPatchDataset, RobustScaler]:
    """
    Build train / val / test OmniPatchDatasets with temporally safe normalisation.

    The RobustScaler is fit exclusively on the training split if not provided.
    """
    train_end = pd.to_datetime(cfg["train_end"])
    val_end   = pd.to_datetime(cfg["val_end"])

    train_df = omni_df.loc[: str(cfg["train_end"])][feature_cols].copy()
    val_df   = omni_df.loc[str(cfg["train_end"]) : str(cfg["val_end"])][feature_cols].copy()
    test_df  = omni_df.loc[str(cfg["val_end"]) :][feature_cols].copy()

    # Interpolate small gaps (keep same convention as OmniWindowDataset)
    for df in (train_df, val_df, test_df):
        df.interpolate(limit=6, limit_direction="both", inplace=True)
        df.fillna(0.0, inplace=True)

    if scaler is None:
        scaler = RobustScaler()
        train_data = scaler.fit_transform(train_df.values).astype(np.float32)
    else:
        train_data = scaler.transform(train_df.values).astype(np.float32)

    val_data   = scaler.transform(val_df.values).astype(np.float32)
    test_data  = scaler.transform(test_df.values).astype(np.float32)

    icme_intervals = build_icme_intervals(cr_icmes)

    kwargs = dict(
        icme_intervals=icme_intervals,
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
        patch_length=cfg["patch_length"],
        patch_stride=cfg["patch_stride"],
        overlap_threshold=cfg["overlap_threshold"],
        window_stride=cfg["window_stride"],
    )

    train_ds = OmniPatchDataset(train_data, train_df.index, **kwargs)
    val_ds   = OmniPatchDataset(val_data,   val_df.index,   **kwargs)
    test_ds  = OmniPatchDataset(test_data,  test_df.index,  **kwargs)

    print(
        f"[dataset] train={len(train_ds):,}  val={len(val_ds):,}  "
        f"test={len(test_ds):,} windows | "
        f"num_patches={train_ds.num_patches} | "
        f"feature_cols={feature_cols}"
    )
    return train_ds, val_ds, test_ds, scaler