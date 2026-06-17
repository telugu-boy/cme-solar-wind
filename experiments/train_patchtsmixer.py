"""
train_patchtsmixer.py
---------------------
End-to-end training script for PatchTSMixer-based ICME detection.

Pipeline
--------
1. Load OMNI Parquet cache + C&R ICME catalog
2. Engineer heliophysics features (clock/cone angles, RMS B, Carrington wave)
3. Temporal train / val / test split (no data leakage)
4. Normalise with RobustScaler fitted on training data only
5. Build OmniPatchDataset (context + future windows, per-patch binary labels)
6. Pretrain PatchTSMixerICMEBackbone with:
      – Forecast head (always on)
      – Anomaly head  (optional, patch-level BCE)
7. Extract latent representations from frozen backbone
8. Fit XGBoost and RandomForest on latent features  ← main downstream model
9. Fit XGBoost and RandomForest on raw windowed data ← baseline
10. Compare patch-level F1 scores

Experiment parameters follow the PatchTSMixer experiment spec
(see experiment spreadsheet, Image 2).

Usage
-----
    python train_patchtsmixer.py

Adjust the CFG dict below for ablations.
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import date as Date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.ensemble import RandomForestClassifier
from transformers import PatchTSMixerConfig

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed; XGBoost classifiers will be skipped.")

from experiments.tsmixer_backbone import PatchTSMixerICMEBackbone, num_patches_from_config

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

CFG: dict[str, Any] = {
    # ── Data ──────────────────────────────────────────────────────────────
    "cache_path":          "data/omni_cache_5min_full.parquet",
    "icme_catalog_path":   "data/icme_catalog.csv",
    "omni_start":          Date(1996, 5, 1),
    "omni_end":            Date(2020, 1, 1),

    # ── Temporal splits (by calendar date, no shuffling) ──────────────────
    # Train 1.5 solar cycles
    "train_end":   Date(2013, 1, 1),
    "val_end":     Date(2015, 7, 1),
    # test: val_end → omni_end

    # ── Feature selection ─────────────────────────────────────────────────
    # Core raw features (always included)
    "raw_feature_cols": ["F", "BX_GSE", "BY_GSE", "BZ_GSE",
                         "flow_speed", "proton_density"],

    # Engineered features (set to False to ablate)
    "use_sin_clock":         True,   # sin(atan2(Bz, By))
    "use_cos_clock":         True,   # cos(atan2(Bz, By))
    "use_cos_cone":          True,   # Bx / |B|
    "use_sin_cone":          False,  # optional (orange in spreadsheet)
    "use_rms_B":             True,   # rolling RMS of |B|
    "use_f10_7":             False,  # f10.7 solar index (column name: f10.7_index)
    "use_carrington_wave":   True,   # sin/cos of 27.27-day rotation
    "rms_window":            12,     # 1 hour at 5-min resolution

    # Start univariate with |B| only?  Override feature cols if True.
    "univariate_test":       False,

    # ── Model architecture ────────────────────────────────────────────────
    # Context window: 32h at 5-min resolution = 32 * 12 = 384 timesteps
    # Ablation options from spreadsheet: 32h, 36h, 48h
    "context_length":   384,           # 32h (try 432=36h, 576=48h)
    "prediction_length": 96,           # 8h ahead
    "patch_length":      16,           # 80 min per patch → 24 patches per 32h window
    "patch_stride":      16,           # non-overlapping patches
    "d_model":           64,
    "num_layers":        6,
    "expansion_factor":  2,
    "dropout":           0.2,
    "head_dropout":      0.1,
    "mode":              "mix_channel",  # "common_channel" or "mix_channel"
    "gated_attn":        True,
    "self_attn":         False,          # tiny self-attn across patches (optional)

    # ── Pretraining heads ──────────────────────────────────────────────────
    "use_anomaly_head":       True,
    "forecast_loss_weight":   1.0,
    "anomaly_loss_weight":    2.0,    # upweight anomaly to match experiment focus

    # ── Patch labelling ────────────────────────────────────────────────────
    "overlap_threshold": 0.10,        # >= 10% of patch timesteps in ICME interval

    # ── Training ───────────────────────────────────────────────────────────
    "pretrain_epochs":   50,
    "batch_size":        64,
    "learning_rate":     1e-3,
    "weight_decay":      1e-4,
    "lr_patience":       5,           # ReduceLROnPlateau patience
    "early_stop_patience": 10,
    "num_workers":       4,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",

    # ── Dataset sliding window ─────────────────────────────────────────────
    "window_stride":     12,          # 1-hour stride for window sampling

    # ── Downstream classifiers ─────────────────────────────────────────────
    "latent_pool":       "mean",      # "mean", "max", "flatten"
    "classification_level": "patch",  # "patch" or "window"

    # XGBoost
    "xgb_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    },

    # Random Forest
    "rf_params": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced",   # handles ICME imbalance
        "n_jobs": -1,
        "random_state": 42,
    },

    # ── Output ─────────────────────────────────────────────────────────────
    "checkpoint_dir": "checkpoints",
    "results_dir":    "results",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (reuse project utilities)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

CARRINGTON_PERIOD_DAYS = 27.2753   # synodic Carrington rotation period

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    def _patch_labels(self, window_start: int) -> np.ndarray:
        """Binary ICME label for each patch in the context window."""
        labels = np.zeros(self.num_patches, dtype=np.float32)
        for p in range(self.num_patches):
            s = window_start + p * self.patch_stride
            e = s + self.patch_length
            frac = self.timestep_labels[s:e].mean()
            if frac >= self.overlap_threshold:
                labels[p] = 1.0
        return labels

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int):
        s = self.window_starts[idx]
        ctx_e = s + self.context_length
        fut_e = ctx_e + self.prediction_length

        past_values   = torch.from_numpy(self.data[s    : ctx_e])   # (L, C)
        future_values = torch.from_numpy(self.data[ctx_e : fut_e])  # (T, C)
        patch_labels  = torch.from_numpy(self._patch_labels(s))     # (P,)

        return past_values, future_values, patch_labels

    # ─────────────────────────────────────────────────────────────────────
    def icme_patch_ratio(self) -> float:
        """Fraction of ICME patches across all valid windows (for pos_weight)."""
        total = pos = 0
        for s in self.window_starts:
            labels = self._patch_labels(s)
            total += len(labels)
            pos   += int(labels.sum())
        neg = total - pos
        ratio = neg / max(pos, 1)
        print(f"[dataset] ICME patch ratio  pos={pos}  neg={neg}  pos_weight≈{ratio:.1f}")
        return float(ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Build datasets helper
# ─────────────────────────────────────────────────────────────────────────────

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


def make_datasets(
    omni_df: pd.DataFrame,
    cr_icmes: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[OmniPatchDataset, OmniPatchDataset, OmniPatchDataset, RobustScaler]:
    """
    Build train / val / test OmniPatchDatasets with temporally safe normalisation.

    The RobustScaler is fit exclusively on the training split.
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

    # Fit scaler on training data only
    scaler = RobustScaler()
    train_data = scaler.fit_transform(train_df.values).astype(np.float32)
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


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(
    model: PatchTSMixerICMEBackbone,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    train: bool = True,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {
        "total": 0.0, "forecast": 0.0, "anomaly": 0.0
    }
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for past, future, p_labels in loader:
            past     = past.to(device)
            future   = future.to(device)
            p_labels = p_labels.to(device)

            out = model(past, future_values=future, patch_labels=p_labels)

            if train:
                optimizer.zero_grad()
                out.total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = past.size(0)
            totals["total"]    += out.total_loss.item()    * bs
            if out.forecast_loss is not None:
                totals["forecast"] += out.forecast_loss.item() * bs
            if out.anomaly_loss is not None:
                totals["anomaly"]  += out.anomaly_loss.item()  * bs
            n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def pretrain(
    model: PatchTSMixerICMEBackbone,
    train_ds: OmniPatchDataset,
    val_ds: OmniPatchDataset,
    cfg: dict,
) -> str:
    """
    Pretrain the backbone with forecast + anomaly heads.

    Returns the path to the best checkpoint.
    """
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(ckpt_dir / "patchtsmixer_icme_best.pt")

    device = cfg["device"]
    model  = model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"], verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        t0 = time.time()
        tr = run_one_epoch(model, train_loader, optimizer, device, train=True)
        va = run_one_epoch(model, val_loader,   None,      device, train=False)
        scheduler.step(va["total"])
        elapsed = time.time() - t0

        loss_parts = (
            f"total={tr['total']:.4f}  forecast={tr['forecast']:.4f}  "
            f"anomaly={tr['anomaly']:.4f}"
        )
        print(
            f"[pretrain] epoch {epoch:03d}/{cfg['pretrain_epochs']}  "
            f"train [{loss_parts}]  val_total={va['total']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.0f}s)"
        )

        if va["total"] < best_val_loss:
            best_val_loss = va["total"]
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"           ✓ saved checkpoint  val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"[pretrain] early stopping at epoch {epoch}.")
                break

    print(f"[pretrain] best val_loss={best_val_loss:.4f} — loading checkpoint.")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for downstream classifiers
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    model: PatchTSMixerICMEBackbone,
    dataset: OmniPatchDataset,
    cfg: dict,
    level: str = "patch",    # "patch" or "window"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract latent features and labels for downstream classification.

    level="patch"
        X shape : (n_windows * num_patches, d_model)   — each patch as a sample
        y shape : (n_windows * num_patches,)            — binary per patch

    level="window"
        X shape : (n_windows, C * d_model)              — pooled over patches
        y shape : (n_windows,)                           — 1 if any ICME patch
    """
    device = cfg["device"]
    model.eval()
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    with torch.no_grad():
        for past, _, p_labels in loader:
            past     = past.to(device)
            p_labels = p_labels.numpy()                          # (B, P)

            if level == "patch":
                # (B, P, D) per-patch embeddings, channel-pooled
                h = model.get_patch_latents(past)                # (B, P, D)
                B, P, D = h.shape
                X_list.append(h.cpu().numpy().reshape(B * P, D))
                y_list.append(p_labels.reshape(B * P))

            else:  # window-level
                h = model.get_latent_representations(            # (B, C*D)
                    past, pool=cfg["latent_pool"]
                )
                X_list.append(h.cpu().numpy())
                # Window is ICME if any patch is ICME
                y_list.append((p_labels.max(axis=1) > 0.5).astype(np.float32))

    return np.concatenate(X_list), np.concatenate(y_list)


def extract_raw_features(
    dataset: OmniPatchDataset,
    cfg: dict,
    level: str = "patch",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline: XGBoost / RF on the raw windowed data (no backbone).

    level="patch"
        X shape : (n_windows * num_patches, patch_length * C)
        y shape : (n_windows * num_patches,)

    level="window"
        X shape : (n_windows, context_length * C)
        y shape : (n_windows,)
    """
    X_list, y_list = [], []
    loader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=cfg["num_workers"]
    )
    PL = cfg["patch_length"]
    PS = cfg["patch_stride"]
    P  = dataset.num_patches

    for past, _, p_labels in loader:
        past     = past.numpy()        # (B, L, C)
        p_labels = p_labels.numpy()   # (B, P)
        B, L, C  = past.shape

        if level == "patch":
            patches = []
            for p in range(P):
                s = p * PS
                e = s + PL
                patches.append(past[:, s:e, :].reshape(B, PL * C))
            # stack → (B, P, PL*C) → (B*P, PL*C)
            X = np.stack(patches, axis=1).reshape(B * P, PL * C)
            y = p_labels.reshape(B * P)
        else:
            X = past.reshape(B, L * C)
            y = (p_labels.max(axis=1) > 0.5).astype(np.float32)

        X_list.append(X)
        y_list.append(y)

    return np.concatenate(X_list), np.concatenate(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# Downstream classifiers
# ─────────────────────────────────────────────────────────────────────────────

def fit_rf(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> RandomForestClassifier:
    rf = RandomForestClassifier(**cfg["rf_params"])
    rf.fit(X_train, y_train.astype(int))
    return rf


def fit_xgb(X_train: np.ndarray, y_train: np.ndarray, cfg: dict):
    if not HAS_XGBOOST:
        return None
    # Compute scale_pos_weight for XGBoost imbalance handling
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = dict(cfg["xgb_params"])
    params["scale_pos_weight"] = scale_pos_weight
    xgb = XGBClassifier(**params)
    xgb.fit(
        X_train, y_train.astype(int),
        eval_set=[(X_train, y_train.astype(int))],
        verbose=False,
    )
    return xgb


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict[str, float]:
    if clf is None:
        return {}
    y_pred = clf.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test.astype(int), y_pred, average="binary", zero_division=0
    )
    print(f"\n── {label} ──")
    print(classification_report(
        y_test.astype(int), y_pred,
        target_names=["ambient", "ICME"],
        zero_division=0,
    ))
    return {"precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone(cfg: dict, pos_weight: Optional[float] = None) -> PatchTSMixerICMEBackbone:
    config = PatchTSMixerConfig(
        context_length=cfg["context_length"],
        patch_length=cfg["patch_length"],
        patch_stride=cfg["patch_stride"],
        num_input_channels=cfg["_n_features"],    # set in main
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        expansion_factor=cfg["expansion_factor"],
        dropout=cfg["dropout"],
        mode=cfg["mode"],
        gated_attn=cfg["gated_attn"],
        self_attn=cfg["self_attn"],
        prediction_length=cfg["prediction_length"],
    )
    model = PatchTSMixerICMEBackbone(
        config,
        use_anomaly_head=cfg["use_anomaly_head"],
        head_dropout=cfg["head_dropout"],
        forecast_loss_weight=cfg["forecast_loss_weight"],
        anomaly_loss_weight=cfg["anomaly_loss_weight"],
        pos_weight=pos_weight,
    )
    model.summary()
    return model


def main(cfg: dict = CFG) -> None:
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    omni_full = read_omni_cache(Path(cfg["cache_path"]))
    omni_df   = omni_full.loc[
        str(cfg["omni_start"]) : str(cfg["omni_end"])
    ].copy()

    cr_icmes  = get_cr_icme_dataframe(
        cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"]
    )
    print(f"[data] OMNI shape={omni_df.shape}  ICME events={len(cr_icmes)}")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    omni_df = engineer_features(omni_df, cfg)
    feature_cols = select_feature_cols(omni_df, cfg)
    cfg["_n_features"] = len(feature_cols)
    print(f"[features] {feature_cols}")

    # ── 3. Datasets ──────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, scaler = make_datasets(
        omni_df, cr_icmes, feature_cols, cfg
    )

    # Compute pos_weight from training set (ICME patch imbalance)
    pos_weight = train_ds.icme_patch_ratio()

    # ── 4. Build & pretrain backbone ─────────────────────────────────────────
    model = build_backbone(cfg, pos_weight=pos_weight)
    best_ckpt = pretrain(model, train_ds, val_ds, cfg)

    # ── 5. Extract latent representations ───────────────────────────────────
    level = cfg["classification_level"]
    print(f"\n[downstream] Extracting {level}-level latents…")

    X_tr_lat, y_tr = extract_features(model, train_ds, cfg, level=level)
    X_va_lat, y_va = extract_features(model, val_ds,   cfg, level=level)
    X_te_lat, y_te = extract_features(model, test_ds,  cfg, level=level)

    # Combine train+val for downstream fitting
    X_tr_all = np.concatenate([X_tr_lat, X_va_lat])
    y_tr_all  = np.concatenate([y_tr,    y_va])

    print(
        f"[downstream] latent train={X_tr_all.shape}  test={X_te_lat.shape}  "
        f"ICME frac (test)={y_te.mean():.3f}"
    )

    # ── 6. Baseline raw features ─────────────────────────────────────────────
    X_tr_raw, y_tr_raw = extract_raw_features(train_ds, cfg, level=level)
    X_va_raw, y_va_raw = extract_raw_features(val_ds,   cfg, level=level)
    X_te_raw, y_te_raw = extract_raw_features(test_ds,  cfg, level=level)

    X_tr_raw_all = np.concatenate([X_tr_raw, X_va_raw])
    y_tr_raw_all  = np.concatenate([y_tr_raw, y_va_raw])

    # ── 7. Fit classifiers ───────────────────────────────────────────────────
    print("\n[downstream] Training classifiers…")

    rf_lat  = fit_rf(X_tr_all,     y_tr_all,     cfg)
    rf_raw  = fit_rf(X_tr_raw_all, y_tr_raw_all, cfg)
    xgb_lat = fit_xgb(X_tr_all,     y_tr_all,     cfg)
    xgb_raw = fit_xgb(X_tr_raw_all, y_tr_raw_all, cfg)

    # ── 8. Evaluate and compare ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST RESULTS  ({level}-level classification)")
    print("=" * 60)

    results = {}
    results["RF on latent"]  = evaluate_classifier(rf_lat,  X_te_lat, y_te,     "RF  on latent representation")
    results["RF on raw"]     = evaluate_classifier(rf_raw,  X_te_raw, y_te_raw, "RF  on raw data (baseline)")
    results["XGB on latent"] = evaluate_classifier(xgb_lat, X_te_lat, y_te,     "XGB on latent representation")
    results["XGB on raw"]    = evaluate_classifier(xgb_raw, X_te_raw, y_te_raw, "XGB on raw data (baseline)")

    print("\n── F1 summary ──────────────────────────────────────────")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<30}  F1={metrics['f1']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    # ── 9. Save artefacts ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T
    out_path = results_dir / "f1_comparison.csv"
    results_df.to_csv(out_path)
    print(f"\n[results] Saved to {out_path}")

    # Save trained backbone
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "feature_cols": feature_cols,
            "scaler": scaler,
        },
        results_dir / "backbone_final.pt",
    )
    print(f"[results] Backbone saved to {results_dir / 'backbone_final.pt'}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTSMixer ICME pretraining + downstream")
    parser.add_argument("--context_length",  type=int,   default=CFG["context_length"])
    parser.add_argument("--patch_length",    type=int,   default=CFG["patch_length"])
    parser.add_argument("--patch_stride",    type=int,   default=CFG["patch_stride"])
    parser.add_argument("--epochs",          type=int,   default=CFG["pretrain_epochs"])
    parser.add_argument("--batch_size",      type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",              type=float, default=CFG["learning_rate"])
    parser.add_argument("--d_model",         type=int,   default=CFG["d_model"])
    parser.add_argument("--num_layers",      type=int,   default=CFG["num_layers"])
    parser.add_argument("--no_anomaly_head", action="store_true")
    parser.add_argument("--univariate",      action="store_true")
    parser.add_argument("--level",           type=str,   default=CFG["classification_level"],
                        choices=["patch", "window"])
    args = parser.parse_args()

    CFG["context_length"]       = args.context_length
    CFG["patch_length"]         = args.patch_length
    CFG["patch_stride"]         = args.patch_stride
    CFG["pretrain_epochs"]      = args.epochs
    CFG["batch_size"]           = args.batch_size
    CFG["learning_rate"]        = args.lr
    CFG["d_model"]              = args.d_model
    CFG["num_layers"]           = args.num_layers
    CFG["use_anomaly_head"]     = not args.no_anomaly_head
    CFG["univariate_test"]      = args.univariate
    CFG["classification_level"] = args.level

    main(CFG)