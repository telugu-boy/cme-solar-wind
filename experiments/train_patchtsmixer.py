"""
train_patchtsmixer.py
---------------------
Pretraining script for PatchTSMixer-based ICME detection.

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
7. Save model package for independent downstream evaluation
"""

from __future__ import annotations

import argparse
import time
from datetime import date as Date
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import PatchTSMixerConfig

from .loaders import (
    read_omni_cache,
    get_cr_icme_dataframe,
    engineer_features,
    select_feature_cols,
    make_datasets,
    VectorizedGPULoader,
    OmniPatchDataset,
)
from .tsmixer_backbone import PatchTSMixerICMEBackbone

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
    "raw_feature_cols": ["F", # "BX_GSE", "BY_GSE", "BZ_GSE",
                         "flow_speed", "proton_density"],

    # Engineered features (set to False to ablate)
    "use_sin_clock":         True,   # sin(atan2(Bz, By))
    "use_cos_clock":         True,   # cos(atan2(Bz, By))
    "use_cos_cone":          True,   # Bx / |B|
    "use_sin_cone":          False,  # optional (orange in spreadsheet)
    "use_rms_B":             True,   # rolling RMS of |B|
    "use_f10_7":             True,   # f10.7 solar index (column name: f10.7_index)
    "use_carrington_wave":   True,   # sin/cos of 27.27-day rotation
    "rms_window":            12,     # 1 hour at 5-min resolution

    # Start univariate with |B| only?  Override feature cols if True.
    "univariate_test":       False,

    # ── Model architecture ────────────────────────────────────────────────
    # Context window: 32h at 5-min resolution = 32 * 12 = 384 timesteps
    # Ablation options from spreadsheet: 32h, 36h, 48h
    "context_length":    192,          # 16h window (192 * 5m = 960m = 16h)
    "prediction_length": 48,           # 4h ahead
    "patch_length":      16,           # 80 min per patch → 12 patches per 16h window
    "patch_stride":      16,           # non-overlapping patches
    "d_model":           64,
    "num_layers":        6,
    "expansion_factor":  2,
    "dropout":           0.4,
    "head_dropout":      0.2,
    "mode":              "mix_channel",  # "common_channel" or "mix_channel"
    "gated_attn":        True,
    "self_attn":         True,           # tiny self-attn across patches (optional)

    # ── Pretraining heads ──────────────────────────────────────────────────
    "use_anomaly_head":       True,
    "forecast_loss_weight":   1.5,
    "anomaly_loss_weight":    3.0,    # upweight anomaly to match experiment focus
    # so if anomaly_loss_weight is 3.0 this may be 1.0 or 1.5.

    # ── Patch labelling ────────────────────────────────────────────────────
    "overlap_threshold": 0.10,        # >= 10% of patch timesteps in ICME interval

    # ── Training ───────────────────────────────────────────────────────────
    "pretrain_epochs":   50,
    "batch_size":        512,
    "learning_rate":     3e-3,
    "weight_decay":      1e-4,
    "lr_patience":       5,           # ReduceLROnPlateau patience
    "early_stop_patience": 10,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",

    # ── Dataset sliding window ─────────────────────────────────────────────
    "window_stride":     24,          # 2-hour stride for window sampling


    # ── Output ─────────────────────────────────────────────────────────────
    "checkpoint_dir": "checkpoints",
    "results_dir":    "results",
    "model_name":     None,
}


def build_backbone(cfg: dict, pos_weight: Optional[float] = None) -> PatchTSMixerICMEBackbone:
    config = PatchTSMixerConfig(
        context_length=cfg["context_length"],
        patch_length=cfg["patch_length"],
        patch_stride=cfg["patch_stride"],
        num_input_channels=cfg["_n_features"],
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
        head_dropout=cfg["head_dropout"],
        forecast_loss_weight=cfg["forecast_loss_weight"],
        anomaly_loss_weight=cfg["anomaly_loss_weight"],
        pos_weight=pos_weight,
    )
    model.summary()
    return model


def run_one_epoch(
    model: PatchTSMixerICMEBackbone,
    loader: VectorizedGPULoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    train: bool = True,
    grad_scaler = None,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {"total": 0.0, "forecast": 0.0, "anomaly": 0.0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for past, future, p_labels in loader:
            if train:
                optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(past, future_values=future, patch_labels=p_labels)

            if train:
                grad_scaler.scale(out.total_loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            bs = past.size(0)
            totals["total"] += out.total_loss.item() * bs
            if out.forecast_loss is not None:
                totals["forecast"] += out.forecast_loss.item() * bs
            if out.anomaly_loss is not None:
                totals["anomaly"] += out.anomaly_loss.item() * bs

            n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def pretrain(
    model: PatchTSMixerICMEBackbone,
    train_ds: OmniPatchDataset,
    val_ds: OmniPatchDataset,
    cfg: dict,
) -> str:
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = cfg.get("checkpoint_name", "patchtsmixer_icme_best.pt")
    best_ckpt = str(ckpt_dir / ckpt_name)

    device = cfg["device"]
    model  = model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    try:
        model = torch.compile(model)
        print("[pretrain] torch.compile enabled")
    except Exception:
        print("[pretrain] torch.compile unavailable, continuing without")

    train_loader = VectorizedGPULoader(train_ds, cfg, shuffle=True, device=device)
    val_loader   = VectorizedGPULoader(val_ds,   cfg, shuffle=False, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"]
    )

    last_lr = float('inf')
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        t0 = time.time()
        tr = run_one_epoch(model, train_loader, optimizer, device, train=True, grad_scaler=scaler)
        va = run_one_epoch(model, val_loader,   None,      device, train=False, grad_scaler=None)
        scheduler.step(va["total"])
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != last_lr:
            print(f"[lr] Learning rate reduced from {last_lr} to {current_lr}")
            last_lr = current_lr

        loss_parts = (
            f"total={tr['total']:.4f}  "
            f"forecast(x{model.forecast_loss_weight:.1f})={tr['forecast']:.4f}  "
            f"anomaly(x{model.anomaly_loss_weight:.1f})={tr['anomaly']:.4f}"
        )
        print(
            f"[pretrain] epoch {epoch:03d}/{cfg['pretrain_epochs']}  "
            f"train [{loss_parts}]  val_total={va['total']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"({elapsed:.0f}s)"
        )

        if va["total"] < best_val_loss:
            best_val_loss = va["total"]
            patience_counter = 0
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), best_ckpt)
            print(f"           * saved checkpoint  val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"[pretrain] early stopping at epoch {epoch}.")
                break

    print(f"[pretrain] best val_loss={best_val_loss:.4f} - loading checkpoint.")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return best_ckpt


def main(cfg: dict = CFG) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    omni_full = read_omni_cache(Path(cfg["cache_path"]))
    omni_df   = omni_full.loc[str(cfg["omni_start"]) : str(cfg["omni_end"])].copy()

    cr_icmes  = get_cr_icme_dataframe(cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"])
    print(f"[data] OMNI shape={omni_df.shape}  ICME events={len(cr_icmes)}")

    omni_df = engineer_features(omni_df, cfg)
    feature_cols = select_feature_cols(omni_df, cfg)
    cfg["_n_features"] = len(feature_cols)
    print(f"[features] {feature_cols}")

    train_ds, val_ds, _, scaler = make_datasets(omni_df, cr_icmes, feature_cols, cfg)
    pos_weight = train_ds.icme_patch_ratio()

    model = build_backbone(cfg, pos_weight=pos_weight)
    pretrain(model, train_ds, val_ds, cfg)

    save_package = {
        "state_dict": model.state_dict(),
        "cfg": cfg,
        "feature_cols": feature_cols,
        "scaler": scaler,
    }
    final_name = cfg.get("checkpoint_name", "patchtsmixer_backbone_final.pt").replace("_best", "_final")
    torch.save(save_package, results_dir / final_name)
    print(f"[results] Backbone saved to {results_dir / final_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTSMixer pretraining script")
    parser.add_argument("--context_length",  type=int,   default=CFG["context_length"])
    parser.add_argument("--patch_length",    type=int,   default=CFG["patch_length"])
    parser.add_argument("--patch_stride",    type=int,   default=CFG["patch_stride"])
    parser.add_argument("--epochs",          type=int,   default=CFG["pretrain_epochs"])
    parser.add_argument("--batch_size",      type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",              type=float, default=CFG["learning_rate"])
    parser.add_argument("--d_model",         type=int,   default=CFG["d_model"])
    parser.add_argument("--num_layers",      type=int,   default=CFG["num_layers"])
    parser.add_argument("--forecast_loss_weight", type=float, default=CFG["forecast_loss_weight"])
    parser.add_argument("--anomaly_loss_weight", type=float, default=CFG["anomaly_loss_weight"])
    parser.add_argument("--univariate",      action="store_true")
    parser.add_argument("--checkpoint_name", type=str,   default="patchtsmixer_backbone_final.pt")
    parser.add_argument("--model_name",      type=str,   default=None, help="If provided, saves to results/full/<model_name>")
    args = parser.parse_args()

    CFG["context_length"]   = args.context_length
    CFG["patch_length"]     = args.patch_length
    CFG["patch_stride"]     = args.patch_stride
    CFG["pretrain_epochs"]  = args.epochs
    CFG["batch_size"]       = args.batch_size
    CFG["learning_rate"]    = args.lr
    CFG["d_model"]          = args.d_model
    CFG["num_layers"]       = args.num_layers
    CFG["forecast_loss_weight"] = args.forecast_loss_weight
    CFG["anomaly_loss_weight"]  = args.anomaly_loss_weight
    CFG["univariate_test"]  = args.univariate
    CFG["checkpoint_name"]  = args.checkpoint_name
    
    if args.model_name:
        CFG["model_name"] = args.model_name
        model_dir = f"results/full/{args.model_name}"
        CFG["checkpoint_dir"] = model_dir
        CFG["results_dir"] = model_dir

    main(CFG)