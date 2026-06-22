"""
xgboost_evaluator.py
---------------------
Downstream classifier evaluation script. 

Extracts features from pre-trained backbone embeddings and fits
XGBoost and Random Forest. Also runs a raw data baseline to assess representation quality.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import PatchTSMixerConfig

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed; XGBoost classifiers will be skipped.")

from .loaders import (
    read_omni_cache,
    get_cr_icme_dataframe,
    engineer_features,
    make_datasets,
    VectorizedGPULoader,
    OmniPatchDataset,
)
from .tsmixer_backbone import PatchTSMixerICMEBackbone


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

    # Use VectorizedGPULoader for accelerated extraction
    loader = VectorizedGPULoader(dataset, cfg, shuffle=False, device=device)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    with torch.no_grad():
        for past, _, p_labels in loader:
            p_labels = p_labels.cpu().numpy()                          # (B, P)

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
    device = cfg["device"]
    # Use VectorizedGPULoader for baseline extraction speedups
    loader = VectorizedGPULoader(dataset, cfg, shuffle=False, device=device)
    
    PL = cfg["patch_length"]
    PS = cfg["patch_stride"]
    P  = dataset.num_patches

    for past, _, p_labels in loader:
        past     = past.cpu().numpy()        # (B, L, C)
        p_labels = p_labels.cpu().numpy()   # (B, P)
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


def build_backbone_from_config(cfg: dict, checkpoint_state: dict) -> PatchTSMixerICMEBackbone:
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
        use_anomaly_head=cfg["use_anomaly_head"],
        head_dropout=cfg["head_dropout"],
        forecast_loss_weight=cfg["forecast_loss_weight"],
        anomaly_loss_weight=cfg["anomaly_loss_weight"],
    )
    
    # Strip torch.compile prefix variations if found
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint_state.items()}
    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream classifiers on PatchTSMixer representations")
    parser.add_argument("--checkpoint", type=str, default="results/backbone_final.pt",
                        help="Path to saved backbone checkpoint package")
    parser.add_argument("--level", type=str, default=None, choices=["patch", "window"],
                        help="Force classification level (overrides config if specified)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint package discovered at {checkpoint_path}")

    print(f"[downstream] Loading pre-trained package from {checkpoint_path}")
    package = torch.load(checkpoint_path, map_location=device)

    # Reconstruct configuration from saved training run
    cfg = package["cfg"]
    feature_cols = package["feature_cols"]
    scaler = package["scaler"]
    state_dict = package["state_dict"]

    cfg["device"] = device
    if args.level is not None:
        cfg["classification_level"] = args.level

    # Define classification models parameters (RF & XGBoost) inside the reconstructed config
    cfg["latent_pool"] = "mean"      # "mean", "max", "flatten"
    cfg["xgb_params"] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }
    cfg["rf_params"] = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced",   # handles ICME imbalance
        "n_jobs": -1,
        "random_state": 42,
    }

    omni_full = read_omni_cache(Path(cfg["cache_path"]))
    omni_df   = omni_full.loc[str(cfg["omni_start"]) : str(cfg["omni_end"])].copy()
    cr_icmes  = get_cr_icme_dataframe(cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"])

    omni_df = engineer_features(omni_df, cfg)

    # Build datasets utilizing the loaded scaler
    train_ds, val_ds, test_ds, _ = make_datasets(
        omni_df, cr_icmes, feature_cols, cfg, scaler=scaler
    )

    model = build_backbone_from_config(cfg, state_dict)
    model = model.to(device)

    level = cfg["classification_level"]
    print(f"\n[downstream] Extracting {level}-level latents...")

    X_tr_lat, y_tr = extract_features(model, train_ds, cfg, level=level)
    X_va_lat, y_va = extract_features(model, val_ds,   cfg, level=level)
    X_te_lat, y_te = extract_features(model, test_ds,  cfg, level=level)

    # Combine train+val for downstream fitting
    X_tr_all = np.concatenate([X_tr_lat, X_va_lat])
    y_tr_all = np.concatenate([y_tr,    y_va])

    print(
        f"[downstream] Latent dimensions: train={X_tr_all.shape}  test={X_te_lat.shape}  "
        f"ICME Fraction (Test Set)={y_te.mean():.3f}"
    )

    # Raw features baseline setup
    X_tr_raw, y_tr_raw = extract_raw_features(train_ds, cfg, level=level)
    X_va_raw, y_va_raw = extract_raw_features(val_ds,   cfg, level=level)
    X_te_raw, y_te_raw = extract_raw_features(test_ds,  cfg, level=level)

    X_tr_raw_all = np.concatenate([X_tr_raw, X_va_raw])
    y_tr_raw_all  = np.concatenate([y_tr_raw, y_va_raw])

    print("\n[downstream] Fitting classifiers...")
    rf_lat  = fit_rf(X_tr_all,     y_tr_all,     cfg)
    rf_raw  = fit_rf(X_tr_raw_all, y_tr_raw_all, cfg)
    xgb_lat = fit_xgb(X_tr_all,     y_tr_all,     cfg)
    xgb_raw = fit_xgb(X_tr_raw_all, y_tr_raw_all, cfg)

    print("\n" + "=" * 60)
    print(f"  TEST RESULTS  ({level}-level classification)")
    print("=" * 60)

    results = {}
    results["RF on latent"]  = evaluate_classifier(rf_lat,  X_te_lat, y_te,     "RF on latent representation")
    results["RF on raw"]     = evaluate_classifier(rf_raw,  X_te_raw, y_te_raw, "RF on raw data (baseline)")
    results["XGB on latent"] = evaluate_classifier(xgb_lat, X_te_lat, y_te,     "XGB on latent representation")
    results["XGB on raw"]    = evaluate_classifier(xgb_raw, X_te_raw, y_te_raw, "XGB on raw data (baseline)")

    print("\n── F1 summary ──────────────────────────────────────────")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<30}  F1={metrics['f1']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    results_df = pd.DataFrame(results).T
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"f1_comparison_{level}.csv"
    results_df.to_csv(out_path)
    print(f"\n[results] Export completed successfully. Saved to: {out_path}")


if __name__ == "__main__":
    main()