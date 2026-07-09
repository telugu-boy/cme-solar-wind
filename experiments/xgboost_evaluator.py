"""
xgboost_evaluator.py
---------------------
Downstream classifier evaluation script. 

Extracts features from pre-trained backbone embeddings and fits
XGBoost. Also runs a raw data baseline to assess representation quality.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    average_precision_score,
    log_loss
)
from transformers import PatchTSMixerConfig

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    raise ImportError("xgboost is required to run this script.")

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
                # (B, C, P, D) per-patch embeddings, preserve channels
                h = model.get_latent_representations(past)  # (B, C, P, D)
                B, C, P, D = h.shape
                # Flatten C and D: (B, P, C * D) -> reshape to (B * P, C * D)
                h_flat = h.permute(0, 2, 1, 3).reshape(B * P, C * D)
                X_list.append(h_flat.cpu().numpy())
                y_list.append(p_labels.reshape(B * P))

            else:  # window-level
                h = model.get_latent_representations(past)  # (B, C, P, D)
                B, C, P, D = h.shape
                # Flatten everything across P, C, and D for a single flat feature vector
                h_flat = h.reshape(B, C * P * D)
                X_list.append(h_flat.cpu().numpy())
                # Window is ICME if any patch is ICME
                y_list.append((p_labels.max(axis=1) > 0.5).astype(np.float32))

    return np.concatenate(X_list), np.concatenate(y_list)


def extract_raw_features(
    dataset: OmniPatchDataset,
    cfg: dict,
    level: str = "patch",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline: XGBoost on the raw windowed data (no backbone).
    """
    X_list, y_list = [], []
    device = cfg["device"]
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
            X = np.stack(patches, axis=1).reshape(B * P, PL * C)
            y = p_labels.reshape(B * P)
        else:
            X = past.reshape(B, L * C)
            y = (p_labels.max(axis=1) > 0.5).astype(np.float32)

        X_list.append(X)
        y_list.append(y)

    return np.concatenate(X_list), np.concatenate(y_list)


def fit_xgb(X_train: np.ndarray, y_train: np.ndarray, cfg: dict):
    # Compute scale_pos_weight for XGBoost imbalance handling
    n_pos = int(y_train.sum())
    n_neg = y_train.size - n_pos
    # Dampen the massive pos_weight using a square root scale to improve precision
    scale_pos_weight = np.sqrt(n_neg / max(n_pos, 1))
    
    params = dict(cfg["xgb_params"])
    params["scale_pos_weight"] = scale_pos_weight
    
    xgb = XGBClassifier(**params)
    
    device = cfg.get("device", "cpu")
    if "cuda" in str(params.get("device", "")) or "cuda" in str(device):
        # Pure PyTorch CUDA tensor casting — zero-copy interface with XGBoost GPU
        X_train_dev = torch.as_tensor(X_train, device=device, dtype=torch.float32)
        y_train_dev = torch.as_tensor(y_train.astype(int), device=device)
    else:
        X_train_dev = X_train
        y_train_dev = y_train.astype(int)
    
    xgb.fit(
        X_train_dev, y_train_dev,
        eval_set=[(X_train_dev, y_train_dev)],
        verbose=False,
    )
    return xgb


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    device: str = "cpu",
) -> dict[str, float]:
    if clf is None:
        return {}
    
    if "cuda" in str(clf.get_params().get("device", "")) or "cuda" in str(device):
        X_test_dev = torch.as_tensor(X_test, device=device, dtype=torch.float32)
    else:
        X_test_dev = X_test

    # Generate both hard labels and continuous probabilities
    y_pred = clf.predict(X_test_dev)
    y_prob = clf.predict_proba(X_test_dev)[:, 1]
    
    # Unpack PyTorch CUDA tensors back to standard CPU NumPy arrays for Sklearn compliance
    if hasattr(y_pred, "cpu"): 
        y_pred = y_pred.cpu().numpy()
    if hasattr(y_prob, "cpu"): 
        y_prob = y_prob.cpu().numpy()

    # Calculate metrics
    y_test_int = y_test.astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_int, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_int, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test_int, y_prob)
    pr_auc = average_precision_score(y_test_int, y_prob)
    test_loss = log_loss(y_test_int, y_prob)

    print(f"\n── {label} ──")
    print(classification_report(
        y_test_int, y_pred,
        target_names=["ambient", "ICME"],
        zero_division=0,
    ))
    print(f"Confusion Matrix: TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
    print(f"AUC ROC: {roc_auc:.4f} | AUC PRC: {pr_auc:.4f} | Test LogLoss: {test_loss:.4f}")

    return {
        "precision": prec, 
        "recall": rec, 
        "f1": f1, 
        "roc_auc": roc_auc, 
        "pr_auc": pr_auc,
        "logloss": test_loss,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cm": cm
    }


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
        head_dropout=cfg.get("head_dropout", 0.1),
        forecast_loss_weight=cfg.get("forecast_loss_weight", 0.0),
        anomaly_loss_weight=cfg.get("anomaly_loss_weight", 1.0),
    )
    
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint_state.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream classifiers on PatchTSMixer representations")
    parser.add_argument("--checkpoint", type=str, default="results/patchtsmixer_backbone_final.pt",
                        help="Path to saved backbone checkpoint package")
    parser.add_argument("--level", type=str, default=None, choices=["patch", "window"],
                        help="Force classification level (overrides config if specified)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint package discovered at {checkpoint_path}")

    print(f"[downstream] Loading pre-trained package from {checkpoint_path}")
    package = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = package["cfg"]
    feature_cols = package["feature_cols"]
    scaler = package["scaler"]
    state_dict = package["state_dict"]

    cfg["device"] = device
    if args.level is not None:
        cfg["classification_level"] = args.level


    # XGBoost GPU Parameters Configuration
    cfg["xgb_params"] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": ["logloss", "auc", "aucpr"], 
        "device": "cuda",             
        "random_state": 42,
    }

    omni_full = read_omni_cache(Path(cfg["cache_path"]))
    omni_df   = omni_full.loc[str(cfg["omni_start"]) : str(cfg["omni_end"])].copy()
    cr_icmes  = get_cr_icme_dataframe(cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"])

    omni_df = engineer_features(omni_df, cfg)

    train_ds, val_ds, test_ds, _ = make_datasets(
        omni_df, cr_icmes, feature_cols, cfg, scaler=scaler
    )

    model = build_backbone_from_config(cfg, state_dict)
    model = model.to(device)

    level = cfg["classification_level"]
    print(f"\n[downstream] Extracting {level}-level latents...")

    X_tr_lat, y_tr = extract_features(model, train_ds, cfg, level=level)
    X_va_lat, y_va = extract_features(model, val_ds,   cfg, level=level)
    
    # Concatenate and immediately delete to prevent massive OOM memory spikes
    X_tr_all = np.concatenate([X_tr_lat, X_va_lat])
    y_tr_all = np.concatenate([y_tr,    y_va])
    del X_tr_lat, X_va_lat, y_tr, y_va
    import gc; gc.collect()

    X_te_lat, y_te = extract_features(model, test_ds,  cfg, level=level)

    print(
        f"[downstream] Latent dimensions: train={X_tr_all.shape}  test={X_te_lat.shape}  "
        f"ICME Fraction (Test Set)={y_te.mean():.3f}"
    )

    # Raw features baseline setup
    X_tr_raw, y_tr_raw = extract_raw_features(train_ds, cfg, level=level)
    X_va_raw, y_va_raw = extract_raw_features(val_ds,   cfg, level=level)
    
    X_tr_raw_all = np.concatenate([X_tr_raw, X_va_raw])
    y_tr_raw_all  = np.concatenate([y_tr_raw, y_va_raw])
    del X_tr_raw, X_va_raw, y_tr_raw, y_va_raw
    gc.collect()

    X_te_raw, y_te_raw = extract_raw_features(test_ds,  cfg, level=level)

    print("\n[downstream] Fitting XGBoost classifiers on GPU...")
    xgb_lat = fit_xgb(X_tr_all,     y_tr_all,     cfg)
    xgb_raw = fit_xgb(X_tr_raw_all, y_tr_raw_all, cfg)

    print("\n" + "=" * 60)
    print(f"  TEST RESULTS  ({level}-level classification)")
    print("=" * 60)

    results = {}
    res_lat = evaluate_classifier(xgb_lat, X_te_lat, y_te, "XGBoost on latent representation", device=device)
    results["XGBoost on latent"] = res_lat
    
    res_raw = evaluate_classifier(xgb_raw, X_te_raw, y_te_raw, "XGBoost on raw data (baseline)", device=device)
    results["XGBoost on raw baseline"] = res_raw
    
    # Visualizations
    from experiments.visualize import plot_predictions
    import os
    ckpt_name = os.path.splitext(cfg["checkpoint_name"])[0]
    
    plot_predictions(
        test_ds, 
        res_lat["y_pred"], 
        res_lat["cm"], 
        f"visualizations/{ckpt_name}_xgb_latent.png", 
        "orange", 
        "XGBoost Latent Predictions"
    )
    plot_predictions(
        test_ds, 
        res_raw["y_pred"], 
        res_raw["cm"], 
        f"visualizations/{ckpt_name}_xgb_raw.png", 
        "brown", 
        "XGBoost Raw Baseline Predictions"
    )

    print("\n── Summary Metrics ──────────────────────────────────────────")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<20}  F1={metrics['f1']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                  f"ROC_AUC={metrics['roc_auc']:.4f}  PR_AUC={metrics['pr_auc']:.4f}  "
                  f"LogLoss={metrics['logloss']:.4f}  "
                  f"CM=[TP:{metrics['cm'][1,1]} FP:{metrics['cm'][0,1]} FN:{metrics['cm'][1,0]} TN:{metrics['cm'][0,0]}]")

    # results_df = pd.DataFrame(results).T
    # results_dir = Path(cfg["results_dir"])
    # results_dir.mkdir(parents=True, exist_ok=True)
    # out_path = results_dir / f"metrics_comparison_{level}.csv"
    # results_df.to_csv(out_path)
    # print(f"\n[results] Export completed successfully. Saved to: {out_path}")


if __name__ == "__main__":
    main()