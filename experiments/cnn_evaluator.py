"""
cnn_evaluator.py
---------------------
Downstream classifier evaluation script using a 1D-CNN. 

Extracts features from pre-trained backbone embeddings and fits
a 1D-CNN to capture sequential patch-level dependencies (p-1, p, p+1).
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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
                # Flatten C and D to pass into CNN: (B, P, C * D)
                h_flat = h.permute(0, 2, 1, 3).reshape(B, P, C * D)
                X_list.append(h_flat.cpu().numpy())
                y_list.append(p_labels)

            else:  # window-level
                h = model.get_latent_representations(past)  # (B, C, P, D)
                B, C, P, D = h.shape
                # Flatten C and D to pass into CNN: (B, P, C * D)
                h_flat = h.permute(0, 2, 1, 3).reshape(B, P, C * D)
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
    Baseline: Raw windowed data chunked into patches for 1D-CNN baseline.
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

        patches = []
        for p in range(P):
            s = p * PS
            e = s + PL
            # Flatten raw values inside the patch: PL * C
            patches.append(past[:, s:e, :].reshape(B, PL * C))
        
        # Shape: (B, P, PL * C)
        X = np.stack(patches, axis=1)

        if level == "patch":
            y = p_labels
        else:
            y = (p_labels.max(axis=1) > 0.5).astype(np.float32)

        X_list.append(X)
        y_list.append(y)

    return np.concatenate(X_list), np.concatenate(y_list)


class DownstreamCNN(nn.Module):
    def __init__(self, in_features: int, hidden_channels: int = 128, level: str = "patch", 
                 is_latent: bool = False, C: int = 10, D: int = 64, proj_channels: int = 4):
        super().__init__()
        self.level = level
        self.is_latent = is_latent
        self.C = C
        self.D = D

        if self.is_latent:
            self.channel_proj = nn.Conv2d(self.C, proj_channels, kernel_size=1)
            cnn_in = proj_channels * self.D
        else:
            cnn_in = in_features

        # Expanded temporal receptive field (kernel_size=5) & heavy regularization
        self.net = nn.Sequential(
            nn.Conv1d(cnn_in, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_latent:
            # x is (B, P, C*D)
            B, P, CD = x.shape
            x = x.view(B, P, self.C, self.D)
            # Permute to (B, C, P, D) for Conv2d
            x = x.permute(0, 2, 1, 3)
            # Mix channels
            x = self.channel_proj(x) # (B, proj_channels, P, D)
            # Flatten mixed channels and embeddings for 1D CNN over P
            x = x.permute(0, 1, 3, 2).reshape(B, -1, P) # (B, proj_channels * D, P)
        else:
            # x is (B, P, in_features) -> (B, in_features, P)
            x = x.transpose(1, 2)
            
        logits = self.net(x) # (B, 1, P)
        
        if self.level == "patch":
            return logits.squeeze(1) # (B, P)
        else:
            # Window-level: Global max pooling across all patches
            return logits.max(dim=2)[0].squeeze(1)

def fit_cnn(X_train: np.ndarray, y_train: np.ndarray, cfg: dict, level: str = "patch", is_latent: bool = False):
    device = cfg.get("device", "cpu")
    
    # Compute scale_pos_weight
    n_pos = int(y_train.sum())
    n_neg = y_train.size - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    
    # Dampen the massive BCE pos_weight using a square root scale
    pos_weight = np.sqrt(pos_weight)
    
    # Zero-copy tensor creation to prevent 10GB+ memory duplication in Colab
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    in_features = X_train.shape[-1]
    model = DownstreamCNN(
        in_features, 
        hidden_channels=128, 
        level=level, 
        is_latent=is_latent,
        C=cfg.get("_n_features", 10),
        D=cfg.get("d_model", 64),
        proj_channels=cfg.get("proj_channels", 4)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    
    model.train()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
    
    return model


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    device: str = "cpu",
) -> dict[str, float]:
    if clf is None:
        return {}
    
    if "cuda" in str(device):
        X_test_dev = torch.tensor(X_test, device=device, dtype=torch.float32)
    else:
        X_test_dev = torch.tensor(X_test, dtype=torch.float32)

    clf.eval()
    with torch.no_grad():
        logits = clf(X_test_dev)
        y_prob = torch.sigmoid(logits)
        y_pred = (y_prob > 0.5).int()
    
    y_pred = y_pred.cpu().numpy().flatten()
    y_prob = y_prob.cpu().numpy().flatten()
    y_test_int = y_test.astype(int).flatten()

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
    print(f"AUC ROC: {roc_auc:.4f} | AUC PRC: {pr_auc:.4f} | Test LogLoss: {test_loss:.4f}")

    return {
        "precision": prec, 
        "recall": rec, 
        "f1": f1, 
        "roc_auc": roc_auc, 
        "pr_auc": pr_auc,
        "logloss": test_loss
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
    
    # Concatenate and immediately delete old arrays to prevent massive OOM memory spikes
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
    print("\n[downstream] Extracting raw features for baseline comparison...")
    X_tr_raw, y_tr_raw = extract_raw_features(train_ds, cfg, level=level)
    X_va_raw, y_va_raw = extract_raw_features(val_ds,   cfg, level=level)
    
    X_tr_raw_all = np.concatenate([X_tr_raw, X_va_raw])
    y_tr_raw_all = np.concatenate([y_tr_raw, y_va_raw])
    del X_tr_raw, X_va_raw, y_tr_raw, y_va_raw
    gc.collect()

    X_te_raw, y_te_raw = extract_raw_features(test_ds,  cfg, level=level)

    print("\n[downstream] Fitting CNN classifiers on GPU...")
    cnn_lat = fit_cnn(X_tr_all,     y_tr_all,     cfg, level=level, is_latent=True)
    cnn_raw = fit_cnn(X_tr_raw_all, y_tr_raw_all, cfg, level=level, is_latent=False)

    print("\n" + "=" * 60)
    print(f"  TEST RESULTS  ({level}-level classification)")
    print("=" * 60)

    results = {}
    results["CNN on latent"] = evaluate_classifier(cnn_lat, X_te_lat, y_te,     "CNN on latent representation", device=device)
    results["CNN on raw baseline"] = evaluate_classifier(cnn_raw, X_te_raw, y_te_raw, "CNN on raw data (baseline)", device=device)

    print("\n── Summary Metrics ──────────────────────────────────────────")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<20}  F1={metrics['f1']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                  f"ROC_AUC={metrics['roc_auc']:.4f}  PR_AUC={metrics['pr_auc']:.4f}  "
                  f"LogLoss={metrics['logloss']:.4f}")

    # results_df = pd.DataFrame(results).T
    # results_dir = Path(cfg["results_dir"])
    # results_dir.mkdir(parents=True, exist_ok=True)
    # out_path = results_dir / f"metrics_comparison_{level}.csv"
    # results_df.to_csv(out_path)
    # print(f"\n[results] Export completed successfully. Saved to: {out_path}")


if __name__ == "__main__":
    main()