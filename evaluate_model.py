import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc
from sklearn.metrics import classification_report

# Import required functions from existing evaluators
from experiments.loaders import read_omni_cache, get_cr_icme_dataframe, engineer_features, make_datasets
from experiments.visualize import plot_predictions

from experiments.cnn_evaluator import (
    extract_features as cnn_extract_features,
    extract_raw_features as cnn_extract_raw,
    fit_cnn,
    evaluate_classifier as cnn_evaluate_classifier,
    build_backbone_from_config as cnn_build_backbone
)

from experiments.xgboost_evaluator import (
    extract_features as xgb_extract_features,
    extract_raw_features as xgb_extract_raw,
    fit_xgb,
    evaluate_classifier as xgb_evaluate_classifier,
    build_backbone_from_config as xgb_build_backbone
)

def run_evaluation(checkpoint_path: Path):
    out_dir = checkpoint_path.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading package from {checkpoint_path}")
    package = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = package["cfg"]
    feature_cols = package["feature_cols"]
    scaler = package["scaler"]
    state_dict = package["state_dict"]
    
    cfg["device"] = device
    level = cfg.get("classification_level", "patch")

    print("Loading data...")
    omni_full = read_omni_cache(Path(cfg["cache_path"]))
    omni_df   = omni_full.loc[str(cfg["omni_start"]) : str(cfg["omni_end"])].copy()
    cr_icmes  = get_cr_icme_dataframe(cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"])

    omni_df = engineer_features(omni_df, cfg)

    train_ds, val_ds, test_ds, _ = make_datasets(
        omni_df, cr_icmes, feature_cols, cfg, scaler=scaler
    )

    # Dictionary to hold the raw results
    res = {}
    
    ckpt_name = checkpoint_path.stem

    # ---------------- CNN Evaluation ----------------
    print("\n=== Running CNN Evaluation ===")
    cnn_model = cnn_build_backbone(cfg, state_dict).to(device)
    
    X_tr_lat_cnn, y_tr_cnn = cnn_extract_features(cnn_model, train_ds, cfg, level=level, flatten=False)
    X_va_lat_cnn, y_va_cnn = cnn_extract_features(cnn_model, val_ds,   cfg, level=level, flatten=False)
    X_tr_all_cnn = np.concatenate([X_tr_lat_cnn, X_va_lat_cnn])
    y_tr_all_cnn = np.concatenate([y_tr_cnn,    y_va_cnn])
    del X_tr_lat_cnn, X_va_lat_cnn, y_tr_cnn, y_va_cnn; gc.collect()
    X_te_lat_cnn, y_te_cnn = cnn_extract_features(cnn_model, test_ds,  cfg, level=level, flatten=False)

    X_tr_raw_cnn, y_tr_raw_cnn = cnn_extract_raw(train_ds, cfg, level=level)
    X_va_raw_cnn, y_va_raw_cnn = cnn_extract_raw(val_ds,   cfg, level=level)
    X_tr_raw_all_cnn = np.concatenate([X_tr_raw_cnn, X_va_raw_cnn])
    y_tr_raw_all_cnn = np.concatenate([y_tr_raw_cnn, y_va_raw_cnn])
    del X_tr_raw_cnn, X_va_raw_cnn, y_tr_raw_cnn, y_va_raw_cnn; gc.collect()
    X_te_raw_cnn, y_te_raw_cnn = cnn_extract_raw(test_ds,  cfg, level=level)

    cnn_lat = fit_cnn(X_tr_all_cnn, y_tr_all_cnn, cfg, level=level, is_latent=True)
    cnn_raw = fit_cnn(X_tr_raw_all_cnn, y_tr_raw_all_cnn, cfg, level=level, is_latent=False)

    res["CNN on latent"] = cnn_evaluate_classifier(cnn_lat, X_te_lat_cnn, y_te_cnn, "CNN Latent", device=device)
    res["CNN on raw (baseline)"] = cnn_evaluate_classifier(cnn_raw, X_te_raw_cnn, y_te_raw_cnn, "CNN Raw", device=device)
    
    res["CNN on latent"]["y_test"] = y_te_cnn
    res["CNN on raw (baseline)"]["y_test"] = y_te_raw_cnn

    plot_predictions(test_ds, res["CNN on latent"]["y_pred"], res["CNN on latent"]["cm"], str(out_dir / f"{ckpt_name}_cnn_latent.png"), "orange", "CNN Latent Predictions")
    plot_predictions(test_ds, res["CNN on raw (baseline)"]["y_pred"], res["CNN on raw (baseline)"]["cm"], str(out_dir / f"{ckpt_name}_cnn_raw.png"), "brown", "CNN Raw Predictions")

    del cnn_model, cnn_lat, cnn_raw; gc.collect()

    # ---------------- XGB Evaluation ----------------
    print("\n=== Running XGB Evaluation ===")
    xgb_model = xgb_build_backbone(cfg, state_dict).to(device)
    
    X_tr_lat_xgb, y_tr_xgb = xgb_extract_features(xgb_model, train_ds, cfg, level=level)
    X_va_lat_xgb, y_va_xgb = xgb_extract_features(xgb_model, val_ds,   cfg, level=level)
    X_tr_all_xgb = np.concatenate([X_tr_lat_xgb, X_va_lat_xgb])
    y_tr_all_xgb = np.concatenate([y_tr_xgb,    y_va_xgb])
    del X_tr_lat_xgb, X_va_lat_xgb, y_tr_xgb, y_va_xgb; gc.collect()
    X_te_lat_xgb, y_te_xgb = xgb_extract_features(xgb_model, test_ds,  cfg, level=level)

    X_tr_raw_xgb, y_tr_raw_xgb = xgb_extract_raw(train_ds, cfg, level=level)
    X_va_raw_xgb, y_va_raw_xgb = xgb_extract_raw(val_ds,   cfg, level=level)
    X_tr_raw_all_xgb = np.concatenate([X_tr_raw_xgb, X_va_raw_xgb])
    y_tr_raw_all_xgb = np.concatenate([y_tr_raw_xgb, y_va_raw_xgb])
    del X_tr_raw_xgb, X_va_raw_xgb, y_tr_raw_xgb, y_va_raw_xgb; gc.collect()
    X_te_raw_xgb, y_te_raw_xgb = xgb_extract_raw(test_ds,  cfg, level=level)
    
    if "xgb_params" not in cfg:
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

    xgb_lat = fit_xgb(X_tr_all_xgb, y_tr_all_xgb, cfg)
    xgb_raw = fit_xgb(X_tr_raw_all_xgb, y_tr_raw_all_xgb, cfg)

    res["XGBoost on latent"] = xgb_evaluate_classifier(xgb_lat, X_te_lat_xgb, y_te_xgb, "XGB Latent", device=device)
    res["XGBoost on raw (baseline)"] = xgb_evaluate_classifier(xgb_raw, X_te_raw_xgb, y_te_raw_xgb, "XGB Raw", device=device)

    res["XGBoost on latent"]["y_test"] = y_te_xgb
    res["XGBoost on raw (baseline)"]["y_test"] = y_te_raw_xgb

    plot_predictions(test_ds, res["XGBoost on latent"]["y_pred"], res["XGBoost on latent"]["cm"], str(out_dir / f"{ckpt_name}_xgb_latent.png"), "orange", "XGBoost Latent Predictions")
    plot_predictions(test_ds, res["XGBoost on raw (baseline)"]["y_pred"], res["XGBoost on raw (baseline)"]["cm"], str(out_dir / f"{ckpt_name}_xgb_raw.png"), "brown", "XGBoost Raw Predictions")

    del xgb_model, xgb_lat, xgb_raw; gc.collect()
    
    return res

def format_tsv(res, out_path, model_type="XGBoost"):
    """
    Format output TSV to strictly match the requested spreadsheet image.
    The spreadsheet has columns:
    Model, Class/Metric, Precision, Recall, F1-Score, Support, ROC AUC, PR AUC (AP), Test LogLoss
    """
    latent_key = f"{model_type} on latent"
    raw_key = f"{model_type} on raw (baseline)"
    
    rows = []
    headers = ["Model", "Class/Metric", "Precision", "Recall", "F1-Score", "Support", "ROC AUC", "PR AUC (AP)", "Test LogLoss"]
    
    def add_model_rows(model_name, m_res):
        if not m_res: return
        cr = classification_report(m_res["y_test"].astype(int).flatten(), m_res["y_pred"], output_dict=True, zero_division=0)
        
        # ambient
        amb = cr["0"]
        rows.append([model_name, "ambient", f"{amb['precision']:.2f}", f"{amb['recall']:.2f}", f"{amb['f1-score']:.2f}", f"{amb['support']}", f"{m_res['roc_auc']:.4f}", f"{m_res['pr_auc']:.4f}", f"{m_res['logloss']:.4f}"])
        # ICME
        icme = cr["1"]
        rows.append([model_name, "ICME", f"{icme['precision']:.2f}", f"{icme['recall']:.2f}", f"{icme['f1-score']:.2f}", f"{icme['support']}", "", "", ""])
        # accuracy
        rows.append([model_name, "accuracy", "", "", f"{cr['accuracy']:.2f}", f"{cr['macro avg']['support']}", "", "", ""])
        # macro avg
        ma = cr["macro avg"]
        rows.append([model_name, "macro avg", f"{ma['precision']:.2f}", f"{ma['recall']:.2f}", f"{ma['f1-score']:.2f}", f"{ma['support']}", "", "", ""])
        # weighted avg
        wa = cr["weighted avg"]
        rows.append([model_name, "weighted avg", f"{wa['precision']:.2f}", f"{wa['recall']:.2f}", f"{wa['f1-score']:.2f}", f"{wa['support']}", "", "", ""])
        rows.append(["", "", "", "", "", "", "", "", ""])
        
    add_model_rows(latent_key, res.get(latent_key))
    add_model_rows(raw_key, res.get(raw_key))
    
    # Summary
    l_res = res.get(latent_key)
    r_res = res.get(raw_key)
    if l_res and r_res:
        l_cr = classification_report(l_res["y_test"].astype(int).flatten(), l_res["y_pred"], output_dict=True, zero_division=0)["1"]
        r_cr = classification_report(r_res["y_test"].astype(int).flatten(), r_res["y_pred"], output_dict=True, zero_division=0)["1"]
        rows.append(["Summary (Latent)", latent_key, f"{l_cr['precision']:.4f}", f"{l_cr['recall']:.4f}", f"{l_cr['f1-score']:.4f}", f"{l_cr['support']}", f"{l_res['roc_auc']:.4f}", f"{l_res['pr_auc']:.4f}", f"{l_res['logloss']:.4f}"])
        rows.append(["Summary (Raw)", raw_key, f"{r_cr['precision']:.4f}", f"{r_cr['recall']:.4f}", f"{r_cr['f1-score']:.4f}", f"{r_cr['support']}", f"{r_res['roc_auc']:.4f}", f"{r_res['pr_auc']:.4f}", f"{r_res['logloss']:.4f}"])
        rows.append(["", "", "", "", "", "", "", "", ""])
        rows.append(["", "", "", "", "", "", "", "", ""])
    
        # Confusion Matrices
        rows.append([latent_key, "", "", raw_key, "", ""])
        rows.append(["", "Predicted Ambient", "Predicted ICME", "", "Predicted Ambient", "Predicted ICME"])
        l_cm = l_res["cm"]
        r_cm = r_res["cm"]
        rows.append(["Actual Ambient", f"{l_cm[0,0]}", f"{l_cm[0,1]}", "Actual Ambient", f"{r_cm[0,0]}", f"{r_cm[0,1]}"])
        rows.append(["Actual ICME", f"{l_cm[1,0]}", f"{l_cm[1,1]}", "Actual ICME", f"{r_cm[1,0]}", f"{r_cm[1,1]}"])
    
    df = pd.DataFrame(rows, columns=headers)
    # Save to TSV
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved TSV to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run both evaluators and output required TSV and images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    res = run_evaluation(ckpt_path)
    
    format_tsv(res, ckpt_path.parent / f"{ckpt_path.stem}_xgb_results.tsv", "XGBoost")
    format_tsv(res, ckpt_path.parent / f"{ckpt_path.stem}_cnn_results.tsv", "CNN")
