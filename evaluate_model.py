import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc
from sklearn.metrics import classification_report

# Import required functions from existing evaluators
from experiments.loaders import read_omni_cache, get_cr_icme_dataframe, engineer_features, make_datasets, OmniPatchDataset, build_icme_intervals
from experiments.visualize import plot_predictions
from experiments.plot_utils import plot_roc_prc, plot_1year_slice
import pickle

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

def run_evaluation(checkpoint_path: Path, run_cnn: bool = True, run_xgb: bool = True):
    out_dir = checkpoint_path.parent
    
    chkpts_dir = out_dir / "checkpoints"
    chkpts_dir.mkdir(exist_ok=True)
    roc_prc_dir = out_dir / "roc_prc_curves"
    roc_prc_dir.mkdir(exist_ok=True)
    slice_dir = out_dir / "1_year_slices"
    slice_dir.mkdir(exist_ok=True)
    test_pred_dir = out_dir / "test_predictions"
    test_pred_dir.mkdir(exist_ok=True)

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

    # Create 1-year slice dataset (July 2015 - July 2016)
    print("Creating July 2015 - July 2016 1-year slice dataset...")
    omni_df_1yr = omni_df.loc["2015-07-01":"2016-07-01"].copy()
    omni_df_1yr.interpolate(limit=6, limit_direction="both", inplace=True)
    omni_df_1yr.fillna(0.0, inplace=True)
    data_1yr = scaler.transform(omni_df_1yr[feature_cols].values).astype(np.float32)
    icme_intervals_1yr = build_icme_intervals(cr_icmes)
    ds_1yr = OmniPatchDataset(
        data_1yr, omni_df_1yr.index,
        icme_intervals=icme_intervals_1yr,
        context_length=cfg["context_length"],
        prediction_length=cfg["prediction_length"],
        patch_length=cfg["patch_length"],
        patch_stride=cfg["patch_stride"],
        overlap_threshold=cfg["overlap_threshold"],
        window_stride=cfg["window_stride"],
    )

    # Dictionary to hold the raw results
    res = {}
    
    ckpt_name = checkpoint_path.stem

    # ---------------- CNN Evaluation ----------------
    if run_cnn:
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
        del X_tr_all_cnn, y_tr_all_cnn; gc.collect()

        cnn_raw = fit_cnn(X_tr_raw_all_cnn, y_tr_raw_all_cnn, cfg, level=level, is_latent=False)
        del X_tr_raw_all_cnn, y_tr_raw_all_cnn; gc.collect()

        res["CNN on latent"] = cnn_evaluate_classifier(cnn_lat, X_te_lat_cnn, y_te_cnn, "CNN Latent", device=device)
        del X_te_lat_cnn; gc.collect()

        res["CNN on raw (baseline)"] = cnn_evaluate_classifier(cnn_raw, X_te_raw_cnn, y_te_raw_cnn, "CNN Raw", device=device)
        del X_te_raw_cnn; gc.collect()
    
        res["CNN on latent"]["y_test"] = y_te_cnn
        res["CNN on raw (baseline)"]["y_test"] = y_te_raw_cnn

        plot_predictions(test_ds, res["CNN on latent"]["y_pred"], res["CNN on latent"]["cm"], str(test_pred_dir / f"{ckpt_name}_cnn_latent.png"), "orange", "CNN Latent Predictions")
        plot_predictions(test_ds, res["CNN on raw (baseline)"]["y_pred"], res["CNN on raw (baseline)"]["cm"], str(test_pred_dir / f"{ckpt_name}_cnn_raw.png"), "brown", "CNN Raw Predictions")

        # Save checkpoints
        torch.save(cnn_lat.state_dict(), chkpts_dir / f"{ckpt_name}_cnn_latent.pt")
        torch.save(cnn_raw.state_dict(), chkpts_dir / f"{ckpt_name}_cnn_raw.pt")

        # Plot ROC and PRC
        plot_roc_prc(y_te_cnn.flatten(), res["CNN on latent"]["y_prob"], 
                     str(roc_prc_dir / f"{ckpt_name}_cnn_latent_roc.png"), 
                     str(roc_prc_dir / f"{ckpt_name}_cnn_latent_prc.png"), "CNN Latent")
        plot_roc_prc(y_te_raw_cnn.flatten(), res["CNN on raw (baseline)"]["y_prob"], 
                     str(roc_prc_dir / f"{ckpt_name}_cnn_raw_roc.png"), 
                     str(roc_prc_dir / f"{ckpt_name}_cnn_raw_prc.png"), "CNN Raw")

        # 1-year slice plots
        X_1yr_lat_cnn, _ = cnn_extract_features(cnn_model, ds_1yr, cfg, level=level, flatten=False)
        X_1yr_raw_cnn, _ = cnn_extract_raw(ds_1yr, cfg, level=level)
    
        cnn_lat.eval()
        cnn_raw.eval()
        with torch.no_grad():
            logits_lat = cnn_lat(torch.tensor(X_1yr_lat_cnn, dtype=torch.float32, device=device))
            prob_lat_1yr = torch.sigmoid(logits_lat).cpu().numpy().flatten()
        
            logits_raw = cnn_raw(torch.tensor(X_1yr_raw_cnn, dtype=torch.float32, device=device))
            prob_raw_1yr = torch.sigmoid(logits_raw).cpu().numpy().flatten()
    
        del X_1yr_lat_cnn, X_1yr_raw_cnn; gc.collect()
        
            plot_1year_slice(ds_1yr, prob_lat_1yr, str(slice_dir / f"{ckpt_name}_cnn_latent_1year.png"), "CNN Latent", color="orange")
            plot_1year_slice(ds_1yr, prob_raw_1yr, str(slice_dir / f"{ckpt_name}_cnn_raw_1year.png"), "CNN Raw", color="brown")
    
            del cnn_model, cnn_lat, cnn_raw; gc.collect()

    # ---------------- XGB Evaluation ----------------
    if run_xgb:
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
        del X_tr_all_xgb, y_tr_all_xgb; gc.collect()

        xgb_raw = fit_xgb(X_tr_raw_all_xgb, y_tr_raw_all_xgb, cfg)
        del X_tr_raw_all_xgb, y_tr_raw_all_xgb; gc.collect()

        res["XGBoost on latent"] = xgb_evaluate_classifier(xgb_lat, X_te_lat_xgb, y_te_xgb, "XGB Latent", device=device)
        del X_te_lat_xgb; gc.collect()

        res["XGBoost on raw (baseline)"] = xgb_evaluate_classifier(xgb_raw, X_te_raw_xgb, y_te_raw_xgb, "XGB Raw", device=device)
        del X_te_raw_xgb; gc.collect()

        res["XGBoost on latent"]["y_test"] = y_te_xgb
        res["XGBoost on raw (baseline)"]["y_test"] = y_te_raw_xgb

        plot_predictions(test_ds, res["XGBoost on latent"]["y_pred"], res["XGBoost on latent"]["cm"], str(test_pred_dir / f"{ckpt_name}_xgb_latent.png"), "orange", "XGBoost Latent Predictions")
        plot_predictions(test_ds, res["XGBoost on raw (baseline)"]["y_pred"], res["XGBoost on raw (baseline)"]["cm"], str(test_pred_dir / f"{ckpt_name}_xgb_raw.png"), "brown", "XGBoost Raw Predictions")

        # Save checkpoints
        with open(chkpts_dir / f"{ckpt_name}_xgb_latent.pkl", "wb") as f:
            pickle.dump(xgb_lat, f)
        with open(chkpts_dir / f"{ckpt_name}_xgb_raw.pkl", "wb") as f:
            pickle.dump(xgb_raw, f)

        # Plot ROC and PRC
        plot_roc_prc(y_te_xgb.flatten(), res["XGBoost on latent"]["y_prob"], 
                     str(roc_prc_dir / f"{ckpt_name}_xgb_latent_roc.png"), 
                     str(roc_prc_dir / f"{ckpt_name}_xgb_latent_prc.png"), "XGBoost Latent")
        plot_roc_prc(y_te_raw_xgb.flatten(), res["XGBoost on raw (baseline)"]["y_prob"], 
                     str(roc_prc_dir / f"{ckpt_name}_xgb_raw_roc.png"), 
                     str(roc_prc_dir / f"{ckpt_name}_xgb_raw_prc.png"), "XGBoost Raw")

        # 1-year slice plots
        X_1yr_lat_xgb, _ = xgb_extract_features(xgb_model, ds_1yr, cfg, level=level)
        X_1yr_raw_xgb, _ = xgb_extract_raw(ds_1yr, cfg, level=level)
    
        if "cuda" in str(xgb_lat.get_params().get("device", "")) or "cuda" in str(device):
            X_1yr_lat_xgb_dev = torch.as_tensor(X_1yr_lat_xgb, device=device, dtype=torch.float32)
            X_1yr_raw_xgb_dev = torch.as_tensor(X_1yr_raw_xgb, device=device, dtype=torch.float32)
        else:
            X_1yr_lat_xgb_dev = X_1yr_lat_xgb
            X_1yr_raw_xgb_dev = X_1yr_raw_xgb

        prob_lat_1yr_xgb = xgb_lat.predict_proba(X_1yr_lat_xgb_dev)[:, 1]
        prob_raw_1yr_xgb = xgb_raw.predict_proba(X_1yr_raw_xgb_dev)[:, 1]
    
        del X_1yr_lat_xgb_dev, X_1yr_raw_xgb_dev
        try: del X_1yr_lat_xgb, X_1yr_raw_xgb
        except: pass
        gc.collect()
    
        if hasattr(prob_lat_1yr_xgb, "cpu"): prob_lat_1yr_xgb = prob_lat_1yr_xgb.cpu().numpy()
        if hasattr(prob_raw_1yr_xgb, "cpu"): prob_raw_1yr_xgb = prob_raw_1yr_xgb.cpu().numpy()

            plot_1year_slice(ds_1yr, prob_lat_1yr_xgb, str(slice_dir / f"{ckpt_name}_xgb_latent_1year.png"), "XGBoost Latent", color="orange")
            plot_1year_slice(ds_1yr, prob_raw_1yr_xgb, str(slice_dir / f"{ckpt_name}_xgb_raw_1year.png"), "XGBoost Raw", color="brown")
    
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
    parser = argparse.ArgumentParser(description="Run evaluators and output required TSV and images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--cnn", action="store_true", help="Run CNN evaluation")
    parser.add_argument("--xgb", action="store_true", help="Run XGBoost evaluation")
    args = parser.parse_args()
    
    if not args.cnn and not args.xgb:
        import sys
        print("Error: You must specify at least one model to evaluate by including the --cnn and/or --xgb flags.")
        sys.exit(1)
    
    ckpt_path = Path(args.checkpoint)
    res = run_evaluation(ckpt_path, run_cnn=args.cnn, run_xgb=args.xgb)
    
    if args.xgb:
        format_tsv(res, ckpt_path.parent / f"{ckpt_path.stem}_xgb_results.tsv", "XGBoost")
    if args.cnn:
        format_tsv(res, ckpt_path.parent / f"{ckpt_path.stem}_cnn_results.tsv", "CNN")
