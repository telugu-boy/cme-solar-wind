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
from experiments.plot_utils import plot_roc_prc, plot_logit_slice
import pickle
import shutil
import tempfile

from experiments.cnn_evaluator import (
    DownstreamCNN,
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

def run_evaluation(checkpoint_path: Path, run_cnn: bool = True, run_xgb: bool = True, use_existing_checkpoints: bool = False, logitplot_start_date: str = '2015-07-01', logitplot_end_date: str = '2016-07-01'):
    out_dir = checkpoint_path.parent
    
    chkpts_dir = out_dir / "checkpoints"
    chkpts_dir.mkdir(exist_ok=True)
    roc_prc_dir = out_dir / "roc_prc_curves"
    roc_prc_dir.mkdir(exist_ok=True)
    slice_dir = out_dir / "logit_plots"
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

    # Create slice dataset based on start and end dates
    print(f"Creating {logitplot_start_date} to {logitplot_end_date} slice dataset...")
    omni_df_1yr = omni_df.loc[logitplot_start_date:logitplot_end_date].copy()
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
        
        X_te_lat_cnn, y_te_cnn = cnn_extract_features(cnn_model, test_ds,  cfg, level=level, flatten=False)
        X_te_raw_cnn, y_te_raw_cnn = cnn_extract_raw(test_ds,  cfg, level=level)
    
        if not use_existing_checkpoints:
            X_tr_lat_cnn, y_tr_cnn = cnn_extract_features(cnn_model, train_ds, cfg, level=level, flatten=False)
            X_va_lat_cnn, y_va_cnn = cnn_extract_features(cnn_model, val_ds,   cfg, level=level, flatten=False)
            X_tr_all_cnn = np.concatenate([X_tr_lat_cnn, X_va_lat_cnn])
            y_tr_all_cnn = np.concatenate([y_tr_cnn,    y_va_cnn])
            del X_tr_lat_cnn, X_va_lat_cnn, y_tr_cnn, y_va_cnn; gc.collect()

            X_tr_raw_cnn, y_tr_raw_cnn = cnn_extract_raw(train_ds, cfg, level=level)
            X_va_raw_cnn, y_va_raw_cnn = cnn_extract_raw(val_ds,   cfg, level=level)
            X_tr_raw_all_cnn = np.concatenate([X_tr_raw_cnn, X_va_raw_cnn])
            y_tr_raw_all_cnn = np.concatenate([y_tr_raw_cnn, y_va_raw_cnn])
            del X_tr_raw_cnn, X_va_raw_cnn, y_tr_raw_cnn, y_va_raw_cnn; gc.collect()

            cnn_lat = fit_cnn(X_tr_all_cnn, y_tr_all_cnn, cfg, level=level, is_latent=True)
            del X_tr_all_cnn, y_tr_all_cnn; gc.collect()

            cnn_raw = fit_cnn(X_tr_raw_all_cnn, y_tr_raw_all_cnn, cfg, level=level, is_latent=False)
            del X_tr_raw_all_cnn, y_tr_raw_all_cnn; gc.collect()
            
            # Save checkpoints
            torch.save(cnn_lat.state_dict(), chkpts_dir / f"{ckpt_name}_cnn_latent.pt")
            torch.save(cnn_raw.state_dict(), chkpts_dir / f"{ckpt_name}_cnn_raw.pt")
        else:
            n_feat = cfg.get("_n_features", len(feature_cols))
            cnn_lat = DownstreamCNN(in_features=80, is_latent=True, level=level, C=n_feat, D=cfg.get("d_model", 64)).to(device)
            cnn_lat.load_state_dict(torch.load(chkpts_dir / f"{ckpt_name}_cnn_latent.pt", map_location=device, weights_only=True))
            cnn_raw = DownstreamCNN(in_features=cfg["patch_length"] * n_feat, is_latent=False, level=level, C=n_feat, D=cfg.get("d_model", 64)).to(device)
            cnn_raw.load_state_dict(torch.load(chkpts_dir / f"{ckpt_name}_cnn_raw.pt", map_location=device, weights_only=True))

        res["CNN on latent"] = cnn_evaluate_classifier(cnn_lat, X_te_lat_cnn, y_te_cnn, "CNN Latent", device=device)
        del X_te_lat_cnn; gc.collect()

        res["CNN on raw (baseline)"] = cnn_evaluate_classifier(cnn_raw, X_te_raw_cnn, y_te_raw_cnn, "CNN Raw", device=device)
        del X_te_raw_cnn; gc.collect()
    
        res["CNN on latent"]["y_test"] = y_te_cnn
        res["CNN on raw (baseline)"]["y_test"] = y_te_raw_cnn

        plot_predictions(test_ds, res["CNN on latent"]["y_pred"], res["CNN on latent"]["cm"], str(test_pred_dir / f"{ckpt_name}_cnn_latent.png"), "purple", "CNN Latent Predictions")
        plot_predictions(test_ds, res["CNN on raw (baseline)"]["y_pred"], res["CNN on raw (baseline)"]["cm"], str(test_pred_dir / f"{ckpt_name}_cnn_raw.png"), "darkgoldenrod", "CNN Raw Predictions")

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
        
        plot_logit_slice(ds_1yr, prob_lat_1yr, str(slice_dir / f"{ckpt_name}_cnn_latent_logit.png"), "CNN Latent", color="purple", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
        plot_logit_slice(ds_1yr, prob_raw_1yr, str(slice_dir / f"{ckpt_name}_cnn_raw_logit.png"), "CNN Raw", color="darkgoldenrod", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
    
        del cnn_model, cnn_lat, cnn_raw; gc.collect()

    # ---------------- XGB Evaluation ----------------
    if run_xgb:
        print("\n=== Running XGB Evaluation ===")
        xgb_model = xgb_build_backbone(cfg, state_dict).to(device)
    
        X_te_lat_xgb, y_te_xgb = xgb_extract_features(xgb_model, test_ds,  cfg, level=level)
        X_te_raw_xgb, y_te_raw_xgb = xgb_extract_raw(test_ds,  cfg, level=level)
    
        if not use_existing_checkpoints:
            X_tr_lat_xgb, y_tr_xgb = xgb_extract_features(xgb_model, train_ds, cfg, level=level)
            X_va_lat_xgb, y_va_xgb = xgb_extract_features(xgb_model, val_ds,   cfg, level=level)
            X_tr_all_xgb = np.concatenate([X_tr_lat_xgb, X_va_lat_xgb])
            y_tr_all_xgb = np.concatenate([y_tr_xgb,    y_va_xgb])
            del X_tr_lat_xgb, X_va_lat_xgb, y_tr_xgb, y_va_xgb; gc.collect()

            X_tr_raw_xgb, y_tr_raw_xgb = xgb_extract_raw(train_ds, cfg, level=level)
            X_va_raw_xgb, y_va_raw_xgb = xgb_extract_raw(val_ds,   cfg, level=level)
            X_tr_raw_all_xgb = np.concatenate([X_tr_raw_xgb, X_va_raw_xgb])
            y_tr_raw_all_xgb = np.concatenate([y_tr_raw_xgb, y_va_raw_xgb])
            del X_tr_raw_xgb, X_va_raw_xgb, y_tr_raw_xgb, y_va_raw_xgb; gc.collect()
        
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
            
            # Save checkpoints
            with open(chkpts_dir / f"{ckpt_name}_xgb_latent.pkl", "wb") as f:
                pickle.dump(xgb_lat, f)
            with open(chkpts_dir / f"{ckpt_name}_xgb_raw.pkl", "wb") as f:
                pickle.dump(xgb_raw, f)
        else:
            with open(chkpts_dir / f"{ckpt_name}_xgb_latent.pkl", "rb") as f:
                xgb_lat = pickle.load(f)
            with open(chkpts_dir / f"{ckpt_name}_xgb_raw.pkl", "rb") as f:
                xgb_raw = pickle.load(f)

        res["XGBoost on latent"] = xgb_evaluate_classifier(xgb_lat, X_te_lat_xgb, y_te_xgb, "XGB Latent", device=device)
        del X_te_lat_xgb; gc.collect()

        res["XGBoost on raw (baseline)"] = xgb_evaluate_classifier(xgb_raw, X_te_raw_xgb, y_te_raw_xgb, "XGB Raw", device=device)
        del X_te_raw_xgb; gc.collect()

        res["XGBoost on latent"]["y_test"] = y_te_xgb
        res["XGBoost on raw (baseline)"]["y_test"] = y_te_raw_xgb

        plot_predictions(test_ds, res["XGBoost on latent"]["y_pred"], res["XGBoost on latent"]["cm"], str(test_pred_dir / f"{ckpt_name}_xgb_latent.png"), "purple", "XGBoost Latent Predictions")
        plot_predictions(test_ds, res["XGBoost on raw (baseline)"]["y_pred"], res["XGBoost on raw (baseline)"]["cm"], str(test_pred_dir / f"{ckpt_name}_xgb_raw.png"), "darkgoldenrod", "XGBoost Raw Predictions")

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

        plot_logit_slice(ds_1yr, prob_lat_1yr_xgb, str(slice_dir / f"{ckpt_name}_xgb_latent_logit.png"), "XGBoost Latent", color="purple", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
        plot_logit_slice(ds_1yr, prob_raw_1yr_xgb, str(slice_dir / f"{ckpt_name}_xgb_raw_logit.png"), "XGBoost Raw", color="darkgoldenrod", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
    
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
    parser.add_argument("--use_existing_checkpoints", action="store_true", help="Load existing checkpoints instead of retraining")
    parser.add_argument("--logitplot_start_date", type=str, default="2015-07-01", help="Start date for plot slice (YYYY-MM-DD)")
    parser.add_argument("--logitplot_end_date", type=str, default="2016-07-01", help="End date for plot slice (YYYY-MM-DD)")
    args = parser.parse_args()
    
    if not args.cnn and not args.xgb:
        import sys
        print("Error: You must specify at least one model to evaluate by including the --cnn and/or --xgb flags.")
        sys.exit(1)
    
    ckpt_path = Path(args.checkpoint)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    package = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = package["cfg"]
    
    val_end_dt = pd.to_datetime(cfg["val_end"])
    omni_end_dt = pd.to_datetime(cfg["omni_end"])
    
    try:
        sd = pd.to_datetime(args.logitplot_start_date)
        ed = pd.to_datetime(args.logitplot_end_date)
    except Exception as e:
        import sys
        print(f"Error parsing dates: {e}")
        sys.exit(1)
        
    if sd < val_end_dt or ed > omni_end_dt:
        import sys
        print(f"Error: The requested dates {args.logitplot_start_date} to {args.logitplot_end_date} are out of bounds of the test set.")
        print(f"The valid test set date range is from {cfg['val_end']} to {cfg['omni_end']}.")
        sys.exit(1)
        
    res = run_evaluation(ckpt_path, run_cnn=args.cnn, run_xgb=args.xgb, use_existing_checkpoints=args.use_existing_checkpoints, logitplot_start_date=args.logitplot_start_date, logitplot_end_date=args.logitplot_end_date)
    
    tsv_dir = ckpt_path.parent / "metrics_tsv"
    tsv_dir.mkdir(exist_ok=True)
    
    if args.xgb:
        format_tsv(res, tsv_dir / f"{ckpt_path.stem}_xgb_results.tsv", "XGBoost")
    if args.cnn:
        format_tsv(res, tsv_dir / f"{ckpt_path.stem}_cnn_results.tsv", "CNN")

    # Zip up the directory containing the checkpoint and outputs
    zip_target = ckpt_path.parent / f"{ckpt_path.parent.name}_results.zip"
    if zip_target.exists():
        zip_target.unlink()
    print(f"\nZipping up results to {zip_target}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zip_base = Path(tmpdir) / "archive"
        shutil.make_archive(str(tmp_zip_base), 'zip', str(ckpt_path.parent))
        shutil.move(f"{tmp_zip_base}.zip", str(zip_target))
    print("Zipping complete!")
