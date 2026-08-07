import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import gc
from sklearn.metrics import classification_report
import pickle
import json
import json

# ---------------------------------------------------------
# PyArrow / PyTorch DLL collision fix on Windows:
# We MUST read the parquet file BEFORE importing torch or 
# any modules that import torch, otherwise pyarrow segfaults.
# ---------------------------------------------------------
default_cache_path = Path("data/omni_cache_5min_full.parquet")
preloaded_omni = None
if default_cache_path.exists():
    print(f"Pre-loading data from {default_cache_path} to prevent PyArrow/PyTorch collision (Windows bug)")
    preloaded_omni = pd.read_parquet(default_cache_path)

import torch
# Import required functions from existing evaluators
from experiments.loaders import read_omni_cache, get_cr_icme_dataframe, engineer_features, make_datasets, OmniPatchDataset, build_icme_intervals, load_f107_index
from experiments.visualize import plot_predictions_events, plot_gap_histogram
from experiments.plot_utils import plot_event_prc, plot_logit_slice
from experiments.event_eval import evaluate_events

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

def run_evaluation(checkpoint_path: Path, run_cnn: bool = True, run_xgb: bool = True, use_existing_checkpoints: bool = False, logitplot_start_date: str = '2015-07-01', logitplot_end_date: str = '2016-07-01', preloaded_omni: pd.DataFrame = None, merge_threshold=4, iou_threshold=0.30, conf_agg='max'):
    out_dir = checkpoint_path.parent
    
    chkpts_dir = out_dir / "checkpoints"
    chkpts_dir.mkdir(exist_ok=True)
    roc_prc_dir = out_dir / "roc_prc_curves"
    roc_prc_dir.mkdir(exist_ok=True)
    slice_dir = out_dir / "logit_plots"
    slice_dir.mkdir(exist_ok=True)
    test_pred_dir = out_dir / "test_predictions"
    test_pred_dir.mkdir(exist_ok=True)
    misc_dir = out_dir / "misc" / "icme_pred_gaps"
    misc_dir.mkdir(parents=True, exist_ok=True)
    misc_dir.mkdir(parents=True, exist_ok=True)

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
    cache_path = Path(cfg["cache_path"])
    if preloaded_omni is not None and cache_path == Path("data/omni_cache_5min_full.parquet"):
        omni_full = preloaded_omni.copy()
        if not isinstance(omni_full.index, pd.DatetimeIndex):
            if "Timestamp" in omni_full.columns:
                omni_full = omni_full.set_index(pd.to_datetime(omni_full["Timestamp"]))
        if omni_full.index.tz is not None:
            omni_full.index = omni_full.index.tz_localize(None)
        
        f107_path = cache_path.parent / "omni_daily_f10.7_index.lst"
        if f107_path.exists():
            f107_series = load_f107_index(f107_path)
            omni_full = omni_full.join(f107_series, on=omni_full.index.normalize())
    else:
        omni_full = read_omni_cache(cache_path)

    omni_df   = omni_full.loc[str(cfg["omni_start"]) : str(cfg["omni_end"])].copy()
    cr_icmes  = get_cr_icme_dataframe(cfg["omni_start"], cfg["omni_end"], cfg["icme_catalog_path"])

    omni_df = engineer_features(omni_df, cfg)

    train_ds, val_ds, test_ds, new_scaler = make_datasets(
        omni_df, cr_icmes, feature_cols, cfg, scaler=None
    )
    scaler = new_scaler

    # Create slice dataset based on start and end dates
    print(f"Creating {logitplot_start_date} to {logitplot_end_date} slice dataset...")
    omni_dflogits = omni_df.loc[logitplot_start_date:logitplot_end_date].copy()
    omni_dflogits.interpolate(limit=6, limit_direction="both", inplace=True)
    omni_dflogits.fillna(0.0, inplace=True)
    datalogits = scaler.transform(omni_dflogits[feature_cols].values).astype(np.float32)
    icme_intervalslogits = build_icme_intervals(cr_icmes)
    dslogits = OmniPatchDataset(
        datalogits, omni_dflogits.index,
        icme_intervals=icme_intervalslogits,
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
    
        res["CNN on latent"].update(evaluate_events(test_ds, res["CNN on latent"]["y_prob"], merge_threshold, iou_threshold, conf_agg))
        res["CNN on raw (baseline)"].update(evaluate_events(test_ds, res["CNN on raw (baseline)"]["y_prob"], merge_threshold, iou_threshold, conf_agg))

        plot_predictions_events(test_ds, res["CNN on latent"]["pred_events"], res["CNN on latent"]["TP"], res["CNN on latent"]["FP"], res["CNN on latent"]["FN"], str(test_pred_dir / f"{ckpt_name}_cnn_latent.png"), "purple", "CNN Latent Predictions")
        plot_predictions_events(test_ds, res["CNN on raw (baseline)"]["pred_events"], res["CNN on raw (baseline)"]["TP"], res["CNN on raw (baseline)"]["FP"], res["CNN on raw (baseline)"]["FN"], str(test_pred_dir / f"{ckpt_name}_cnn_raw.png"), "darkgoldenrod", "CNN Raw Predictions")

        plot_gap_histogram(
            y_te_cnn, res["CNN on latent"]["y_pred"], 
            str(misc_dir / f"true_gaps.png"), 
            str(misc_dir / f"{ckpt_name}_cnn_latent_gaps.png"), 
            "CNN Latent"
        )
        plot_gap_histogram(
            y_te_raw_cnn, res["CNN on raw (baseline)"]["y_pred"], 
            str(misc_dir / f"true_gaps.png"), 
            str(misc_dir / f"{ckpt_name}_cnn_raw_gaps.png"), 
            "CNN Raw"
        )

        # Plot PRC
        plot_event_prc(res["CNN on latent"]["prc_recall"], res["CNN on latent"]["prc_precision"], res["CNN on latent"]["pr_auc"],
                     str(roc_prc_dir / f"{ckpt_name}_cnn_latent_prc.png"), "CNN Latent")
        plot_event_prc(res["CNN on raw (baseline)"]["prc_recall"], res["CNN on raw (baseline)"]["prc_precision"], res["CNN on raw (baseline)"]["pr_auc"],
                     str(roc_prc_dir / f"{ckpt_name}_cnn_raw_prc.png"), "CNN Raw")

        # 1-year slice plots
        Xlogits_lat_cnn, _ = cnn_extract_features(cnn_model, dslogits, cfg, level=level, flatten=False)
        Xlogits_raw_cnn, _ = cnn_extract_raw(dslogits, cfg, level=level)
    
        cnn_lat.eval()
        cnn_raw.eval()
        with torch.no_grad():
            logits_lat = cnn_lat(torch.tensor(Xlogits_lat_cnn, dtype=torch.float32, device=device))
            prob_latlogits = torch.sigmoid(logits_lat).cpu().numpy().flatten()
        
            logits_raw = cnn_raw(torch.tensor(Xlogits_raw_cnn, dtype=torch.float32, device=device))
            prob_rawlogits = torch.sigmoid(logits_raw).cpu().numpy().flatten()
    
        del Xlogits_lat_cnn, Xlogits_raw_cnn; gc.collect()
        
        plot_logit_slice(dslogits, prob_latlogits, str(slice_dir / f"{ckpt_name}_cnn_latent_logit.png"), "CNN Latent", color="purple", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
        plot_logit_slice(dslogits, prob_rawlogits, str(slice_dir / f"{ckpt_name}_cnn_raw_logit.png"), "CNN Raw", color="darkgoldenrod", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
    
        del cnn_model, cnn_lat, cnn_raw; gc.collect()

    # ---------------- XGB Evaluation ----------------
    if run_xgb:
        print("\n=== Running XGB Evaluation ===")
        xgb_model = xgb_build_backbone(cfg, state_dict).to(device)
    
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

        # ---------------- XGB Latent ----------------
        if not use_existing_checkpoints:
            print("Extracting XGB Latent Features...")
            X_tr_lat_xgb, y_tr_xgb = xgb_extract_features(xgb_model, train_ds, cfg, level=level)
            X_va_lat_xgb, y_va_xgb = xgb_extract_features(xgb_model, val_ds,   cfg, level=level)
            X_tr_all_xgb = np.concatenate([X_tr_lat_xgb, X_va_lat_xgb])
            y_tr_all_xgb = np.concatenate([y_tr_xgb,    y_va_xgb])
            del X_tr_lat_xgb, X_va_lat_xgb, y_tr_xgb, y_va_xgb; gc.collect()

            print("Fitting XGB Latent...")
            xgb_lat = fit_xgb(X_tr_all_xgb, y_tr_all_xgb, cfg)
            del X_tr_all_xgb, y_tr_all_xgb; gc.collect()

            with open(chkpts_dir / f"{ckpt_name}_xgb_latent.pkl", "wb") as f:
                pickle.dump(xgb_lat, f)
        else:
            with open(chkpts_dir / f"{ckpt_name}_xgb_latent.pkl", "rb") as f:
                xgb_lat = pickle.load(f)

        print("Testing XGB Latent...")
        X_te_lat_xgb, y_te_xgb = xgb_extract_features(xgb_model, test_ds, cfg, level=level)
        res["XGBoost on latent"] = xgb_evaluate_classifier(xgb_lat, X_te_lat_xgb, y_te_xgb, "XGBoost Latent")
        res["XGBoost on latent"].update(evaluate_events(test_ds, res["XGBoost on latent"]["y_prob"], merge_threshold, iou_threshold, conf_agg))
        
        plot_predictions_events(test_ds, res["XGBoost on latent"]["pred_events"], res["XGBoost on latent"]["TP"], res["XGBoost on latent"]["FP"], res["XGBoost on latent"]["FN"], str(test_pred_dir / f"{ckpt_name}_xgb_latent.png"), "purple", "XGBoost Latent Predictions")
        plot_gap_histogram(
            y_te_xgb, res["XGBoost on latent"]["y_pred"], 
            str(misc_dir / f"true_gaps.png"), 
            str(misc_dir / f"{ckpt_name}_xgb_latent_gaps.png"), 
            "XGBoost Latent"
        )
        plot_event_prc(res["XGBoost on latent"]["prc_recall"], res["XGBoost on latent"]["prc_precision"], res["XGBoost on latent"]["pr_auc"],
                     str(roc_prc_dir / f"{ckpt_name}_xgb_latent_prc.png"), "XGBoost Latent")

        del X_te_lat_xgb; gc.collect()

        # ---------------- XGB Raw ----------------
        if not use_existing_checkpoints:
            print("Extracting XGB Raw Features...")
            X_tr_raw_xgb, y_tr_raw_xgb = xgb_extract_raw(train_ds, cfg, level=level)
            X_va_raw_xgb, y_va_raw_xgb = xgb_extract_raw(val_ds,   cfg, level=level)
            X_tr_raw_all_xgb = np.concatenate([X_tr_raw_xgb, X_va_raw_xgb])
            y_tr_raw_all_xgb = np.concatenate([y_tr_raw_xgb, y_va_raw_xgb])
            del X_tr_raw_xgb, X_va_raw_xgb, y_tr_raw_xgb, y_va_raw_xgb; gc.collect()
        
            print("Fitting XGB Raw...")
            xgb_raw = fit_xgb(X_tr_raw_all_xgb, y_tr_raw_all_xgb, cfg)
            del X_tr_raw_all_xgb, y_tr_raw_all_xgb; gc.collect()
            
            with open(chkpts_dir / f"{ckpt_name}_xgb_raw.pkl", "wb") as f:
                pickle.dump(xgb_raw, f)
        else:
            with open(chkpts_dir / f"{ckpt_name}_xgb_raw.pkl", "rb") as f:
                xgb_raw = pickle.load(f)

        X_te_raw_xgb, y_te_raw_xgb = xgb_extract_raw(test_ds, cfg, level=level)
        res["XGBoost on raw (baseline)"] = xgb_evaluate_classifier(xgb_raw, X_te_raw_xgb, y_te_raw_xgb, "XGBoost Raw")
        res["XGBoost on raw (baseline)"].update(evaluate_events(test_ds, res["XGBoost on raw (baseline)"]["y_prob"], merge_threshold, iou_threshold, conf_agg))

        plot_predictions_events(test_ds, res["XGBoost on raw (baseline)"]["pred_events"], res["XGBoost on raw (baseline)"]["TP"], res["XGBoost on raw (baseline)"]["FP"], res["XGBoost on raw (baseline)"]["FN"], str(test_pred_dir / f"{ckpt_name}_xgb_raw.png"), "darkgoldenrod", "XGBoost Raw Predictions")

        plot_gap_histogram(
            y_te_raw_xgb, res["XGBoost on raw (baseline)"]["y_pred"], 
            str(misc_dir / f"true_gaps.png"), 
            str(misc_dir / f"{ckpt_name}_xgb_raw_gaps.png"), 
            "XGBoost Raw"
        )

        plot_event_prc(res["XGBoost on raw (baseline)"]["prc_recall"], res["XGBoost on raw (baseline)"]["prc_precision"], res["XGBoost on raw (baseline)"]["pr_auc"],
                     str(roc_prc_dir / f"{ckpt_name}_xgb_raw_prc.png"), "XGBoost Raw")

        # 1-year slice plots
        Xlogits_lat_xgb, _ = xgb_extract_features(xgb_model, dslogits, cfg, level=level)
        Xlogits_raw_xgb, _ = xgb_extract_raw(dslogits, cfg, level=level)
    
        if "cuda" in str(xgb_lat.get_params().get("device", "")) or "cuda" in str(device):
            Xlogits_lat_xgb_dev = torch.as_tensor(Xlogits_lat_xgb, device=device, dtype=torch.float32)
            Xlogits_raw_xgb_dev = torch.as_tensor(Xlogits_raw_xgb, device=device, dtype=torch.float32)
        else:
            Xlogits_lat_xgb_dev = Xlogits_lat_xgb
            Xlogits_raw_xgb_dev = Xlogits_raw_xgb

        prob_latlogits_xgb = xgb_lat.predict_proba(Xlogits_lat_xgb_dev)[:, 1]
        prob_rawlogits_xgb = xgb_raw.predict_proba(Xlogits_raw_xgb_dev)[:, 1]
    
        del Xlogits_lat_xgb_dev, Xlogits_raw_xgb_dev
        try: del Xlogits_lat_xgb, Xlogits_raw_xgb
        except: pass
        gc.collect()
    
        if hasattr(prob_latlogits_xgb, "cpu"): prob_latlogits_xgb = prob_latlogits_xgb.cpu().numpy()
        if hasattr(prob_rawlogits_xgb, "cpu"): prob_rawlogits_xgb = prob_rawlogits_xgb.cpu().numpy()

        plot_logit_slice(dslogits, prob_latlogits_xgb, str(slice_dir / f"{ckpt_name}_xgb_latent_logit.png"), "XGBoost Latent", color="purple", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
        plot_logit_slice(dslogits, prob_rawlogits_xgb, str(slice_dir / f"{ckpt_name}_xgb_raw_logit.png"), "XGBoost Raw", color="darkgoldenrod", logitplot_start_date=logitplot_start_date, logitplot_end_date=logitplot_end_date)
    
        del xgb_model, xgb_lat, xgb_raw; gc.collect()
        
    return res

def format_tsv(res, out_path, model_type="XGBoost"):
    latent_key = f"{model_type} on latent"
    raw_key = f"{model_type} on raw (baseline)"
    
    rows = []
    headers = ["Model", "Precision", "Recall", "F1-Score", "TP", "FP", "FN", "PR AUC"]
    
    def add_model_rows(model_name, m_res):
        if not m_res: return
        rows.append([model_name, f"{m_res['precision']:.4f}", f"{m_res['recall']:.4f}", f"{m_res['f1']:.4f}", f"{m_res['TP']}", f"{m_res['FP']}", f"{m_res['FN']}", f"{m_res['pr_auc']:.4f}"])
        
    add_model_rows(latent_key, res.get(latent_key))
    add_model_rows(raw_key, res.get(raw_key))
    
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
    parser.add_argument("--merge_threshold", type=int, default=4, help="Window length in patches to merge predictions (default 4)")
    parser.add_argument("--iou_threshold", type=float, default=0.30, help="IoU threshold for evaluating TPs, FPs (default 0.30)")
    parser.add_argument("--conf_agg", type=str, default="max", choices=["max", "mean", "median"], help="Aggregation function for confidence score (default max)")
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
        
    res = run_evaluation(
        ckpt_path, 
        run_cnn=args.cnn, 
        run_xgb=args.xgb, 
        use_existing_checkpoints=args.use_existing_checkpoints, 
        logitplot_start_date=args.logitplot_start_date, 
        logitplot_end_date=args.logitplot_end_date,
        preloaded_omni=preloaded_omni,
        merge_threshold=args.merge_threshold,
        iou_threshold=args.iou_threshold,
        conf_agg=args.conf_agg
    )
    
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
