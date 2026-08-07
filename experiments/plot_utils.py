import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_combined_prc_merged(res, out_path, merge_threshold=None, iou_threshold=None, conf_agg=None):
    fig_prc, ax_prc = plt.subplots(figsize=(7, 6), dpi=300)
    
    colors = {
        'CNN on latent': 'purple',
        'CNN on raw (baseline)': 'darkgoldenrod',
        'XGBoost on latent': 'magenta',
        'XGBoost on raw (baseline)': 'orange'
    }
    
    for model_name, model_res in res.items():
        if "prc_recall" in model_res and "prc_precision" in model_res:
            c = colors.get(model_name, 'blue')
            pr_auc = model_res.get("pr_auc", 0.0)
            ax_prc.plot(model_res["prc_recall"], model_res["prc_precision"], color=c, lw=2, label=f'{model_name} (area = {pr_auc:.4f})')
            
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall (Sensitivity)')
    ax_prc.set_ylabel('Precision (Positive Predictive Value)')
    
    param_items = []
    if merge_threshold is not None:
        param_items.append(f"merge_threshold={merge_threshold}")
    if iou_threshold is not None:
        param_items.append(f"iou_threshold={iou_threshold}")
    if conf_agg is not None:
        param_items.append(f"conf_agg={conf_agg}")
    
    title = 'Merged Precision-Recall Curve'
    if param_items:
        title += f"\n({', '.join(param_items)})"
    ax_prc.set_title(title)
    ax_prc.legend(loc="lower left")
    fig_prc.tight_layout()
    fig_prc.savefig(out_path)
    plt.close(fig_prc)

def plot_combined_roc_patch(res, out_path):
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6), dpi=300)
    
    colors = {
        'CNN on latent': 'purple',
        'CNN on raw (baseline)': 'darkgoldenrod',
        'XGBoost on latent': 'magenta',
        'XGBoost on raw (baseline)': 'orange'
    }
    
    for model_name, model_res in res.items():
        if "y_true_patch" in model_res and "y_prob_patch" in model_res:
            c = colors.get(model_name, 'blue')
            y_true = model_res["y_true_patch"].flatten()
            y_prob = model_res["y_prob_patch"].flatten()
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_a = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=c, lw=2, label=f'{model_name} (AUC = {roc_a:.4f})')
            
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Patch-level ROC Curve')
    ax_roc.legend(loc="lower right")

    fig_roc.tight_layout()
    fig_roc.savefig(out_path)
    plt.close(fig_roc)

def plot_combined_prc_patch(res, out_path):
    fig_prc, ax_prc = plt.subplots(figsize=(7, 6), dpi=300)
    
    colors = {
        'CNN on latent': 'purple',
        'CNN on raw (baseline)': 'darkgoldenrod',
        'XGBoost on latent': 'magenta',
        'XGBoost on raw (baseline)': 'orange'
    }
    
    for model_name, model_res in res.items():
        if "y_true_patch" in model_res and "y_prob_patch" in model_res:
            c = colors.get(model_name, 'blue')
            y_true = model_res["y_true_patch"].flatten()
            y_prob = model_res["y_prob_patch"].flatten()
            
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            prc_a = auc(rec, prec)
            ax_prc.plot(rec, prec, color=c, lw=2, label=f'{model_name} (AUC = {prc_a:.4f})')

    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall (Sensitivity)')
    ax_prc.set_ylabel('Precision (Positive Predictive Value)')
    ax_prc.set_title('Patch-level Precision-Recall Curve')
    ax_prc.legend(loc="lower left")

    fig_prc.tight_layout()
    fig_prc.savefig(out_path)
    plt.close(fig_prc)

def plot_combined_logit_slice(dataset, logit_probs_dict, out_path, logitplot_start_date="2015-07-01", logitplot_end_date="2016-07-01"):
    models = ['CNN on latent', 'CNN on raw (baseline)', 'XGBoost on latent', 'XGBoost on raw (baseline)']
    colors = {
        'CNN on latent': 'purple',
        'CNN on raw (baseline)': 'darkgoldenrod',
        'XGBoost on latent': 'purple',
        'XGBoost on raw (baseline)': 'darkgoldenrod'
    }
    
    F_field = dataset.data[:, 0]
    gt = dataset.timestep_labels
    
    P = dataset.num_patches
    PS = dataset.patch_stride
    PL = dataset.patch_length
    
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(24, 12), dpi=300)
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        ax.plot(dataset.times, F_field, color='green', alpha=0.3, label='Total B-field (F)')
        
        ax2 = ax.twinx()
        
        if model_name in logit_probs_dict:
            y_prob = logit_probs_dict[model_name].flatten()
            y_reshaped = y_prob.reshape(len(dataset), P)
            
            pred_ts = np.zeros(len(dataset.times))
            count_ts = np.zeros(len(dataset.times))
            
            for d_idx, s in enumerate(dataset.window_starts):
                for p in range(P):
                    p_start = s + p * PS
                    p_end = p_start + PL
                    pred_ts[p_start:p_end] += y_reshaped[d_idx, p]
                    count_ts[p_start:p_end] += 1
                    
            mask = count_ts > 0
            pred_ts[mask] /= count_ts[mask]
            
            color = colors.get(model_name, 'orange')
            ax2.plot(dataset.times, pred_ts, color=color, alpha=0.8, label=f'{model_name} Probability')
            
        ax2.set_ylim(-0.05, 1.05)
        
        ax2.fill_between(dataset.times, 0, 1, where=(gt > 0.5), 
                        color='black', alpha=0.15, step='pre', label='Ground Truth ICME')
        
        ax.set_title(f"{model_name} - Logit Plot ({logitplot_start_date} to {logitplot_end_date})")
        ax.set_ylabel("Normalized B-field")
        ax2.set_ylabel("Probability (0 to 1)")
        
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        
    axes[-1].set_xlabel("Time")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
