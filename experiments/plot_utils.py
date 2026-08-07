import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_event_prc(prc_recall, prc_precision, pr_auc, out_path_prc, title):
    fig_prc, ax_prc = plt.subplots(figsize=(7, 6), dpi=300)
    ax_prc.plot(prc_recall, prc_precision, color='blue', lw=2, label=f'PRC curve (area = {pr_auc:.4f})')
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall (Sensitivity)')
    ax_prc.set_ylabel('Precision (Positive Predictive Value)')
    ax_prc.set_title(f'{title} - Precision-Recall Curve')
    ax_prc.legend(loc="lower left")
    fig_prc.tight_layout()
    fig_prc.savefig(out_path_prc)
    plt.close(fig_prc)

def plot_logit_slice(dataset, y_prob, out_path, title, color="orange", logitplot_start_date="2015-07-01", logitplot_end_date="2016-07-01"):
    P = dataset.num_patches
    PS = dataset.patch_stride
    PL = dataset.patch_length
    
    y_prob = y_prob.flatten()
    y_reshaped = y_prob.reshape(len(dataset), P)
    
    pred_ts = np.zeros(len(dataset.times))
    count_ts = np.zeros(len(dataset.times))
    
    for idx, s in enumerate(dataset.window_starts):
        for p in range(P):
            p_start = s + p * PS
            p_end = p_start + PL
            pred_ts[p_start:p_end] += y_reshaped[idx, p]
            count_ts[p_start:p_end] += 1
            
    mask = count_ts > 0
    pred_ts[mask] /= count_ts[mask]
    
    F_field = dataset.data[:, 0]
    
    fig, ax = plt.subplots(figsize=(24, 6), dpi=300)
    
    ax.plot(dataset.times, F_field, color='green', alpha=0.3, label='Total B-field (F)')
    
    ax2 = ax.twinx()
    ax2.plot(dataset.times, pred_ts, color=color, alpha=0.8, label=f'{title} Probability')
    ax2.set_ylim(-0.05, 1.05)
    
    gt = dataset.timestep_labels
    ax2.fill_between(dataset.times, 0, 1, where=(gt > 0.5), 
                    color='black', alpha=0.15, step='pre', label='Ground Truth ICME')
    
    ax.set_title(f"{title} - Logit Plot ({logitplot_start_date} to {logitplot_end_date})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Features (B-field)")
    ax2.set_ylabel("Probability (0 to 1)")
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
