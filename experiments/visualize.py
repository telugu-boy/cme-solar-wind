import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_predictions_events(dataset, pred_events, tp, fp, fn, out_path: str, color: str, title: str):
    """
    Plots the B-field with predicted event ranges overlaid as shaded bounding boxes.
    dataset: OmniPatchDataset
    pred_events: List of [start_idx, end_idx] for merged predicted ICME events
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    F_field = dataset.data[:, 0]
    
    fig, ax = plt.subplots(figsize=(24, 6), dpi=300)
    ax.plot(dataset.times, F_field, color='green', alpha=0.7, label='Total B-field (F)')
    
    y_min = F_field.min()
    y_max = F_field.max()
    
    # Shade predicted events
    for i, e in enumerate(pred_events):
        start_t = dataset.times[e[0]]
        end_t = dataset.times[e[1]]
        lbl = f'{title}' if i == 0 else None
        ax.axvspan(start_t, end_t, color=color, alpha=0.4, label=lbl)
    
    # Ground truth
    gt = dataset.timestep_labels
    starts = []
    ends = []
    in_icme = False
    for i in range(len(gt)):
        if gt[i] > 0.5 and not in_icme:
            starts.append(dataset.times[i])
            in_icme = True
        elif gt[i] <= 0.5 and in_icme:
            ends.append(dataset.times[i])
            in_icme = False
            
    if in_icme:
        ends.append(dataset.times[-1])
    
    for i, (s, e) in enumerate(zip(starts, ends)):
        lbl = 'Ground Truth ICME' if i == 0 else None
        ax.axvspan(s, e, color='black', alpha=0.15, label=lbl)
        ax.axvline(s, color='black', linestyle='--', alpha=0.8)
        ax.axvline(e, color='black', linestyle='--', alpha=0.8)
        
    ax.set_title(f"{title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Features")
    
    # Add metric text
    metrics_text = f"IoU Events Metrics:\nTP: {tp}  FP: {fp}\nFN: {fn}"
    ax.text(0.01, 0.95, metrics_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved {out_path}")

def plot_gap_histogram(y_true, y_pred, out_path_true, out_path_pred, model_name, max_gap=20):
    true_idx = np.where(y_true.flatten() == 1)[0]
    true_gaps = np.diff(true_idx) if len(true_idx) > 1 else np.array([])
    true_gaps = true_gaps[true_gaps > 1]
    
    pred_idx = np.where(y_pred.flatten() == 1)[0]
    pred_gaps = np.diff(pred_idx) if len(pred_idx) > 1 else np.array([])
    pred_gaps = pred_gaps[pred_gaps > 1]
    
    if len(true_gaps) > 0:
        true_gaps_clipped = np.clip(true_gaps, 0, max_gap)
        plt.figure(figsize=(10, 6))
        plt.hist(true_gaps_clipped, bins=np.arange(1.5, max_gap + 1.5, 1), edgecolor='black')
        plt.title(f"True ICME Gaps (>= {max_gap} binned together)")
        plt.xlabel("Gap size (number of patches)")
        plt.ylabel("Frequency")
        plt.savefig(out_path_true)
        plt.close()
        
    if len(pred_gaps) > 0:
        pred_gaps_clipped = np.clip(pred_gaps, 0, max_gap)
        plt.figure(figsize=(10, 6))
        plt.hist(pred_gaps_clipped, bins=np.arange(1.5, max_gap + 1.5, 1), edgecolor='black')
        plt.title(f"{model_name} Predicted ICME Gaps (>= {max_gap} binned together)")
        plt.xlabel("Gap size (number of patches)")
        plt.ylabel("Frequency")
        plt.savefig(out_path_pred)
        plt.close()
