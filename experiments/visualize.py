import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_predictions(dataset, y_pred, cm, out_path: str, color: str, title: str):
    """
    Plots the B-field with predictions overlaid.
    dataset: OmniPatchDataset
    y_pred: (total_patches,) predicted binary labels
    cm: confusion matrix array
    out_path: str to save PNG
    color: 'orange' or 'brown'
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    P = dataset.num_patches
    PS = dataset.patch_stride
    PL = dataset.patch_length
    
    # Ensure y_pred is flattened correctly just in case
    y_pred = y_pred.flatten()
    
    if len(y_pred) != len(dataset) * P:
        print(f"Warning: y_pred length ({len(y_pred)}) does not match dataset length * P ({len(dataset) * P}). Visualization might be inaccurate.")
        # Try to slice or pad, though this shouldn't happen
        y_pred = y_pred[:len(dataset)*P]
        
    y_pred_reshaped = y_pred.reshape(len(dataset), P)
    pred_ts = np.zeros(len(dataset.times))
    count_ts = np.zeros(len(dataset.times))
    
    for idx, s in enumerate(dataset.window_starts):
        for p in range(P):
            p_start = s + p * PS
            p_end = p_start + PL
            pred_ts[p_start:p_end] += y_pred_reshaped[idx, p]
            count_ts[p_start:p_end] += 1
            
    mask = count_ts > 0
    pred_ts[mask] /= count_ts[mask]
    
    # Features are normalized. F is at index 0, flow_speed is at index 4
    # based on CFG["raw_feature_cols"] = ["F", "BX_GSE", "BY_GSE", "BZ_GSE", "flow_speed", ...]
    F_field = dataset.data[:, 0]
    flow_speed = dataset.data[:, 4]
    
    fig, ax = plt.subplots(figsize=(24, 6), dpi=300)
    
    ax.plot(dataset.times, F_field, color='green', alpha=0.7, label='Total B-field (F)')
    ax.plot(dataset.times, flow_speed, color='purple', alpha=0.7, label='Solar Wind Speed')
    
    y_min = min(F_field.min(), flow_speed.min())
    y_max = max(F_field.max(), flow_speed.max())
    
    # Overlay predictions
    # If pred_ts > 0.5 (majority of overlapping windows predict ICME)
    ax.fill_between(dataset.times, y_min, y_max, where=(pred_ts > 0.5), 
                    color=color, alpha=0.4, step='pre', label=f'{title}')
    
    # Ground truth
    gt = dataset.timestep_labels
    ax.fill_between(dataset.times, y_min, y_max, where=(gt > 0.5), 
                    color='black', alpha=0.15, step='pre', label='Ground Truth ICME')
    
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
    
    for s, e in zip(starts, ends):
        ax.axvline(s, color='black', linestyle='--', alpha=0.8)
        ax.axvline(e, color='black', linestyle='--', alpha=0.8)
        
    ax.set_title(f"{title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Features")
    
    # Add confusion matrix
    cm_text = f"Confusion Matrix:\nTN: {cm[0,0]:,}  FP: {cm[0,1]:,}\nFN: {cm[1,0]:,}  TP: {cm[1,1]:,}"
    ax.text(0.01, 0.95, cm_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved {out_path}")
