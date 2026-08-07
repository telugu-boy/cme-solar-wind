import numpy as np
from sklearn.metrics import auc

def get_events(binary_ts):
    """Find start and end indices of contiguous 1s in a boolean array."""
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(binary_ts):
        if val and not in_event:
            start = i
            in_event = True
        elif not val and in_event:
            events.append([start, i - 1])
            in_event = False
    if in_event:
        events.append([start, len(binary_ts) - 1])
    return events

def merge_events(events, merge_gap):
    """Merge events that are separated by <= merge_gap indices."""
    if not events:
        return []
    merged = [events[0]]
    for e in events[1:]:
        prev = merged[-1]
        if e[0] - prev[1] - 1 <= merge_gap:
            prev[1] = e[1]
        else:
            merged.append(e)
    return merged

def compute_iou(e1, e2):
    """Compute Intersection over Union for two 1D events [start, end]."""
    intersection = max(0, min(e1[1], e2[1]) - max(e1[0], e2[0]) + 1)
    if intersection == 0:
        return 0.0
    union = max(e1[1], e2[1]) - min(e1[0], e2[0]) + 1
    return intersection / union

def evaluate_events(dataset, y_prob, merge_threshold_patches=4, iou_threshold=0.30, conf_agg='max'):
    P = dataset.num_patches
    PS = dataset.patch_stride
    PL = dataset.patch_length
    
    y_prob = y_prob.flatten()
    y_reshaped = y_prob.reshape(len(dataset), P)
    
    prob_ts = np.zeros(len(dataset.times))
    count_ts = np.zeros(len(dataset.times))
    
    for idx, s in enumerate(dataset.window_starts):
        for p in range(P):
            p_start = s + p * PS
            p_end = p_start + PL
            prob_ts[p_start:p_end] += y_reshaped[idx, p]
            count_ts[p_start:p_end] += 1
            
    mask = count_ts > 0
    prob_ts[mask] /= count_ts[mask]
    
    # We define base events at a 0.5 threshold to get the shapes, then assign confidences to them
    pred_bool = prob_ts > 0.5
    raw_pred_events = get_events(pred_bool)
    
    merge_gap = merge_threshold_patches * PL
    pred_events = merge_events(raw_pred_events, merge_gap)
    
    gt_bool = dataset.timestep_labels > 0.5
    gt_events = get_events(gt_bool)
    
    # Assign confidence scores
    pred_confs = []
    for e in pred_events:
        segment = prob_ts[e[0]:e[1]+1]
        if conf_agg == 'max':
            pred_confs.append(np.max(segment))
        elif conf_agg == 'mean':
            pred_confs.append(np.mean(segment))
        elif conf_agg == 'median':
            pred_confs.append(np.median(segment))
        else:
            pred_confs.append(np.max(segment))
            
    # Match events
    G = len(gt_events)
    P_cnt = len(pred_events)
    
    iou_matrix = np.zeros((P_cnt, G))
    for i, p_e in enumerate(pred_events):
        for j, g_e in enumerate(gt_events):
            iou_matrix[i, j] = compute_iou(p_e, g_e)
            
    # For each pred, find best GT
    tp_mask = np.zeros(P_cnt, dtype=bool)
    if G > 0 and P_cnt > 0:
        best_gt_for_pred = np.argmax(iou_matrix, axis=1)
        best_iou_for_pred = np.max(iou_matrix, axis=1)
        
        # Group preds by the GT they overlap most
        for j in range(G):
            candidates = np.where((best_gt_for_pred == j) & (best_iou_for_pred >= iou_threshold))[0]
            if len(candidates) > 0:
                # The one with highest IoU gets TP
                best_candidate = candidates[np.argmax(best_iou_for_pred[candidates])]
                tp_mask[best_candidate] = True

    TP = np.sum(tp_mask)
    FP = P_cnt - TP
    FN = G - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / G if G > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute PRC points
    # Need to sort all predicted events by confidence
    sorted_indices = np.argsort(pred_confs)[::-1]
    sorted_tps = tp_mask[sorted_indices]
    
    cum_tp = np.cumsum(sorted_tps)
    cum_fp = np.cumsum(~sorted_tps)
    
    prc_precision = cum_tp / (cum_tp + cum_fp)
    prc_precision = np.nan_to_num(prc_precision)
    prc_recall = cum_tp / G if G > 0 else np.zeros_like(cum_tp)
    
    # Add (1.0, 0.0) or (0.0, 1.0) points for auc? 
    # Usually standard precision-recall curves add the (recall=0, precision=max) point
    prc_recall = np.concatenate(([0.0], prc_recall))
    prc_precision = np.concatenate(([prc_precision[0] if len(prc_precision) > 0 else 1.0], prc_precision))
    pr_auc = auc(prc_recall, prc_precision) if len(prc_recall) > 1 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "pr_auc": pr_auc,
        "prc_precision": prc_precision,
        "prc_recall": prc_recall,
        "pred_events": pred_events,
        "gt_events": gt_events,
        "pred_confs": pred_confs,
        "prob_ts": prob_ts
    }
