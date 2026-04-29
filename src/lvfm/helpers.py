import numpy as np

def compute_metrics(V_pred, V_gt):
    mae = np.mean(np.abs(V_pred - V_gt))

    pred_set = V_pred <= 0.0
    gt_set = V_gt <= 0.0

    intersection = np.logical_and(pred_set, gt_set).sum()
    union = np.logical_or(pred_set, gt_set).sum()
    iou = intersection / union if union > 0 else 1.0

    return float(mae), float(iou)

def squeeze_last(x):
    return x.squeeze(-1) if x.ndim > 1 and x.shape[-1] == 1 else x