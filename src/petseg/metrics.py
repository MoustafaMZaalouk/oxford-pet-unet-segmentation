import numpy as np

def confusion(pred, gt, k=3):
    """pred, gt: (H,W) int arrays. Returns kxk matrix, rows=gt, cols=pred."""
    return np.bincount(gt.reshape(-1).astype(np.int64) * k + pred.reshape(-1), minlength=k * k).reshape(k, k)

def scores(cm):
    """cm: (k,k) summed confusion. Returns per-class IoU, mIoU, pixel accuracy."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return iou, float(iou.mean()), float(tp.sum() / cm.sum())

def bootstrap(per_image_cm, n_boot=1000, seed=0, other=None):
    """Image-level bootstrap. per_image_cm: (N,k,k). Resamples images with replacement.
    If `other` (N,k,k) is given, the same resamples are used and the paired difference
    (per_image_cm minus other) of mIoU is also returned. Returns dict of (lo, hi) 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    n, k, _ = per_image_cm.shape
    res, diff = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        iou, miou, acc = scores(per_image_cm[idx].sum(0))
        res.append([*iou, miou, acc])
        if other is not None:
            diff.append(miou - scores(other[idx].sum(0))[1])
    res = np.array(res)
    names = [f"iou_{i}" for i in range(k)] + ["miou", "pixel_acc"]
    out = {nm: (float(np.percentile(res[:, j], 2.5)), float(np.percentile(res[:, j], 97.5))) for j, nm in enumerate(names)}
    if other is not None:
        out["miou_diff"] = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    return out
