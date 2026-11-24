"""Chunked, resumable inference + metrics for the U-Net and the linear-probe baseline."""
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .common import DATA, RESULTS, CLASSES, Deadline, Incomplete, atomic_save_npy, sha256_file, log
from .metrics import confusion, scores, bootstrap
from .train import normalize

RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
CHUNK = 128


def chunked(tag, n, fn, deadline):
    """Run fn(lo, hi) -> ndarray for each chunk, caching to data/work/<tag>_<i>.npy; resumable."""
    d = DATA / "work"; d.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, lo in enumerate(range(0, n, CHUNK)):
        p = d / f"{tag}_{i:04d}.npy"
        if not p.exists():
            if fn is None:
                raise FileNotFoundError(p)
            if deadline.hit():
                raise Incomplete(f"{tag}: stopped at chunk {i}")
            atomic_save_npy(p, fn(lo, min(n, lo + CHUNK)))
        parts.append(np.load(p))
    return np.concatenate(parts)


def resnet18():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    f = os.path.join(torch.hub.get_dir(), "checkpoints", "resnet18-f37072fd.pth")
    assert sha256_file(f) == RESNET18_SHA256, "ResNet-18 weights checksum mismatch"
    return m.eval()


def feats_layer3(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    return m.layer3(m.layer2(m.layer1(x)))   # (B,256,H/16,W/16)


class LinearProbe(nn.Module):
    """Per-pixel linear classifier: 1x1 conv on frozen ResNet-18 layer3 features (stride 16),
    logits bilinearly upsampled to the input size (equivalent to a linear map on upsampled features)."""
    def __init__(self, c=256, k=3):
        super().__init__(); self.fc = nn.Conv2d(c, k, 1)
    def forward(self, f, size):
        return F.interpolate(self.fc(f), size=size, mode="bilinear", align_corners=False)


def extract_features(split, X, deadline):
    m = resnet18()
    def fn(lo, hi):
        with torch.no_grad():
            return feats_layer3(m, normalize(torch.from_numpy(X[lo:hi]))).half().numpy()
    return chunked(f"feat_{split}", len(X), fn, deadline)


def train_probe(Ftr, Ytr, epochs=10, bs=64, lr=1e-2, seed=0):
    """Frozen features, no augmentation (features are cached). Fast; re-run from scratch if interrupted."""
    torch.manual_seed(seed)
    probe = LinearProbe(); opt = torch.optim.Adam(probe.parameters(), lr=lr)
    Ft, Yt = torch.from_numpy(Ftr), torch.from_numpy(Ytr)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(Ft), generator=torch.Generator().manual_seed(seed * 1000 + ep))
        for i in range(0, len(Ft) - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(probe(Ft[idx].float(), Yt.shape[-2:]), Yt[idx].long())
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        log(f"probe epoch {ep + 1}/{epochs} loss {np.mean(losses[-50:]):.4f}")
    return probe, losses


def per_image_cms(tag, preds_fn, Y, deadline):
    def fn(lo, hi):
        p = preds_fn(lo, hi)
        return np.stack([confusion(p[j], Y[lo + j]) for j in range(hi - lo)])
    return chunked(tag, len(Y), fn, deadline)


def summarize(cm_img, other=None):
    iou, miou, acc = scores(cm_img.sum(0))
    ci = bootstrap(cm_img, 1000, 0, other)
    out = {"per_class_iou": {c: float(v) for c, v in zip(CLASSES, iou)}, "miou": miou, "pixel_acc": acc,
           "ci95": {k: list(v) for k, v in ci.items()}, "n_images": int(len(cm_img))}
    return out
