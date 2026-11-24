"""Builds results/metrics.json, results/results.md, figures, and README.md (from README.template.md)
purely from committed artifacts + the cached test split. Every README number comes from metrics.json."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, RESULTS, CKPT, CLASSES, DATA, SIZE, log
from .metrics import scores
from . import evaluate as E
from . import data as D
from .train import normalize
from .unet import UNet

COLORS = np.array([[230, 60, 60], [40, 40, 40], [240, 220, 60]], dtype=np.uint8)  # pet, background, border


def fmt(v, ci=None, pct=True):
    s = f"{100 * v:.1f}" if pct else f"{v:.3f}"
    if ci:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return s


def compute_metrics():
    u = np.load(RESULTS / "per_image_confusion_unet.npy"); p = np.load(RESULTS / "per_image_confusion_probe.npy")
    m = {"unet": E.summarize(u, other=p), "probe": E.summarize(p)}
    m["unet_minus_probe_miou_ci95"] = m["unet"]["ci95"].pop("miou_diff")
    m["unet_minus_probe_miou"] = m["unet"]["miou"] - m["probe"]["miou"]
    tl = json.load(open(RESULTS / "train_log.json"))
    m["train"] = {k: tl[k] for k in ("epochs", "bs", "lr", "base", "seed", "params", "steps", "train_seconds", "torch", "threads")}
    m["train"]["final_loss_avg_last_50"] = float(np.mean(tl["losses"][-50:]))
    m["probe_params"] = json.load(open(RESULTS / "probe_train_log.json"))["params"]
    m["n_test_images"] = int(len(u)); m["n_trainval_images"] = int(m["train"]["steps"] // m["train"]["epochs"] * m["train"]["bs"])
    json.dump(m, open(RESULTS / "metrics.json", "w"), indent=1)
    return m, u, p


def overlay(x, y):
    return COLORS[y]


def panel(path, idxs, X, Y, names, unet, probe, Fte, title):
    fig, ax = plt.subplots(len(idxs), 3, figsize=(6, 2 * len(idxs)))
    for r, i in enumerate(idxs):
        with torch.no_grad():
            pu = unet(normalize(torch.from_numpy(X[i:i + 1]))).argmax(1)[0].numpy()
            pp = probe(torch.from_numpy(Fte[i:i + 1]).float(), (SIZE, SIZE)).argmax(1)[0].numpy()
        for c, (im, t) in enumerate([(overlay(X[i], Y[i]), names[i] + " ground truth"), (overlay(X[i], pu), "U-Net"), (overlay(X[i], pp), "probe")]):
            ax[r, c].imshow(im); ax[r, c].axis("off"); ax[r, c].set_title(t, fontsize=6)
    fig.suptitle(title + "\n(masks only, no photos; red=pet, dark=background, yellow=border)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def make():
    m, u, p = compute_metrics()
    Xt, Yt, names = D.load("test")
    st = torch.load(CKPT / "unet.pt", weights_only=False)
    unet = UNet(base=st["base"]).to(memory_format=torch.channels_last); unet.load_state_dict(st["model"]); unet.eval()
    probe = E.LinearProbe(); probe.load_state_dict(torch.load(CKPT / "probe.pt", weights_only=True)); probe.eval()
    Fte = E.chunked("feat_test", len(Xt), None, None)
    # per-image mean IoU over classes present (for ranking failures only)
    def img_miou(cm):
        tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1) - tp
        ok = den > 0
        return (tp[ok] / den[ok]).mean()
    scores_u = np.array([img_miou(c) for c in u])
    worst = list(np.argsort(scores_u)[:8])
    rng = np.random.default_rng(0); rand = list(rng.choice(len(Xt), 8, replace=False))
    panel(RESULTS / "failure_panel_unet_worst8.png", worst, Xt, Yt, names, unet, probe, Fte, "U-Net: 8 lowest per-image mIoU on test")
    panel(RESULTS / "random_panel_seed0.png", rand, Xt, Yt, names, unet, probe, Fte, "8 random test images (seed 0, not cherry-picked)")
    # Label-noise check: some raw trimaps contain no pet/border pixels at all (all background)
    empty = u[:, 0, :].sum(1) + u[:, 2, :].sum(1) == 0   # gt rows: pet(0), border(2) have zero pixels
    keep = ~empty
    m["n_test_empty_gt"] = int(empty.sum())
    m["excl_empty_gt"] = {"n_images": int(keep.sum())}
    for nm, cm in (("unet", u), ("probe", p)):
        iou, miou, acc = scores(cm[keep].sum(0))
        m["excl_empty_gt"][nm] = {"miou": float(miou), "pixel_acc": float(acc), "per_class_iou": {c: float(v) for c, v in zip(CLASSES, iou)}}
    worst_nonempty = [i for i in np.argsort(scores_u) if keep[i]][:8]
    panel(RESULTS / "failure_panel_unet_worst8_nonempty_gt.png", worst_nonempty, Xt, Yt, names, unet, probe, Fte, "U-Net: 8 lowest per-image mIoU among test images with non-empty ground truth")
    m["failure_images_unet_worst8_nonempty_gt"] = [names[i] for i in worst_nonempty]
    m["n_worst8_with_empty_gt"] = int(sum(empty[i] for i in worst))
    m["failure_images_unet_worst8"] = [names[i] for i in worst]
    m["worst8_unet_per_image_miou"] = [float(scores_u[i]) for i in worst]
    m["median_per_image_miou_unet"] = float(np.median(scores_u))
    json.dump(m, open(RESULTS / "metrics.json", "w"), indent=1)
    tl = json.load(open(RESULTS / "train_log.json"))["losses"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(np.convolve(tl, np.ones(10) / 10, mode="valid")); ax[0].set_xlabel("step"); ax[0].set_ylabel("train CE (10-step avg)"); ax[0].set_title("U-Net training loss")
    w = 0.38; xs = np.arange(4)
    for k, (nm, off) in enumerate([("unet", -w / 2), ("probe", w / 2)]):
        vals = [m[nm]["per_class_iou"][c] for c in CLASSES] + [m[nm]["miou"]]
        errs = np.array([[m[nm]["ci95"][f"iou_{j}"][0], m[nm]["ci95"][f"iou_{j}"][1]] for j in range(3)] + [m[nm]["ci95"]["miou"]]).T
        ax[1].bar(xs + off, vals, w, yerr=np.abs(errs - np.array(vals)), label=nm, capsize=2)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(CLASSES + ["mIoU"]); ax[1].legend(); ax[1].set_title("Test IoU (95% image-level bootstrap CI)")
    fig.tight_layout(); fig.savefig(RESULTS / "iou_and_loss.png", dpi=110); plt.close(fig)
    write_tables(m); build_readme(m)


def write_tables(m):
    rows = ["| Model | pet IoU | background IoU | border IoU | mIoU | pixel acc |", "|---|---|---|---|---|---|"]
    for nm, label in [("unet", "U-Net from scratch (7.85M params)"), ("probe", "Frozen ResNet-18 (ImageNet) + linear probe (771 params)")]:
        r = m[nm]; ci = r["ci95"]
        rows.append(f"| {label} | " + " | ".join(fmt(r["per_class_iou"][c], ci[f"iou_{j}"]) for j, c in enumerate(CLASSES)) +
                    f" | {fmt(r['miou'], ci['miou'])} | {fmt(r['pixel_acc'], ci['pixel_acc'])} |")
    (RESULTS / "results.md").write_text("\n".join(rows) + "\n")


def build_readme(m):
    t = m["train"]; d = m["unet_minus_probe_miou_ci95"]
    vals = {
        "RESULTS_TABLE": (RESULTS / "results.md").read_text().strip(),
        "UNET_PARAMS": f"{t['params']:,}", "PROBE_PARAMS": f"{m['probe_params']:,}",
        "EPOCHS": str(t["epochs"]), "STEPS": str(t["steps"]), "BS": str(t["bs"]),
        "TRAIN_MIN": f"{t['train_seconds'] / 60:.0f}", "N_TV": str(m["n_trainval_images"]), "N_TEST": str(m["n_test_images"]),
        "DIFF": f"{100 * m['unet_minus_probe_miou']:+.1f}", "DIFF_LO": f"{100 * d[0]:+.1f}", "DIFF_HI": f"{100 * d[1]:+.1f}",
        "FINAL_LOSS": f"{t['final_loss_avg_last_50']:.3f}", "MEDIAN_IMG_MIOU": f"{100 * m['median_per_image_miou_unet']:.1f}",
        "WORST_MIOU": f"{100 * m['worst8_unet_per_image_miou'][0]:.1f}",
        "TORCH": t["torch"],
        "N_EMPTY": str(m["n_test_empty_gt"]), "N_KEEP": str(m["excl_empty_gt"]["n_images"]),
        "N_W8_EMPTY": str(m["n_worst8_with_empty_gt"]),
        "EX_UNET": f"{100 * m['excl_empty_gt']['unet']['miou']:.1f}", "EX_PROBE": f"{100 * m['excl_empty_gt']['probe']['miou']:.1f}",
        "WINNER": ("The U-Net scores higher than the probe on mIoU and the paired 95% CI excludes zero."
                   if d[0] > 0 else ("The probe scores higher than the U-Net on mIoU and the paired 95% CI excludes zero."
                                     if d[1] < 0 else "The paired 95% CI for the mIoU difference includes zero, so this run does not separate the two models.")),
    }
    s = (ROOT / "README.template.md").read_text()
    for k, v in vals.items():
        s = s.replace("{{" + k + "}}", v)
    assert "{{" not in s, "unfilled placeholder"
    (ROOT / "README.md").write_text(s)
    log("README.md written")
