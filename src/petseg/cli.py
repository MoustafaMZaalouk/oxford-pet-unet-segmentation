import argparse, json, sys, time
import numpy as np
import torch
from .common import *
from . import data as D
from .train import train, normalize
from .unet import UNet, n_params
from . import evaluate as E


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "train", "probe", "eval", "report", "all"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop at next checkpoint/chunk after this; re-run resumes (exit 3)")
    a = ap.parse_args(argv)
    dl = Deadline(a.max_seconds)
    RESULTS.mkdir(exist_ok=True, parents=True); CKPT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    stages = ["prep", "train", "probe", "eval", "report"] if a.stage == "all" else [a.stage]
    try:
        for s in stages:
            log("== stage", s); globals()["stage_" + s](a, dl)
    except Incomplete as e:
        log("INCOMPLETE:", e, "- re-run the same command to resume"); return 3
    return 0


def stage_prep(a, dl):
    D.prepare()


def stage_train(a, dl):
    X, Y, _ = D.load("trainval")
    train(X, Y, CKPT / "unet.pt", epochs=a.epochs, deadline=dl, log_path=RESULTS / "train_log.json")


def stage_probe(a, dl):
    X, Y, _ = D.load("trainval"); Xt, Yt, _ = D.load("test")
    Ftr = E.extract_features("trainval", X, dl); Fte = E.extract_features("test", Xt, dl)
    if not (RESULTS / "probe_train_log.json").exists():
        probe, losses = E.train_probe(Ftr, Y)
        torch.save(probe.state_dict(), CKPT / "probe.pt")
        json.dump({"losses": losses, "params": n_params(probe)}, open(RESULTS / "probe_train_log.json", "w"))
    probe = E.LinearProbe(); probe.load_state_dict(torch.load(CKPT / "probe.pt", weights_only=True)); probe.eval()
    def preds(lo, hi):
        with torch.no_grad():
            return probe(torch.from_numpy(Fte[lo:hi]).float(), (SIZE, SIZE)).argmax(1).numpy()
    cm = E.per_image_cms("cm_probe", preds, Yt, dl)
    atomic_save_npy(RESULTS / "per_image_confusion_probe.npy", cm)


def stage_eval(a, dl):
    Xt, Yt, _ = D.load("test")
    st = torch.load(CKPT / "unet.pt", weights_only=False)
    assert st["step"] == (len(D.load("trainval")[0]) // st["bs"]) * st["epochs"], "training not finished"
    model = UNet(base=st["base"]).to(memory_format=torch.channels_last); model.load_state_dict(st["model"]); model.eval()
    def preds(lo, hi):
        with torch.no_grad():
            # sub-batches of 16 keep peak RAM well under 1GB
            return np.concatenate([model(normalize(torch.from_numpy(Xt[i:min(hi, i + 16)]))).argmax(1).numpy() for i in range(lo, hi, 16)])
    cm = E.per_image_cms("cm_unet", preds, Yt, dl)
    atomic_save_npy(RESULTS / "per_image_confusion_unet.npy", cm)
    # predictions of both models for failure panels (cheap: reuse saved per-image confusions to rank, recompute worst few later)


def stage_report(a, dl):
    from . import report
    report.make()
