"""Resumable U-Net training. All randomness is derived from (seed, epoch, step), so a run resumed
from a checkpoint is bit-identical to an uninterrupted run (on the same machine/threads)."""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import MEAN, STD, SEED, Deadline, Incomplete, log
from .unet import UNet, n_params


def normalize(xb_uint8):
    x = xb_uint8.permute(0, 3, 1, 2).float() / 255.0
    m = torch.tensor(MEAN).view(1, 3, 1, 1); s = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous(memory_format=torch.channels_last)


def augment(x, y, gen, shift=12):
    """Training-only augmentation policy (disclosed in README):
    horizontal flip p=0.5; random translation up to +-12px (reflect padding, applied to image and mask);
    brightness x U(0.8,1.2) and contrast x U(0.8,1.2) on the image only. x float (B,3,H,W) normalized, y long (B,H,W)."""
    B, _, H, W = x.shape
    xs, ys = [], []
    for i in range(B):
        xi, yi = x[i], y[i]
        if torch.rand(1, generator=gen).item() < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        dx, dy = (torch.randint(-shift, shift + 1, (2,), generator=gen)).tolist()
        if dx or dy:
            p = shift
            xp = F.pad(xi[None], (p, p, p, p), mode="reflect")[0]
            yp = F.pad(yi[None, None].float(), (p, p, p, p), mode="reflect")[0, 0].long()
            xi = xp[:, p + dy:p + dy + H, p + dx:p + dx + W]
            yi = yp[p + dy:p + dy + H, p + dx:p + dx + W]
        b, c = (0.8 + 0.4 * torch.rand(2, generator=gen)).tolist()
        mu = xi.mean()
        xi = (xi - mu) * c + mu * b
        xs.append(xi); ys.append(yi)
    return torch.stack(xs).contiguous(memory_format=torch.channels_last), torch.stack(ys)


def save_ckpt(path, state):
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic: a kill mid-write never corrupts the previous checkpoint


def train(X, Y, ckpt_path, epochs=6, bs=16, lr=1e-3, base=32, save_every=10, seed=SEED, deadline=None,
          log_path=None):
    """X uint8 (N,H,W,3), Y uint8 (N,H,W). Resumes from ckpt_path if present. Raises Incomplete at deadline."""
    deadline = deadline or Deadline(None)
    torch.set_num_threads(int(os.environ.get("PETSEG_THREADS", "1")))
    torch.manual_seed(seed)
    model = UNet(base=base).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X); spe = n // bs; total = epochs * spe   # drop last partial batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, losses, train_seconds = 0, [], 0.0
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, weights_only=False)
        assert (st["epochs"], st["bs"], st["lr"], st["base"], st["seed"], st["n"]) == (epochs, bs, lr, base, seed, n), "config mismatch"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, losses, train_seconds = st["step"], st["losses"], st["train_seconds"]
        log(f"resumed from step {step}/{total}")
    cfg = dict(epochs=epochs, bs=bs, lr=lr, base=base, seed=seed, n=n)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    model.train()
    t0 = time.time(); last_save = step
    while step < total:
        ep, i = divmod(step, spe)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + ep))
        idx = perm[i * bs:(i + 1) * bs]
        gen = torch.Generator().manual_seed(seed * 10_000_000 + step)
        x, y = augment(normalize(Xt[idx]), Yt[idx].long(), gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        step += 1; losses.append(loss.item())
        if step % save_every == 0 or step == total or deadline.hit():
            save_ckpt(ckpt_path, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                      step=step, losses=losses, train_seconds=train_seconds + time.time() - t0, **cfg))
            if step % (save_every * 3) == 0 or step == total:
                log(f"step {step}/{total} epoch {step / spe:.2f} loss(avg last 10) {np.mean(losses[-10:]):.4f}")
        if deadline.hit() and step < total:
            raise Incomplete(f"stopped at step {step}/{total}")
    train_seconds += time.time() - t0
    if log_path:
        json.dump(dict(cfg, params=n_params(model), steps=total, train_seconds=train_seconds,
                       torch=torch.__version__, threads=torch.get_num_threads(), losses=losses), open(log_path, "w"))
    return model
