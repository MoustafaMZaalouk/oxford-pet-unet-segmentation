import hashlib, os, time, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("PETSEG_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("PETSEG_DATA", ROOT / "data"))      # downloads + caches (git-ignored)
RESULTS = Path(os.environ.get("PETSEG_RESULTS", ROOT / "results"))
CKPT = Path(os.environ.get("PETSEG_CKPT", ROOT / "checkpoints"))  # git-ignored
CLASSES = ["pet", "background", "border"]   # trimap 1,2,3 -> 0,1,2
SIZE = 128
SEED = 0
MEAN = (0.485, 0.456, 0.406)   # ImageNet constants (fixed, not fit on any split)
STD = (0.229, 0.224, 0.225)


class Deadline:
    def __init__(self, seconds=None):
        self.t_end = None if not seconds else time.time() + seconds
    def hit(self):
        return self.t_end is not None and time.time() >= self.t_end


class Incomplete(Exception):
    """Raised when a stage stopped at its time budget; re-run to resume."""


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def atomic_save_npy(path, arr):
    path = Path(path); tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr); os.replace(tmp, path)


def log(*a):
    print(*a, flush=True)
