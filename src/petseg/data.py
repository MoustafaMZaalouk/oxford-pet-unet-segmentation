"""Dataset download (checksum-pinned), preprocessing to 128px uint8 caches, and the official split.

The Oxford-IIIT Pet images are CC BY-SA 4.0 (copyright remains with original owners); they are
downloaded at build time into ./data (git-ignored) and are never committed or redistributed.
"""
import io, tarfile, urllib.request
import numpy as np
from PIL import Image
from .common import DATA, SIZE, sha256_file, log

URLS = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
}
SHA256 = {
    "images.tar.gz": "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
    "annotations.tar.gz": "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
}
EXPECTED = {"trainval": 3680, "test": 3669}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        p = DATA / name
        if not p.exists():
            log("downloading", url)
            urllib.request.urlretrieve(url, str(p) + '.part')
            (DATA / (name + '.part')).replace(p)   # never leave a partial file under the final name
        got = sha256_file(p)
        if got != SHA256[name]:
            raise RuntimeError(f"checksum mismatch for {name}: {got}")
        log("ok", name, got[:12])


def read_split(lines, name):
    out = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        out.append(ln.split()[0])
    return out


def _resize_img(im):
    return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def _resize_mask(im):
    m = np.asarray(im.resize((SIZE, SIZE), Image.NEAREST), dtype=np.uint8)
    assert m.min() >= 1 and m.max() <= 3
    return m - 1   # 0 pet, 1 background, 2 border


def prepare():
    """Build data/cache/{trainval,test}_{x,y}.npy and _names.txt from the verified tarballs."""
    download()
    cache = DATA / "cache"; cache.mkdir(exist_ok=True)
    if all((cache / f"{s}_y.npy").exists() for s in EXPECTED):
        return
    with tarfile.open(DATA / "annotations.tar.gz") as t:
        splits = {s: read_split(t.extractfile(f"annotations/{s}.txt").read().decode().splitlines(), s) for s in EXPECTED}
        masks = {}
        for m in t.getmembers():
            if m.name.startswith("annotations/trimaps/") and m.name.endswith(".png") and "/._" not in m.name:
                masks[m.name.split("/")[-1][:-4]] = _resize_mask(Image.open(io.BytesIO(t.extractfile(m).read())))
    for s, n in EXPECTED.items():
        assert len(splits[s]) == n, (s, len(splits[s]))
    assert not set(splits["trainval"]) & set(splits["test"])
    wanted = {n for v in splits.values() for n in v}
    imgs = {}
    with tarfile.open(DATA / "images.tar.gz") as t:
        for m in t:
            if m.name.endswith(".jpg") and "/._" not in m.name:
                k = m.name.split("/")[-1][:-4]
                if k in wanted:
                    imgs[k] = _resize_img(Image.open(io.BytesIO(t.extractfile(m).read())))
    for s, names in splits.items():
        np.save(cache / f"{s}_x.npy", np.stack([imgs[n] for n in names]))
        np.save(cache / f"{s}_y.npy", np.stack([masks[n] for n in names]))
        (cache / f"{s}_names.txt").write_text("\n".join(names))
        log("cached", s, len(names))


def load(split):
    c = DATA / "cache"
    return (np.load(c / f"{split}_x.npy"), np.load(c / f"{split}_y.npy"),
            (c / f"{split}_names.txt").read_text().split("\n"))
