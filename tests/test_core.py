import re, json, pathlib
import numpy as np, torch
from petseg.unet import UNet, n_params
from petseg.metrics import confusion, scores, bootstrap
from petseg.train import augment, normalize
from petseg import data as D

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_unet_param_budget_and_shape():
    m = UNet(base=32)
    assert 7_000_000 < n_params(m) < 9_000_000
    assert m(torch.zeros(2, 3, 128, 128)).shape == (2, 3, 128, 128)


def test_metrics_known_values():
    gt = np.array([[0, 0], [1, 2]]); pred = np.array([[0, 1], [1, 2]])
    iou, miou, acc = scores(confusion(pred, gt))
    assert np.allclose(iou, [0.5, 0.5, 1.0]) and acc == 0.75


def test_bootstrap_contains_point_estimate_and_is_seeded():
    r = np.random.default_rng(0)
    cms = np.stack([confusion(r.integers(0, 3, (8, 8)), r.integers(0, 3, (8, 8))) for _ in range(60)])
    ci = bootstrap(cms, 200, 1); ci2 = bootstrap(cms, 200, 1)
    assert ci == ci2
    miou = scores(cms.sum(0))[1]
    assert ci["miou"][0] <= miou <= ci["miou"][1]


def test_split_parsing_and_disjointness():
    tv = D.read_split(["# header", "Abyssinian_1 1 1 1", "Bengal_2 2 1 1"], "trainval")
    te = D.read_split(["Birman_3 3 1 1"], "test")
    assert tv == ["Abyssinian_1", "Bengal_2"] and not set(tv) & set(te)


def test_real_cache_split_is_disjoint_if_present():
    c = D.DATA / "cache"
    if not (c / "test_names.txt").exists():
        return
    a = set((c / "trainval_names.txt").read_text().split("\n")); b = set((c / "test_names.txt").read_text().split("\n"))
    assert len(a) == 3680 and len(b) == 3669 and not a & b


def test_augmentation_keeps_mask_aligned_and_is_train_only():
    x = torch.randn(4, 3, 32, 32); y = torch.randint(0, 3, (4, 32, 32))
    gen = torch.Generator().manual_seed(0)
    xa, ya = augment(x, y, gen, shift=4)
    assert xa.shape == x.shape and ya.shape == y.shape and set(ya.unique().tolist()) <= {0, 1, 2}
    # evaluation path in cli.py uses normalize() only
    src = (ROOT / "src/petseg/cli.py").read_text()
    eval_body = src.split("def stage_eval")[1].split("def stage_report")[0]
    assert "augment" not in eval_body and "trainval" not in eval_body.replace('D.load("trainval")[0]', "")


def test_train_code_never_reads_test_split():
    for f in ["train.py", "unet.py"]:
        assert "test" not in (ROOT / "src/petseg" / f).read_text().lower().replace("tests", "")
    cli = (ROOT / "src/petseg/cli.py").read_text()
    assert 'D.load("test")' not in cli.split("def stage_train")[1].split("def stage_probe")[0]


def test_normalization_constants_are_fixed_imagenet():
    from petseg.common import MEAN, STD
    assert MEAN == (0.485, 0.456, 0.406) and STD == (0.229, 0.224, 0.225)


def test_no_dataset_files_tracked():
    gi = (ROOT / ".gitignore").read_text()
    assert "data/" in gi and "checkpoints/" in gi


def test_readme_numbers_match_metrics_json():
    mj = ROOT / "results/metrics.json"
    if not mj.exists():
        return
    m = json.load(open(mj)); readme = (ROOT / "README.md").read_text()
    assert f"{100 * m['unet']['miou']:.1f}" in readme and f"{100 * m['probe']['miou']:.1f}" in readme
    assert f"{m['train']['params']:,}" in readme
