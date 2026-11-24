import numpy as np, torch, pytest
from petseg.train import train
from petseg.common import Deadline, Incomplete


def toy(n=24, s=32, seed=1):
    r = np.random.default_rng(seed)
    return r.integers(0, 256, (n, s, s, 3), dtype=np.uint8), r.integers(0, 3, (n, s, s), dtype=np.uint8)


KW = dict(epochs=2, bs=4, base=8, save_every=2)


def test_resume_is_bit_identical(tmp_path):
    X, Y = toy()
    full = train(X, Y, tmp_path / "full.pt", **KW)
    # interrupt after 5 steps (deadline already hit => stops at first checkpoint boundary check), then resume
    class StopAt:
        def __init__(self): self.calls = 0
        def hit(self): self.calls += 1; return self.calls > 5
    with pytest.raises(Incomplete):
        train(X, Y, tmp_path / "r.pt", deadline=StopAt(), **KW)
    st = torch.load(tmp_path / "r.pt", weights_only=False)
    assert 0 < st["step"] < 12
    resumed = train(X, Y, tmp_path / "r.pt", **KW)
    for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), k


def test_resume_rejects_config_change(tmp_path):
    X, Y = toy()
    train(X, Y, tmp_path / "c.pt", **KW)
    with pytest.raises(AssertionError):
        train(X, Y, tmp_path / "c.pt", **{**KW, "epochs": 3})


def test_no_partial_checkpoint_left(tmp_path):
    X, Y = toy()
    train(X, Y, tmp_path / "c.pt", **KW)
    assert not (tmp_path / "c.pt.tmp").exists()
