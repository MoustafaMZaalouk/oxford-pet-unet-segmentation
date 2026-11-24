"""Small U-Net implemented from scratch (no pretrained weights)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net, widths (b, 2b, 4b, 8b, 16b); bilinear upsampling. b=32 gives ~7.8M params."""

    def __init__(self, n_classes=3, base=32, in_ch=3):
        super().__init__()
        w = [base * m for m in (1, 2, 4, 8, 16)]
        self.enc = nn.ModuleList([_block(in_ch, w[0])] + [_block(w[i], w[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([_block(w[i + 1] + w[i], w[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(w[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for blk in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([s, x], 1))
        return self.head(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
