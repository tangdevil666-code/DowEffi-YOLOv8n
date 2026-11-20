# Ultralytics YOLO 🚀
# SDI: Shallow Detail Injection module
# Function: inject P2/C2 shallow detail into P3 feature without adding P2 detection head.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class SDI(nn.Module):
    """Shallow Detail Injection module.

    This module takes two inputs:
        x[0]: P3 feature, usually 80x80 for 640 input
        x[1]: P2/C2 shallow feature, usually 160x160 for 640 input

    It downsamples P2/C2 to P3 scale, aligns channels, and injects it into P3.

    Structure:
        C2/P2 -> DWConv 3x3 stride=2 -> Conv 1x1 -> aligned shallow detail
        P3_enhanced = P3 + sigmoid(alpha) * aligned_detail

    It does NOT add a P2 detection head.
    """

    def __init__(
        self,
        c_p3: int,
        c_p2: int,
        k: int = 3,
        init_alpha: float = -1.0986,
        refine: bool = False,
        act: bool = True,
    ):
        super().__init__()

        if c_p3 <= 0 or c_p2 <= 0:
            raise ValueError(f"Invalid channels: c_p3={c_p3}, c_p2={c_p2}")

        self.c_p3 = c_p3
        self.c_p2 = c_p2

        # Lightweight downsample: P2/C2 160x160 -> P3 80x80
        self.down = Conv(c_p2, c_p2, k=k, s=2, g=c_p2, act=act)

        # Channel alignment: c_p2 -> c_p3
        self.align = Conv(c_p2, c_p3, k=1, s=1, act=act)

        # Learnable injection strength
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

        # Optional lightweight spatial refinement
        self.refine = Conv(c_p3, c_p3, k=3, s=1, g=c_p3, act=act) if refine else nn.Identity()

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                "SDI expects two inputs: [P3_feature, P2_feature]. "
                f"Got type={type(x)}, len={len(x) if isinstance(x, (list, tuple)) else 'N/A'}."
            )

        p3, p2 = x

        if p3.ndim != 4 or p2.ndim != 4:
            raise RuntimeError(
                f"SDI expects 4D tensors [B, C, H, W], got p3={p3.shape}, p2={p2.shape}."
            )

        if p3.shape[1] != self.c_p3:
            raise RuntimeError(f"SDI P3 channel mismatch: expected {self.c_p3}, got {p3.shape[1]}.")

        if p2.shape[1] != self.c_p2:
            raise RuntimeError(f"SDI P2 channel mismatch: expected {self.c_p2}, got {p2.shape[1]}.")

        detail = self.align(self.down(p2))

        # Robust spatial alignment for non-standard input sizes
        if detail.shape[-2:] != p3.shape[-2:]:
            detail = F.interpolate(detail, size=p3.shape[-2:], mode="nearest")

        gate = torch.sigmoid(self.alpha)
        out = p3 + gate * detail
        out = self.refine(out)

        return out