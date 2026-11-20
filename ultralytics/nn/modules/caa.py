import torch
import torch.nn as nn

from .conv import Conv


class CAA(nn.Module):
    """
    Context Anchor Attention for YOLOv8.

    Args:
        c1: input channels
        c2: output channels, normally same as c1
        stage_idx: network stage index, kernel_size = 11 + 2 * stage_idx
        kernel_size: manually specified stripe kernel size
        avg_kernel_size: average pooling kernel size
    """

    def __init__(self, c1, c2=None, stage_idx=3, kernel_size=None, avg_kernel_size=7):
        super().__init__()

        c2 = c1 if c2 is None else c2

        if kernel_size is None:
            kernel_size = 11 + 2 * stage_idx

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        self.avg_pool = nn.AvgPool2d(
            kernel_size=avg_kernel_size,
            stride=1,
            padding=avg_kernel_size // 2
        )

        self.conv1 = Conv(c2, c2, 1, 1)

        self.h_conv = Conv(
            c2,
            c2,
            k=(1, kernel_size),
            s=1,
            g=c2,
            act=False
        )

        self.v_conv = Conv(
            c2,
            c2,
            k=(kernel_size, 1),
            s=1,
            g=c2,
            act=False
        )

        self.conv2 = Conv(c2, c2, 1, 1, act=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)

        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        attn = self.act(attn)

        return x * attn