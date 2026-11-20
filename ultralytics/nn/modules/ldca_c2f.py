import torch
import torch.nn as nn

from .conv import Conv


def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.reshape(b, c, h, w)
    return x


class PConv(nn.Module):
    """
    Partial Convolution.
    只对部分通道做 3×3 卷积，其余通道保留原始信息。
    适合小目标细节保留和轻量化。
    """

    def __init__(self, c, n_div=4):
        super().__init__()
        self.dim_conv = max(1, c // n_div)
        self.dim_untouched = c - self.dim_conv

        self.partial_conv = nn.Sequential(
            nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim_conv),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        out = torch.cat((x1, x2), dim=1)
        out = channel_shuffle(out, groups=2)
        return out


class LocalContrastAttention(nn.Module):
    """
    局部对比注意力。
    针对阴天、湿土、低曝光图像中玉米苗与背景对比度低的问题。
    """

    def __init__(self, k=7):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(k, stride=1, padding=k // 2)
        self.spatial = nn.Conv2d(1, 1, k, stride=1, padding=k // 2, bias=True)

        # 初始为 0，训练初期接近原始网络，避免注意力扰动过大
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        local_mean = self.avg_pool(x)
        contrast = torch.abs(x - local_mean)
        contrast = torch.mean(contrast, dim=1, keepdim=True)

        attn = torch.sigmoid(self.spatial(contrast))

        return x * (1.0 + self.gamma * attn)


class DirectionalContextConv(nn.Module):
    """
    方向上下文卷积。
    使用 3×3 DWConv + 1×k DWConv + k×1 DWConv，
    用于建模玉米苗行向、叶片方向和局部上下文。
    """

    def __init__(self, c, k=7):
        super().__init__()

        if k % 2 == 0:
            k += 1

        self.dw3 = Conv(c, c, 3, 1, g=c)

        self.dw_h = Conv(c, c, k=(1, k), s=1, g=c, act=False)
        self.dw_v = Conv(c, c, k=(k, 1), s=1, g=c, act=False)

        self.fuse = Conv(c, c, 1, 1)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        local = self.dw3(x)
        context = self.dw_v(self.dw_h(local))
        out = local + self.beta * context
        out = self.fuse(out)
        return out


class LDCABottleneck(nn.Module):
    """
    LDCA Bottleneck:
    PConv + Directional Context Conv + Local Contrast Attention.
    """

    def __init__(self, c1, c2, shortcut=True, k=7, n_div=4, e=1.0):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.pconv = PConv(c_, n_div=n_div)
        self.dconv = DirectionalContextConv(c_, k=k)
        self.lca = LocalContrastAttention(k=7)
        self.cv2 = Conv(c_, c2, 1, 1)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y = self.pconv(y)
        y = self.dconv(y)
        y = self.lca(y)
        y = self.cv2(y)

        return x + y if self.add else y


class C2f_LDCA(nn.Module):
    """
    LDCA-C2f for maize seedling small object detection.

    Args:
        c1: input channels
        c2: output channels
        n: number of bottlenecks
        shortcut: whether to use residual connection
        k: directional stripe kernel size
        n_div: PConv channel division ratio
        e: expansion ratio
    """

    def __init__(self, c1, c2, n=1, shortcut=False, k=7, n_div=4, e=0.5):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        self.m = nn.ModuleList(
            LDCABottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                k=k,
                n_div=n_div,
                e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))