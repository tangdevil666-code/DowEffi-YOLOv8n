import torch
import torch.nn as nn

from .conv import Conv
from .block import Bottleneck


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
    只处理部分通道，其余通道保留原始信息。
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
    注意：这里不用 x * sigmoid(attn) 直接压制特征，
    而是采用残差增强形式，避免弱小目标被抑制。
    """

    def __init__(self, k=7):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(k, stride=1, padding=k // 2)
        self.spatial = nn.Conv2d(1, 1, k, stride=1, padding=k // 2, bias=True)

        # 初始为 0，训练初期不改变特征
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        local_mean = self.avg_pool(x)
        contrast = torch.abs(x - local_mean)
        contrast = torch.mean(contrast, dim=1, keepdim=True)

        attn = torch.sigmoid(self.spatial(contrast))

        # 残差式注意力：不会直接把特征压到 0~1
        return x * (1.0 + self.gamma * attn)


class DirectionalContextConv(nn.Module):
    """
    方向上下文卷积。
    用于建模玉米苗行向结构、细长叶片和局部方向纹理。
    """

    def __init__(self, c, k=7):
        super().__init__()

        if k % 2 == 0:
            k += 1

        self.dw3 = Conv(c, c, 3, 1, g=c)

        self.dw_h = Conv(c, c, k=(1, k), s=1, g=c, act=False)
        self.dw_v = Conv(c, c, k=(k, 1), s=1, g=c, act=False)

        self.fuse = Conv(c, c, 1, 1)

        # 初始较小，避免方向上下文一开始过强
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        local = self.dw3(x)
        context = self.dw_v(self.dw_h(local))
        out = local + self.beta * context
        out = self.fuse(out)
        return out


class LDCAEnhance(nn.Module):
    """
    LDCA 增强分支。
    只作为 C2f 后面的残差增强，不直接替代主干特征。
    """

    def __init__(self, c, k=7, n_div=4):
        super().__init__()
        self.pconv = PConv(c, n_div=n_div)
        self.dconv = DirectionalContextConv(c, k=k)
        self.lca = LocalContrastAttention(k=7)
        self.proj = Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        y = self.pconv(x)
        y = self.dconv(y)
        y = self.lca(y)
        y = self.proj(y)
        return y


class C2f_ResLDCA(nn.Module):
    """
    Stable residual LDCA-C2f.

    主分支完全保留原始 C2f 结构；
    LDCA 作为残差增强分支；
    alpha 初始很小，保证训练初期接近原始 C2f。

    Args:
        c1: input channels
        c2: output channels
        n: number of Bottlenecks
        shortcut: shortcut in Bottleneck
        k: directional stripe kernel size
        n_div: PConv channel division ratio
        e: expansion ratio
        alpha_init: residual enhancement initial weight
    """

    def __init__(self, c1, c2, n=1, shortcut=False, k=7, n_div=4, e=0.5, alpha_init=1e-3):
        super().__init__()

        self.c = int(c2 * e)

        # 保持和原始 C2f 一样的属性名：cv1、cv2、m
        # 这样更容易加载 yolov8n.pt 中对应 C2f 的预训练权重
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g=1, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        self.enhance = LDCAEnhance(c2, k=k, n_div=n_div)

        # 残差缩放系数，初始极小
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, dim=1))

        # 残差增强
        out = out + self.alpha * self.enhance(out)

        return out

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, dim=1))
        out = out + self.alpha * self.enhance(out)
        return out