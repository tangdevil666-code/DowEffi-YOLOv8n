"""
SPD-Conv 及其动态变体
- SPDConv: Space-to-Depth Conv, 无损下采样 (Sunkara & Luo, 2022)
- DFFSPDConv: Dynamic Feature Focusing SPD, 用学习的空间 mask 动态决定
              每个区域走"细节分支"还是"轻量分支", 训练时配合稀疏正则可节省 FLOPs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SPDConv", "DFFSPDConv")


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution.

    将空间维度信息重排到通道维度 (c -> c*s*s, h,w -> h/s, w/s),
    再用一个 stride=1 的卷积压回 c2 通道。相比 stride=2 卷积无信息丢失,
    对小目标特别友好。

    Args:
        c1 (int): 输入通道数 (parse_model 自动传入)
        c2 (int): 输出通道数
        k  (int): 卷积核大小, 默认 3
        scale (int): 空间->深度缩放因子, 默认 2 (即 2x 下采样)
    """

    def __init__(self, c1, c2, k=3, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * scale * scale, c2, k, 1, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        # 不能整除时右下角 pad 零
        if h % s or w % s:
            pad_h = (s - h % s) % s
            pad_w = (s - w % s) % s
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w
        # Space-to-Depth: (B, C, H, W) -> (B, C*s*s, H/s, W/s)
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * s * s, h // s, w // s)
        return self.conv(x)


class DFFSPDConv(nn.Module):
    """
    Dynamic Feature Focusing SPD-Conv.

    双分支 + 学习的空间 gate:
      - detail 分支: 标准 SPD-Conv, 保留全部信息
      - light  分支: AvgPool + 1x1 conv, 廉价处理平滑区域
      - gate: 预测每个输出位置的"细节得分", Gumbel-Sigmoid + STE 二值化

    训练时需在 loss 里读取 self.sparsity 并加入稀疏正则,
    否则 mask 会全部学到 1, 退化为普通 SPD。

    Args:
        c1, c2: 同上
        k:      detail 分支卷积核, 默认 3
        scale:  下采样倍数, 默认 2
        tau:    Gumbel-Sigmoid 温度, 越小越接近硬 0/1
        hard:   True 用直通估计输出硬 mask, False 用软 mask
        mask_init_bias: gate 最后一层 bias 初值, 负数 -> 初始偏稀疏
    """

    def __init__(self, c1, c2, k=3, scale=2, tau=1.0, hard=True, mask_init_bias=-1.0):
        super().__init__()
        self.scale = scale
        self.tau = tau
        self.hard = hard

        # detail: 标准 SPD
        self.detail = nn.Sequential(
            nn.Conv2d(c1 * scale * scale, c2, k, 1, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # light: 廉价分支, FLOPs ≈ detail 的 1/(s*s*k*k)
        self.light = nn.Sequential(
            nn.AvgPool2d(scale, scale),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # gate: 输出下采样分辨率上的 1 通道 logits
        hidden = max(c1 // 4, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(c1, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.AvgPool2d(scale, scale),
            nn.Conv2d(hidden, 1, 1, 1, 0),
        )
        nn.init.constant_(self.gate[-1].bias, mask_init_bias)

        # 给外部 loss 读取的稀疏度
        self.sparsity = None

    def _gumbel_sigmoid(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            noise = torch.log(u) - torch.log1p(-u)  # logistic noise
            soft = torch.sigmoid((logits + noise) / self.tau)
        else:
            soft = torch.sigmoid(logits / self.tau)
        if self.hard:
            hard = (soft > 0.5).float()
            return hard + soft - soft.detach()  # straight-through
        return soft

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        if h % s or w % s:
            pad_h = (s - h % s) % s
            pad_w = (s - w % s) % s
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w

        # gate mask: (B, 1, H/s, W/s)
        mask = self._gumbel_sigmoid(self.gate(x))
        self.sparsity = mask.mean().detach

        # detail 分支
        x_d = (
            x.view(b, c, h // s, s, w // s, s)
             .permute(0, 1, 3, 5, 2, 4).contiguous()
             .view(b, c * s * s, h // s, w // s)
        )
        out_d = self.detail(x_d)

        # light 分支
        out_l = self.light(x)

        return mask * out_d + (1.0 - mask) * out_l