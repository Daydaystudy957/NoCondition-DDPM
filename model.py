"""3D UNet（用于体素扩散模型）。

参考原 DiffusionMP/Model.py 的整体结构（TimeEmbedding + down/middle/up + tail），
但修正了原代码的若干错误：ResBlock 里 stride=2 导致的尺寸塌缩、BatchNorm 通道数
写反、head 卷积 stride=2 等。这里采用标准的 guided-diffusion 风格实现。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """正弦时间步编码 + MLP。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),   # 输出回到 dim，与 ResBlock 的 time_proj 维度一致
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class AttentionBlock(nn.Module):
    """3D 自注意力（展平体素做 attention）。"""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv3d(ch, ch, 1)
        self.k = nn.Conv3d(ch, ch, 1)
        self.v = nn.Conv3d(ch, ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)   # (B, N, C)
        k = self.k(h).view(B, C, -1)                     # (B, C, N)
        v = self.v(h).view(B, C, -1).permute(0, 2, 1)    # (B, N, C)
        attn = torch.bmm(q, k) * (1.0 / math.sqrt(C))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                         # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, D, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0, attn=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttentionBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(temb))[:, :, None, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch=64, ch_mult=(1, 2, 2, 4),
                 num_res_blocks=2, attn_layers=(2, 3), dropout=0.15):
        super().__init__()
        assert all(i < len(ch_mult) for i in attn_layers), "attn_layers 越界"
        time_dim = ch * 4
        self.time_embedding = TimeEmbedding(time_dim)

        self.head = nn.Conv3d(in_ch, ch, 3, padding=1)

        # 下采样路径
        self.down = nn.ModuleList()
        now_ch = ch
        chs = [ch]  # 记录用于 skip connection 的通道
        for i, mult in enumerate(ch_mult):
            block_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(now_ch, block_ch, time_dim, dropout, attn=(i in attn_layers)))
                now_ch = block_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.down.append(Downsample(now_ch))
                chs.append(now_ch)

        # 中间层
        self.middle = nn.ModuleList([
            ResBlock(now_ch, now_ch, time_dim, dropout, attn=True),
            ResBlock(now_ch, now_ch, time_dim, dropout, attn=False),
        ])

        # 上采样路径
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            block_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(chs.pop() + now_ch, block_ch, time_dim, dropout, attn=(i in attn_layers)))
                now_ch = block_ch
            if i != 0:
                self.up.append(Upsample(now_ch))

        assert not chs
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv3d(now_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, t_emb)
            hs.append(h)
        for layer in self.middle:
            h = layer(h, t_emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, t_emb)
        assert not hs
        return self.tail(h)
