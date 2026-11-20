"""3D UNet（条件扩散模型）。

在无条件 UNet 基础上增加条件嵌入：三相占比 3 维向量 -> ConditionalEmbedding
-> 注入每个 ResBlock（与时间嵌入并列），用于 classifier-free guidance。
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
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class ConditionalEmbedding(nn.Module):
    """把条件向量（三相占比）嵌入到 time_dim 维。"""

    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, c):
        return self.mlp(c)


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
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1).permute(0, 2, 1)
        attn = torch.bmm(q, k) * (1.0 / math.sqrt(C))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
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
        self.cond_proj = nn.Linear(time_dim, out_ch)  # 条件嵌入注入
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttentionBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb, cemb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(temb))[:, :, None, None, None]
        h = h + self.cond_proj(cemb)[:, :, None, None, None]  # 注入条件
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

    def forward(self, x, temb=None, cemb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x, temb=None, cemb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, ch=64, ch_mult=(1, 2, 2, 4),
                 num_res_blocks=2, attn_layers=(2, 3), dropout=0.15, cond_dim=3):
        super().__init__()
        assert all(i < len(ch_mult) for i in attn_layers), "attn_layers 越界"
        time_dim = ch * 4
        self.time_embedding = TimeEmbedding(time_dim)
        self.cond_embedding = ConditionalEmbedding(cond_dim, time_dim)

        self.head = nn.Conv3d(in_ch, ch, 3, padding=1)

        self.down = nn.ModuleList()
        now_ch = ch
        chs = [ch]
        for i, mult in enumerate(ch_mult):
            block_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(now_ch, block_ch, time_dim, dropout, attn=(i in attn_layers)))
                now_ch = block_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.down.append(Downsample(now_ch))
                chs.append(now_ch)

        self.middle = nn.ModuleList([
            ResBlock(now_ch, now_ch, time_dim, dropout, attn=True),
            ResBlock(now_ch, now_ch, time_dim, dropout, attn=False),
        ])

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

    def forward(self, x, t, cond):
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(cond)
        h = self.head(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, temb, cemb)
            hs.append(h)
        for layer in self.middle:
            h = layer(h, temb, cemb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        assert not hs
        return self.tail(h)
