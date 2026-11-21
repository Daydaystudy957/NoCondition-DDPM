"""计算并绘制三相微结构的两点相关函数 S2(r) 对比图。

S2_i(r) = <I_i(x) * I_i(x+r)>，相距 r 的两点同时属于相 i 的概率。
s2[0] = 体积分数；s2[r] 随 r 增大衰减到体积分数的平方。

用法（批量对比生成样本 vs 验证集，直接出 PNG 图）：
    python s2_correlation.py --dir ./outputs/ckpt99_ddpm --ref ./data/val --ref-num 10
"""
import argparse
import glob
import os
import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use("Agg")  # 服务器无显示，用非交互后端
import matplotlib.pyplot as plt


def s2_isotropic(label, phase, max_r=32):
    """计算相 phase 的各向同性 S2(r)，返回 (r_values, s2_values)。"""
    indicator = (label == phase).astype(np.float64)
    D, H, W = indicator.shape
    N = D * H * W

    f = fftn(indicator)
    corr = ifftn(f * np.conj(f)).real

    dz = np.fft.fftfreq(D) * D
    dy = np.fft.fftfreq(H) * H
    dx = np.fft.fftfreq(W) * W
    dist = np.sqrt(
        np.abs(dz)[:, None, None] ** 2
        + np.abs(dy)[None, :, None] ** 2
        + np.abs(dx)[None, None, :] ** 2
    )

    max_r = min(max_r, min(D, H, W) // 2)
    r_vals = np.arange(max_r + 1)
    s2 = np.zeros(max_r + 1)
    for r in r_vals:
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.sum() > 0:
            s2[r] = corr[mask].mean() / N
    return r_vals.astype(float), s2


def sample_s2_curves(files, max_r=32, num_phases=3):
    """对多个样本计算平均 S2 曲线。返回 (r, [各相平均曲线])。"""
    acc = [[] for _ in range(num_phases)]
    r = None
    for f in files:
        lab = np.load(f)
        for p in range(num_phases):
            r, s2 = s2_isotropic(lab, p, max_r)
            acc[p].append(s2)
    means = [np.mean(acc[p], axis=0) for p in range(num_phases)]
    return r, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="生成样本目录（含 *_label.npy）")
    ap.add_argument("--ref", default=None, help="参考目录（如验证集）")
    ap.add_argument("--ref-num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-r", type=int, default=32, help="最大距离（体素）")
    ap.add_argument("--out", default=None, help="输出图片路径（默认存到生成样本目录下）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_label.npy")))
    if not files:
        print(f"[警告] {args.dir} 里没有 *_label.npy")
        return

    r, gen_means = sample_s2_curves(files, args.max_r)
    print(f"生成样本：{len(files)} 个")

    ref_means = None
    if args.ref:
        ref_files = sorted(glob.glob(os.path.join(args.ref, "*.npy")))
        rng = np.random.default_rng(args.seed)
        n = min(args.ref_num, len(ref_files))
        chosen = rng.choice(ref_files, size=n, replace=False)
        _, ref_means = sample_s2_curves(chosen, args.max_r)
        print(f"验证集参考：随机 {n} 个（seed={args.seed}）")

    phase_names = ["Phase 0 (Pore)", "Phase 1 (Hydrate)", "Phase 2 (Unhydrated)"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for p in range(3):
        ax = axes[p]
        ax.plot(r, gen_means[p], "-", color=colors[p], lw=2, label="Generated")
        if ref_means is not None:
            ax.plot(r, ref_means[p], "--", color="gray", lw=2, label="Validation")
        ax.set_xlabel("Distance r (voxel)")
        ax.set_ylabel("S2(r)")
        ax.set_title(phase_names[p])
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"S2 two-point correlation (generated vs validation)", y=1.02)
    plt.tight_layout()
    out_path = args.out if args.out else os.path.join(args.dir, "s2_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存图片: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
