"""计算验证集所有样本的三相占比，保存到 txt 文件。

用法：
    python val_phase_fractions.py --dir ./data/val --out val_phase_fractions.txt
"""
import argparse
import glob
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./data/val", help="验证集目录")
    ap.add_argument("--out", default="val_phase_fractions.txt", help="输出 txt 文件")
    ap.add_argument("--num-phases", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npy")))
    assert files, f"{args.dir} 里没有 .npy 文件"

    lines = []
    all_fracs = []
    for f in files:
        lab = np.load(f)
        fracs = [(lab == i).mean() * 100.0 for i in range(args.num_phases)]
        all_fracs.append(fracs)
        name = os.path.basename(f)
        lines.append(f"{name}: " + "  ".join(f"相{i}={fracs[i]:.2f}%" for i in range(args.num_phases)))

    all_fracs = np.array(all_fracs)
    mean = all_fracs.mean(axis=0)
    std = all_fracs.std(axis=0)
    lines.append("")
    lines.append(f"共 {len(files)} 个样本")
    lines.append("平均: " + "  ".join(f"相{i}={mean[i]:.2f}%" for i in range(args.num_phases)))
    lines.append("标准差: " + "  ".join(f"相{i}=±{std[i]:.2f}%" for i in range(args.num_phases)))

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已保存到 {args.out}（共 {len(files)} 个样本）")


if __name__ == "__main__":
    main()
