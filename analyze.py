"""统计生成样本的三相占比，并与验证集随机样本对比。

生成样本：读 --dir 下所有 sample_XXX_label.npy（如 10 个）。
验证集参考：从 --ref（默认 ./data/val）随机选 --ref-num 个（默认 10）。

用法：
    python analyze.py --dir ./outputs/ckpt79_ddpm --ref ./data/val --ref-num 10
"""
import argparse
import glob
import os
import numpy as np


def phase_fractions(label, num_phases=3):
    return np.array([(label == i).mean() * 100.0 for i in range(num_phases)])


def analyze_generated(path, num_phases):
    """读取生成的所有 *_label.npy，逐个打印并返回平均±std。"""
    files = sorted(glob.glob(os.path.join(path, "*_label.npy")))
    if not files:
        print(f"[警告] {path} 里没有 *_label.npy")
        return None
    all_fracs = []
    for f in files:
        lab = np.load(f)
        fracs = phase_fractions(lab, num_phases)
        all_fracs.append(fracs)
        print(f"  {os.path.basename(f)}: " + "  ".join(f"相{i}={fracs[i]:.2f}%" for i in range(num_phases)))
    all_fracs = np.array(all_fracs)
    return all_fracs.mean(axis=0), all_fracs.std(axis=0), len(files)


def analyze_reference(path, num_phases, num_samples, seed):
    """从验证集随机选 num_samples 个，逐个打印并返回平均±std。"""
    files = sorted(glob.glob(os.path.join(path, "*.npy")))
    if not files:
        print(f"[警告] {path} 里没有 .npy")
        return None
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(files))
    chosen = rng.choice(files, size=n, replace=False)
    print(f"  随机选取 {n} 个（seed={seed}）：")
    all_fracs = []
    for f in chosen:
        lab = np.load(f)
        fracs = phase_fractions(lab, num_phases)
        all_fracs.append(fracs)
        print(f"  {os.path.basename(f)}: " + "  ".join(f"相{i}={fracs[i]:.2f}%" for i in range(num_phases)))
    all_fracs = np.array(all_fracs)
    return all_fracs.mean(axis=0), all_fracs.std(axis=0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="生成样本目录（含 sample_XXX_label.npy）")
    ap.add_argument("--ref", default="./data/val", help="验证集目录（默认 ./data/val）")
    ap.add_argument("--ref-num", type=int, default=10, help="从验证集随机选的样本数")
    ap.add_argument("--seed", type=int, default=0, help="验证集随机选取的种子")
    ap.add_argument("--num-phases", type=int, default=3)
    args = ap.parse_args()

    print("== 生成样本三相占比 ==")
    r = analyze_generated(args.dir, args.num_phases)
    if r is None:
        return
    mean, std, n = r
    print(f"\n生成样本平均（{n} 个）：")
    for i in range(args.num_phases):
        print(f"  相{i}: {mean[i]:.2f}% ± {std[i]:.2f}%")

    print("\n== 验证集随机样本三相占比（参考）==")
    r = analyze_reference(args.ref, args.num_phases, args.ref_num, args.seed)
    if r is not None:
        ref_mean, ref_std, ref_n = r
        print(f"\n验证集平均（{ref_n} 个）：")
        for i in range(args.num_phases):
            print(f"  相{i}: {ref_mean[i]:.2f}% ± {ref_std[i]:.2f}%")


if __name__ == "__main__":
    main()
