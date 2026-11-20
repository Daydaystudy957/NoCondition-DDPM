"""预处理：16-bit tif -> 阈值分相(0/1/2) -> 物理空间隔离切 64³ patch -> 分 train/val。

物理隔离：沿一个轴切成若干等份条带，随机挑一部分条带做训练、一部分做验证。
训练区与验证区在空间上互不重叠（避免数据泄漏）；每个条带内部用 stride 重叠采样。

用法示例：
    python preprocess.py --input /path/to/volume.tif --output ./data \
        --size 64 --stride 32 --num-strips 5 --val-strips 1
"""
import argparse
import glob
import os
import numpy as np
import tifffile

from config import SEGMENT_T1, SEGMENT_T2


def load_tif_volume(path):
    """读取单个多页 tif，或一个目录下的 tif 切片序列。"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.tif")) + glob.glob(os.path.join(path, "*.tiff")))
        assert files, f"{path} 目录里没有 tif/tiff 切片"
        slices = [tifffile.imread(f) for f in files]
        vol = np.stack(slices, axis=0)
    else:
        vol = tifffile.imread(path)
    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[None, ...]
    assert vol.ndim == 3, f"体素应为 3D，实际 {vol.ndim}D"
    return vol


def segment(vol, t1, t2):
    """两阈值分 3 相：<=t1 -> 0，t1~t2 -> 1，>t2 -> 2。"""
    label = (vol > t1).astype(np.uint8) + (vol > t2).astype(np.uint8)
    return label


def extract_patches(vol, size, stride):
    """在单个条带/体积内按 stride 滑动采样 size³ 小块。"""
    patches = []
    D, H, W = vol.shape
    for z in range(0, D - size + 1, stride):
        for y in range(0, H - size + 1, stride):
            for x in range(0, W - size + 1, stride):
                patches.append(vol[z:z + size, y:y + size, x:x + size])
    return patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="16-bit tif 文件或切片目录")
    ap.add_argument("--output", default="./data", help="输出目录（含 train/ 和 val/）")
    ap.add_argument("--size", type=int, default=64, help="patch 尺寸")
    ap.add_argument("--stride", type=int, default=32, help="重叠采样步长（<size 即重叠）")
    ap.add_argument("--t1", type=int, default=SEGMENT_T1)
    ap.add_argument("--t2", type=int, default=SEGMENT_T2)
    ap.add_argument("--split-axis", type=int, default=0, help="沿哪个轴做空间划分（0/1/2）")
    ap.add_argument("--num-strips", type=int, default=5, help="切成的条带数")
    ap.add_argument("--val-strips", type=int, default=1, help="验证集占的条带数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.num_strips > args.val_strips, "num-strips 必须大于 val-strips"

    vol = load_tif_volume(args.input)
    print(f"原始体素: shape={vol.shape}, dtype={vol.dtype}, 灰度范围 {vol.min()}~{vol.max()}")

    label = segment(vol, args.t1, args.t2)
    for c in range(3):
        print(f"  相 {c} 占比: {(label == c).mean() * 100:.2f}%")

    # 沿指定轴切成 num_strips 条带（空间隔离的基础）
    strips = np.array_split(label, args.num_strips, axis=args.split_axis)
    print(f"沿轴 {args.split_axis} 切成 {len(strips)} 条带，每条 shape: {strips[0].shape}")

    # 随机挑 val_strips 条做验证，其余训练
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(args.num_strips)
    val_ids = set(order[:args.val_strips].tolist())
    train_ids = [i for i in range(args.num_strips) if i not in val_ids]
    print(f"训练条带: {sorted(train_ids)}，验证条带: {sorted(val_ids)}")

    os.makedirs(args.output, exist_ok=True)
    counters = {"train": 0, "val": 0}
    for split, ids in [("train", train_ids), ("val", sorted(val_ids))]:
        d = os.path.join(args.output, split)
        os.makedirs(d, exist_ok=True)
        for sid in ids:
            patches = extract_patches(strips[sid], args.size, args.stride)
            for p in patches:
                idx = counters[split]
                np.save(os.path.join(d, f"{split}_{idx:05d}.npy"), p.astype(np.uint8))
                counters[split] += 1
            print(f"  {split} 条带 {sid}: {len(patches)} 个 patch")

    print(f"完成: train={counters['train']}, val={counters['val']} -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
