"""读取预处理好的 64³ 标签体素，转成 one-hot 三通道 + 三相占比条件。"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelDataset(Dataset):
    def __init__(self, data_dir, num_phases=3, normalize=True):
        self.paths = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")
        )
        assert len(self.paths) > 0, f"{data_dir} 里没有 .npy 文件"
        self.num_phases = num_phases
        self.normalize = normalize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = np.load(self.paths[idx])            # (D,H,W) uint8，值 0..num_phases-1
        label = label.astype(np.int64)
        onehot = np.eye(self.num_phases, dtype=np.float32)[label]  # (D,H,W,C)
        onehot = np.moveaxis(onehot, -1, 0)         # (C,D,H,W)，值 0/1
        if self.normalize:
            onehot = onehot * 2.0 - 1.0             # 归一化到 [-1, 1]

        # 三相占比作为条件（值 0~1，和为 1）
        cond = np.array([(label == i).mean() for i in range(self.num_phases)], dtype=np.float32)
        return torch.from_numpy(onehot), torch.from_numpy(cond)
