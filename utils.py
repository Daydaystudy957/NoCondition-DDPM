"""标签与体素的转换/保存工具。"""
import numpy as np
import tifffile


def labels_to_onehot(labels, num_phases):
    """整数标签 (D,H,W) -> one-hot (num_phases,D,H,W)，值 0/1。"""
    onehot = np.eye(num_phases, dtype=np.float32)[labels]
    return np.moveaxis(onehot, -1, 0)


def onehot_to_labels(onehot):
    """连续/one-hot 通道 (C,D,H,W) -> 整数标签 (D,H,W)，按通道 argmax。"""
    return np.argmax(onehot, axis=0).astype(np.uint8)


def save_label_npy(labels, path):
    np.save(path, labels)


def save_label_tif(labels, path):
    """把 0/1/2 标签映射成对比明显的 uint16 灰度后存 tif（方便 ImageJ/Avizo 看）。"""
    labels = np.asarray(labels).astype(np.uint8)
    n = int(labels.max()) + 1
    if n == 3:
        gray = np.zeros_like(labels, dtype=np.uint16)
        gray[labels == 0] = 0
        gray[labels == 1] = 32768
        gray[labels == 2] = 65535
    else:
        gray = (labels.astype(np.float32) / max(n - 1, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, gray)
