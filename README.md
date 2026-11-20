# Cement — 水泥三相微结构 DDPM 生成

参考 `3DMPDDPM` 的水泥入口（`DiffusionMP`）重写的干净项目：无条件多相 3D DDPM，用 3D UNet
从噪声生成 64³ 的三相水泥微结构（孔隙 / 水化产物 / 未水化颗粒）。

相对原代码的修复：统一训练/采样 beta 调度、修正 3D UNet 的 ResBlock/下采样尺寸错误、
按相数自适应通道、修正 DDIM 采样器与 import 问题、去掉调试残留。

## 环境

```bash
pip install -r requirements.txt   # torch / numpy / tifffile / tqdm / scipy
```

## 1. 预处理：tif → 分相 → 物理隔离切块

```bash
python preprocess.py --input /path/to/volume.tif --output ./data \
    --size 64 --stride 32 --num-strips 5 --val-strips 1
```

- `--input` 可以是单个多页 tif，也可以是一叠 tif 切片的目录。
- 分相规则（uint16 灰度）：
  - `0 ~ 16484` → 相 0 孔隙
  - `16485 ~ 29225` → 相 1 水化产物
  - `29226 ~ 65535` → 相 2 未水化颗粒
- **物理空间隔离**：沿 `--split-axis`（默认 0，即第一个轴）切成 `--num-strips` 条带，
  随机挑 `--val-strips` 条做验证、其余做训练，训练/验证在空间上互不重叠，避免数据泄漏。
- 每个条带内部用 `--stride`（默认 32）重叠采样 64³ patch。
- 输出 `./data/train/*.npy`、`./data/val/*.npy`（每块 `(64,64,64)` uint8 标签）。

## 2. 训练

```bash
python train.py --data ./data --out ./models --device cuda \
    --epoch 200000 --batch-size 5 --lr 1e-5
```

- 模型自动保存到 `models/<时间戳>/ckpt_*.pt`（如 `models/20261119-170725/`），每次运行独立归档。
- 超参数在 `config.py`（网络 `ch=64, ch_mult=[1,2,2,4], attn=[2,3], num_res_blocks=2`、
  扩散 `T=1000, beta∈[1e-4,0.02]`）。每 10000 epoch 存一个 `ckpt_*.pt`。

## 3. 采样生成

```bash
python sample.py --ckpt ./models/20261119-170725/ckpt_199999.pt --out ./outputs \
    --num 10 --method ddim --sampling-T 100
```

- 结果自动保存到 `outputs/<时间戳>/`（默认自动生成；也可用 `--timestamp 20261119-170725` 与训练时间戳对齐）。

每个样本输出三个文件：
- `sample_XXX_score.npy` —— 连续三通道分数（[0,1]，用于后续统计）
- `sample_XXX_label.npy` —— argmax 后的 0/1/2 标签
- `sample_XXX_label.tif` —— 映射成对比灰度的 uint16 tif，可用 ImageJ/Avizo 查看
  （0→0，1→32768，2→65535）

## 说明

- 训练数据是**分相后的离散标签**（one-hot 三通道，归一化到 [-1,1]）。
- 生成是连续值，最后用 argmax 阈值化回三相（对应 `utils.onehot_to_labels`）。
