"""水泥三相微结构 DDPM 超参数配置。

参考 3DMPDDPM 的 DiffusionMP（水泥入口），修正了原代码的问题：
- 训练/采样使用同一套 beta 调度
- 网络通道按相数自适应
"""

# 模型（3D UNet）超参数
MODEL_CONFIG = {
    "in_ch": 3,              # 相数（孔隙/水化产物/未水化颗粒）
    "out_ch": 3,
    "ch": 64,                # 基础通道数
    "ch_mult": (1, 2, 2, 4),  # 各下采样层通道倍增
    "num_res_blocks": 2,
    "attn_layers": (2, 3),   # 第 2、3 个分辨率层加自注意力
    "dropout": 0.15,
}

# 扩散过程超参数
DIFFUSION_CONFIG = {
    "T": 1000,               # 训练扩散步数
    "beta_1": 1e-4,          # 线性 beta 起点
    "beta_T": 0.02,          # 线性 beta 终点
    "schedule": "linear",    # "linear" 或 "cosine"
}

# 分相阈值（uint16 灰度）
SEGMENT_T1 = 16484          # 孔隙 / 水化产物 分界
SEGMENT_T2 = 29225          # 水化产物 / 未水化颗粒 分界

# 默认训练配置（train.py 可用命令行覆盖）
TRAIN_DEFAULTS = {
    "epoch": 300,          # 约 5 万步，约 8 小时
    "batch_size": 4,       # 4090 24GB 的安全上限
    "lr": 1e-5,
    "warmup_epoch": 30,    # 300 epoch 的 10%
    "grad_clip": 1.0,
    "save_interval": 50,   # 每 50 epoch 存一个 ckpt（约每 1.3 小时）
}
