"""从训练好的条件 DDPM 采样生成 3 相水泥结构，保存 .npy + .tif。

无条件采样（w=0）：
    python sample.py --ckpt ./models/xxx/ckpt_299.pt --out ./outputs --num 10

条件采样（指定三相占比 + 引导）：
    python sample.py --ckpt ./models/xxx/ckpt_299.pt --out ./outputs --num 10 \
        --cond "0.30,0.40,0.30" --w 1.8
"""
import argparse
import datetime
import os
import numpy as np
import torch

from config import MODEL_CONFIG, DIFFUSION_CONFIG, COND_CONFIG
from model import UNet
from diffusion import GaussianDiffusion
from utils import onehot_to_labels, save_label_tif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="模型权重路径")
    ap.add_argument("--out", default="./outputs", help="输出根目录（其下自动建时间戳子目录）")
    ap.add_argument("--timestamp", default=None, help="手动指定时间戳(如 20261119-170725)，默认自动生成")
    ap.add_argument("--num", type=int, default=10, help="生成数量")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--method", default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--sampling-T", type=int, default=100, help="DDIM 采样步数")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM 随机性(0=确定性)")
    ap.add_argument("--cond", default=None, help="目标三相占比，逗号分隔，如 '0.30,0.40,0.30'")
    ap.add_argument("--w", type=float, default=None, help="引导强度（默认取 config 的 guidance_w）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.out, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = UNet(**MODEL_CONFIG, cond_dim=COND_CONFIG["cond_dim"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" in ckpt:  # 新版 train.py 保存的完整状态 dict
        model.load_state_dict(ckpt["model"])
    else:  # 旧版只存权重的 state_dict
        model.load_state_dict(ckpt)
    model.eval()
    diffusion = GaussianDiffusion(model, **DIFFUSION_CONFIG, uncond_prob=COND_CONFIG["uncond_prob"]).to(device)

    # 目标三相占比条件
    if args.cond:
        cond = torch.tensor([float(v) for v in args.cond.split(",")], device=device).unsqueeze(0)
        print(f"条件采样：目标三相占比 {args.cond}")
    else:
        cond = torch.zeros(1, COND_CONFIG["cond_dim"], device=device)
        print("无条件采样（条件为零向量）")

    w = args.w if args.w is not None else COND_CONFIG["guidance_w"]
    print(f"引导强度 w = {w}")

    shape = (1, MODEL_CONFIG["in_ch"], args.size, args.size, args.size)
    for i in range(args.num):
        with torch.no_grad():
            if args.method == "ddpm":
                x0 = diffusion.p_sample_loop(shape, cond, w=w)
            else:
                x0 = diffusion.ddim_sample_loop(shape, cond, sampling_T=args.sampling_T, w=w, eta=args.eta)

        score = ((x0[0] + 1.0) / 2.0).clamp(0, 1).cpu().numpy()   # (C,D,H,W) 连续分数
        labels = onehot_to_labels(score)                          # argmax -> 0/1/2

        np.save(os.path.join(save_dir, f"sample_{i:03d}_score.npy"), score)
        np.save(os.path.join(save_dir, f"sample_{i:03d}_label.npy"), labels)
        save_label_tif(labels, os.path.join(save_dir, f"sample_{i:03d}_label.tif"))
        print(f"已生成 sample_{i:03d}")

    print("完成，输出目录:", os.path.abspath(save_dir))


if __name__ == "__main__":
    main()
