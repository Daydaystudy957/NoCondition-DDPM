"""训练 3 相水泥微结构 DDPM。

用法示例：
    python train.py --data ./data --out ./models --device cuda \
        --epoch 300 --batch-size 4 --lr 1e-5

断点续训（恢复 model/optimizer/scheduler/epoch，无缝继续）：
    python train.py --data ./data --out ./models --device cuda \
        --epoch 300 --batch-size 4 --lr 1e-5 \
        --resume ./models/20261119-173203/ckpt_39.pt
"""
import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from config import MODEL_CONFIG, DIFFUSION_CONFIG, TRAIN_DEFAULTS
from dataset import VoxelDataset
from model import UNet
from diffusion import GaussianDiffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="预处理输出目录（含 train/）")
    ap.add_argument("--out", default="./models", help="模型根目录（全新训练时其下自动建时间戳子目录）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epoch", type=int, default=TRAIN_DEFAULTS["epoch"])
    ap.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=TRAIN_DEFAULTS["lr"])
    ap.add_argument("--warmup-epoch", type=int, default=TRAIN_DEFAULTS["warmup_epoch"])
    ap.add_argument("--resume", default=None, help="从某个 ckpt 断点续训（恢复 model/optimizer/scheduler/epoch）")
    ap.add_argument("--save-interval", type=int, default=TRAIN_DEFAULTS["save_interval"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    dataset = VoxelDataset(os.path.join(args.data, "train"), num_phases=MODEL_CONFIG["in_ch"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=True)
    print(f"训练样本数: {len(dataset)}, 每 epoch 步数: {len(loader)}")

    model = UNet(**MODEL_CONFIG).to(device)
    diffusion = GaussianDiffusion(model, **DIFFUSION_CONFIG).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epoch)
    cosine = CosineAnnealingLR(opt, T_max=max(args.epoch - args.warmup_epoch, 1))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[args.warmup_epoch])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {n_params:.2f}M")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:  # 新格式：完整训练状态
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
        else:  # 旧格式：只有权重（向后兼容）
            model.load_state_dict(ckpt)
            start_epoch = 0
        save_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"已恢复，从 epoch {start_epoch} 继续到 epoch {args.epoch}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(args.out, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print("模型保存目录:", save_dir)

    for e in range(start_epoch, args.epoch):
        model.train()
        pbar = tqdm(total=len(dataset), desc=f"epoch {e}/{args.epoch}", dynamic_ncols=True)
        for x in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad()
            loss = diffusion.train_loss(x).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_DEFAULTS["grad_clip"])
            opt.step()
            pbar.update(x.shape[0])
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.close()
        scheduler.step()

        if e % 100 == 0 or e == args.epoch - 1:
            print(f"[epoch {e:6d}] loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if (e + 1) % args.save_interval == 0:
            ckpt = os.path.join(save_dir, f"ckpt_{e}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": e,
            }, ckpt)
            print("已保存:", ckpt)


if __name__ == "__main__":
    main()
