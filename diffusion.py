"""DDPM 训练 + DDPM/DDIM 采样。

训练与采样共用同一套 beta 调度（修复原代码“训练线性、采样余弦”的不一致）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v, t, x_shape):
    """按时间步 t 取出系数并广播到 x_shape 维度。"""
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def make_beta_schedule(beta_1, beta_T, T, schedule="linear"):
    if schedule == "linear":
        return torch.linspace(beta_1, beta_T, T)
    elif schedule == "cosine":
        return cosine_beta_schedule(T)
    raise ValueError(f"未知 schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(self, model, beta_1=1e-4, beta_T=0.02, T=1000, schedule="linear"):
        super().__init__()
        self.model = model
        self.T = T

        betas = make_beta_schedule(beta_1, beta_T, T, schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """前向加噪：x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps。"""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def train_loss(self, x0):
        """返回每个样本的 MSE 噪声预测损失（shape: [batch]）。"""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise, reduction="none").mean(dim=list(range(1, pred.ndim)))

    def predict_x0_from_eps(self, x_t, t, eps):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean(self, x0, x_t, t):
        return (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def p_sample(self, x_t, t):
        eps = self.model(x_t, t)
        x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(-1, 1)
        mean = self.q_posterior_mean(x0, x_t, t)
        if t[0].item() == 0:
            return mean
        variance = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """DDPM 采样（逐步，慢）。shape = (batch, C, D, H, W)。"""
        device = next(self.model.parameters()).device
        x_t = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.T)), desc="DDPM sampling", total=self.T, dynamic_ncols=True):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, t_tensor)
        return x_t

    @torch.no_grad()
    def ddim_sample(self, x_t, t, prev_t, eta=0.0):
        eps = self.model(x_t, t)
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = extract(self.alphas_cumprod, prev_t, x_t.shape)
        x0 = (x_t - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        # eta=0 确定性 DDIM，eta=1 接近 DDPM
        sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
        noise = torch.randn_like(x_t)
        x_prev = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev - sigma ** 2) * eps + sigma * noise
        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, shape, sampling_T=100, eta=0.0):
        """DDIM 跳步采样（快）。"""
        device = next(self.model.parameters()).device
        x_t = torch.randn(shape, device=device)
        steps = torch.linspace(self.T - 1, 0, sampling_T).long()
        for i in tqdm(range(sampling_T - 1), desc="DDIM sampling", total=sampling_T - 1, dynamic_ncols=True):
            t = torch.full((shape[0],), steps[i].item(), dtype=torch.long, device=device)
            prev_t = torch.full((shape[0],), steps[i + 1].item(), dtype=torch.long, device=device)
            x_t = self.ddim_sample(x_t, t, prev_t, eta)
        return x_t
