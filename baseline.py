# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from dataclasses import dataclass
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric

# =============================================================================
# 【完整复用 train.py 的全部核心部分】
# =============================================================================

@dataclass
class ModelConfig:
    in_channels: int = 4
    out_channels: int = 1
    model_channels: int = 64
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)
    dropout: float = 0.1
    channel_mult: tuple = (1, 2, 4, 8)
    num_heads: int = 4
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    learning_rate: float = 1e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_epochs: int = 50          # Baseline 对比时适当减少 epochs（可自行调高）
    wave_velocity: float = 0.1
    center_freq: float = 100e6
    sampling_rate: float = 1e9

class AdaptiveLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, loss_components):
        total_loss = 0
        for i, loss in enumerate(loss_components):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wave_velocity = config.wave_velocity
    def wave_equation_loss(self, pred, target):
        d2_dt2_pred = pred[:, :, 2:, :] - 2*pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
        d2_dx2_pred = pred[:, :, :, 2:] - 2*pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
        d2_dt2_pred = d2_dt2_pred[:, :, :, 1:-1]
        d2_dx2_pred = d2_dx2_pred[:, :, 1:-1, :]
        wave_residual_pred = d2_dt2_pred - (self.wave_velocity**2) * d2_dx2_pred
        return torch.mean(wave_residual_pred**2)
    def hyperbola_consistency_loss(self, pred, target):
        grad_x_pred = torch.gradient(pred, dim=3)[0]
        grad_t_pred = torch.gradient(pred, dim=2)[0]
        grad_x_target = torch.gradient(target, dim=3)[0]
        grad_t_target = torch.gradient(target, dim=2)[0]
        grad_x_pred = torch.tanh(grad_x_pred * 10)
        grad_t_pred = torch.tanh(grad_t_pred * 10)
        grad_x_target = torch.tanh(grad_x_target * 10)
        grad_t_target = torch.tanh(grad_t_target * 10)
        curvature_pred = grad_x_pred**2 + grad_t_pred**2
        curvature_target = grad_x_target**2 + grad_t_target**2
        return F.mse_loss(curvature_pred, curvature_target)
    def spectral_consistency_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_mag = torch.abs(pred_fft) / (torch.abs(pred_fft).mean() + 1e-8)
        target_mag = torch.abs(target_fft) / (torch.abs(target_fft).mean() + 1e-8)
        return F.mse_loss(pred_mag, target_mag)
    def forward(self, pred, target):
        wave_loss = torch.clamp(self.wave_equation_loss(pred, target), max=5.0)
        hyperbola_loss = torch.clamp(self.hyperbola_consistency_loss(pred, target), max=5.0)
        spectral_loss = self.spectral_consistency_loss(pred, target)
        return 0.3 * wave_loss + 0.4 * hyperbola_loss + 0.3 * spectral_loss

# ResBlock / AttentionBlock / TimeEmbedding / UNet
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, time_emb):
        h = x
        h = self.norm1(h); h = F.silu(h); h = self.conv1(h)
        time_emb_projected = self.time_emb_proj(time_emb)
        h = h + time_emb_projected.view(h.shape[0], h.shape[1], 1, 1)
        h = self.norm2(h); h = F.silu(h); h = self.dropout(h); h = self.conv2(h)
        return h + self.skip_connection(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(b, self.num_heads, c // self.num_heads * 3, h * w)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        scale = (c // self.num_heads) ** -0.5
        attention = torch.einsum('b h c s, b h c t -> b h s t', q, k) * scale
        attention = F.softmax(attention, dim=-1)
        out = torch.einsum('b h s t, b h c t -> b h c s', attention, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj_out(out)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_channels = config.model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(config.in_channels, model_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        self.encoder_out_channels = [ch]
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                out_ch = model_channels * mult
                layers = [ResBlock(ch, out_ch, config.dropout, time_embed_dim)]
                ch = out_ch
                if mult in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, config.num_heads))
                self.down_blocks.append(nn.ModuleList(layers))
                self.encoder_out_channels.append(ch)
            if level != len(config.channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                self.encoder_out_channels.append(ch)
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, config.dropout, time_embed_dim),
            AttentionBlock(ch, config.num_heads),
            ResBlock(ch, ch, config.dropout, time_embed_dim)
        ])
        self.up_blocks = nn.ModuleList()
        skip_channels = list(reversed(self.encoder_out_channels))
        skip_idx = 0
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            for _ in range(config.num_res_blocks + 1):
                skip_ch = skip_channels[skip_idx]
                skip_idx += 1
                out_ch = model_channels * mult
                layers = [ResBlock(ch + skip_ch, out_ch, config.dropout, time_embed_dim)]
                ch = out_ch
                if mult in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, config.num_heads))
                self.up_blocks.append(nn.ModuleList(layers))
            if level != 0:
                self.up_blocks.append(nn.ModuleList([nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)]))
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, config.out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps):
        t_emb = self.time_embed(timesteps)
        hs = []
        h = self.input_conv(x)
        hs.append(h)
        for module_list in self.down_blocks:
            for module in module_list:
                if isinstance(module, ResBlock): h = module(h, t_emb)
                elif isinstance(module, AttentionBlock): h = module(h)
                elif isinstance(module, nn.Conv2d): h = module(h)
            hs.append(h)
        for module in self.middle_block:
            if isinstance(module, ResBlock): h = module(h, t_emb)
            else: h = module(h)
        for module_list in self.up_blocks:
            for module in module_list:
                if isinstance(module, ResBlock):
                    skip = hs.pop()
                    if h.shape[2:] != skip.shape[2:]:
                        h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                    h = module(h, t_emb)
                elif isinstance(module, AttentionBlock): h = module(h)
                elif isinstance(module, nn.ConvTranspose2d): h = module(h)
        return self.out(h)

class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unet = UNet(config)
        self.physics_loss_fn = PhysicsGuidedLoss(config)
        self.register_buffer('betas', torch.linspace(config.beta_start, config.beta_end, config.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start +
                self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)

    def _predict_x0_from_eps(self, x_t, t, eps):
        return (x_t - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * eps) / \
               self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)

    def p_losses(self, artea_clean, t, raw_condition, noise=None):
        if noise is None: noise = torch.randn_like(artea_clean)
        artea_noisy = self.q_sample(artea_clean, t, noise)
        model_input = torch.cat([artea_noisy, raw_condition], dim=1)
        predicted_noise = self.unet(model_input, t)
        diffusion_loss = F.l1_loss(predicted_noise, noise)
        artea_recon = self._predict_x0_from_eps(artea_noisy, t, predicted_noise)
        physics_raw_loss = self.physics_loss_fn(artea_recon, artea_clean)
        return diffusion_loss, physics_raw_loss

    @torch.no_grad()
    def sample(self, raw_condition):
        device = raw_condition.device
        shape = (raw_condition.shape[0], 1, raw_condition.shape[2], raw_condition.shape[3])
        x = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.config.num_timesteps)), desc="Sampling", leave=False):
            t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            model_input = torch.cat([x, raw_condition], dim=1)
            predicted_noise = self.unet(model_input, t)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            pred_x0 = (x - sqrt_one_minus_alphas * predicted_noise) / sqrt_alpha_cumprod
            if i > 0:
                noise = torch.randn_like(x)
                beta_t = self.betas[t].view(-1, 1, 1, 1)
                x = (x - beta_t / sqrt_one_minus_alphas * predicted_noise) / torch.sqrt(alpha_t)
                x += torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
        return x

# =============================================================================
# 修改后的 Dataset（支持 train/test 划分）
# =============================================================================
class GPRHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, patch_size=(256, 100), split='train', test_ratio=0.2):
        self.h5_file_path = h5_file_path
        self.patch_size = patch_size
        self.index_cache = []
        print(f"Scanning HDF5 for {split} split...")
        with h5py.File(h5_file_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
            total_patches = 0
            ph, pw = patch_size
            step_h, step_w = ph // 2, pw // 2
            temp_cache = []
            for i in range(self.num_samples):
                grp = f[f"sample_{i}"]
                h, w = grp['raw'].shape
                if h < ph or w < pw: continue
                n_h = (h - ph) // step_h + 1
                n_w = (w - pw) // step_w + 1
                if n_h * n_w > 0:
                    temp_cache.append({
                        'sample_idx': i, 'n_w': n_w,
                        'step_h': step_h, 'step_w': step_w,
                        'start_idx': total_patches, 'count': n_h * n_w
                    })
                    total_patches += n_h * n_w
            split_idx = int(len(temp_cache) * (1 - test_ratio))
            self.index_cache = temp_cache[:split_idx] if split == 'train' else temp_cache[split_idx:]
            total_patches = 0
            for info in self.index_cache:
                info['start_idx'] = total_patches
                total_patches += info['count']
            self.total_patches = total_patches
            print(f"✅ Indexed {self.total_patches} patches for {split}.")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        for info in self.index_cache:
            if info['start_idx'] <= idx < info['start_idx'] + info['count']:
                target_info = info
                break
        local_idx = idx - target_info['start_idx']
        patch_row = local_idx // target_info['n_w']
        patch_col = local_idx % target_info['n_w']
        y = patch_row * target_info['step_h']
        x = patch_col * target_info['step_w']
        ph, pw = self.patch_size
        with h5py.File(self.h5_file_path, 'r') as f:
            grp = f[f"sample_{target_info['sample_idx']}"]
            raw = torch.from_numpy(grp['raw'][y:y+ph, x:x+pw]).unsqueeze(0)
            artea = torch.from_numpy(grp['artea'][y:y+ph, x:x+pw]).unsqueeze(0)
            gt = torch.from_numpy(grp['grad_t'][y:y+ph, x:x+pw]).unsqueeze(0)
            gx = torch.from_numpy(grp['grad_x'][y:y+ph, x:x+pw]).unsqueeze(0)
        return artea, torch.cat([raw, gt, gx], dim=0)

# =============================================================================
# 传统方法 
# =============================================================================
def apply_fk_filter(raw_2d):
    """ f-k 滤波：输入应为 (H, W) 的 numpy 数组 """
    H, W = raw_2d.shape
    fft = np.fft.fft2(raw_2d)
    fft_shift = np.fft.fftshift(fft)
    
    mask = np.ones_like(fft_shift, dtype=complex)
    k_cutoff = 8  
    # 过滤掉高波数（远离中心的区域）
    mask[:, :W//2 - k_cutoff] = 0
    mask[:, W//2 + k_cutoff:] = 0
    
    filtered_fft = fft_shift * mask
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
    # 归一化
    filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
    return filtered

def apply_svd_filter(raw_2d, rank=8):
    """ SVD 低秩逼近：输入应为 (H, W) 的 numpy 数组 """
    U, S, Vt = np.linalg.svd(raw_2d, full_matrices=False)
    S[rank:] = 0
    denoised = U @ np.diag(S) @ Vt
    denoised = (denoised - denoised.mean()) / (denoised.std() + 1e-8)
    return denoised

# =============================================================================
# DnCNN / Pix2Pix-like
# =============================================================================
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        layers = [nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers += [nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(64, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Pix2PixGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = UNet(config)
    def forward(self, condition):
        b, c, h, w = condition.shape
        dummy_noisy = torch.zeros(b, 1, h, w, device=condition.device)
        model_input = torch.cat([dummy_noisy, condition], dim=1)
        dummy_t = torch.zeros(b, dtype=torch.long, device=condition.device)
        return self.unet(model_input, dummy_t)

# =============================================================================
# 指标 & 评估函数（已加入归一化保护）
# =============================================================================
def compute_metrics(pred, target):
    # 限制范围到 [-1, 1]
    pred = pred.clamp(-1, 1)
    target = target.clamp(-1, 1)
    
    # 1. 计算 MSE 和 PSNR
    mse = F.mse_loss(pred, target).item()
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 100.0
    
    # 2. 计算 SSIM (针对 Batch 中的每一张图计算后求平均)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    ssim_val = 0.0
    batch_size = pred_np.shape[0]
    for i in range(batch_size):
        p = pred_np[i, 0] # 提取单通道图像 (H, W)
        t = target_np[i, 0]
        # 因为数据范围是 [-1, 1]，所以 data_range = 1 - (-1) = 2.0
        ssim_val += ssim_metric(t, p, data_range=2.0)
    ssim_val /= batch_size
    
    return {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim_val}

@torch.no_grad()
def evaluate_method(method, test_loader, device, method_name, config=None):
    """
    评估指定方法的性能指标 (MSE, PSNR)
    """
    metrics_list = []
    samples = []
    
    for i, (artea_target, raw_condition) in enumerate(tqdm(test_loader, desc=f"Eval {method_name}")):
        artea_target = artea_target.to(device)      # [B, 1, H, W]
        raw_condition = raw_condition.to(device)    # [B, 4, H, W]
        batch_size = artea_target.shape[0]

        # 1. 获取预测值 (处理不同方法的维度差异)
        if method_name in ["f-k Filter", "SVD"]:
            # 传统方法：逐张处理 Batch 里的图像
            preds_in_batch = []
            raw_images = raw_condition[:, 0].cpu().numpy()  # 取第一个通道: (B, H, W)
            
            for b in range(batch_size):
                if method_name == "f-k Filter":
                    res = apply_fk_filter(raw_images[b])
                else:
                    res = apply_svd_filter(raw_images[b])
                # res 形状为 (H, W)，转为 Tensor
                preds_in_batch.append(torch.from_numpy(res))
            
            # 关键修复：stack 得到 (B, H, W)，unsqueeze(1) 得到 (B, 1, H, W)
            pred = torch.stack(preds_in_batch).unsqueeze(1).to(device)
            
        elif method_name in ["DnCNN", "Pix2Pix-like"]:
            pred = method(raw_condition)
        else:  # DDPM / PIFC-Diff
            pred = method.sample(raw_condition)
        
        # 2. 维度对齐与安全性检查
        # 强制取第一个通道并确保形状为 [B, 1, H, W]
        pred = pred[:, :1] 
        if pred.shape != artea_target.shape:
            pred = pred.view(artea_target.shape)

        # 3. 统一归一化 (非常重要：确保不同方法在同一量纲下对比指标)
        def normalize_tensor(t):
            mean = t.mean(dim=(2, 3), keepdim=True)
            std = t.std(dim=(2, 3), keepdim=True) + 1e-8
            return (t - mean) / std

        pred = normalize_tensor(pred)
        target = normalize_tensor(artea_target)
        
        # 4. 计算指标
        metrics = compute_metrics(pred, target)
        metrics_list.append(metrics)
        
        # 5. 缓存前几个 Batch 的第一张图用于可视化
        if i < 5:
            # 存储 (Raw, GT, Pred)
            samples.append((
                raw_condition[0:1, 0:1].cpu(), 
                target[0:1].cpu(), 
                pred[0:1].cpu()
            ))
    
    # 计算全集平均指标
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print(f"✅ {method_name} 平均指标: {avg_metrics}")
    
    # 6. 保存采样图
    save_dir = "baseline_results/samples"
    os.makedirs(save_dir, exist_ok=True)
    for idx, (raw, gt, p) in enumerate(samples):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(raw.squeeze(), cmap='gray'); axs[0].set_title('Raw Input')
        axs[1].imshow(gt.squeeze(), cmap='gray'); axs[1].set_title('Ground Truth')
        axs[2].imshow(p.squeeze(), cmap='gray'); axs[2].set_title(f'Pred ({method_name})')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{method_name.replace(' ', '_')}_sample_{idx}.png")
        plt.close()
        
    return avg_metrics

# =============================================================================
# 主函数
# =============================================================================
def main():
    h5_path = "/home/gpr_training_data.h5"
    if not os.path.exists(h5_path):
        print("❌ HDF5 文件不存在！")
        return
    config = ModelConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    train_dataset = GPRHDF5Dataset(h5_path, split='train', test_ratio=0.2)
    test_dataset = GPRHDF5Dataset(h5_path, split='test', test_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs("baseline_results", exist_ok=True)
    results = {}

    # 1. 传统方法（无需训练）
    for name in ["f-k Filter", "SVD"]:
        print(f"\n=== {name} ===")
        avg = evaluate_method(None, test_loader, device, name, config)
        results[name] = avg

    # 2. DnCNN
    print("\n=== 训练 DnCNN ===")
    dncnn = DnCNN().to(device)
    opt = torch.optim.Adam(dncnn.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs // 2):
        dncnn.train()
        for artea, cond in tqdm(train_loader):
            artea, cond = artea.to(device), cond.to(device)
            opt.zero_grad()
            pred = dncnn(cond)
            loss = F.l1_loss(pred, artea)
            loss.backward()
            opt.step()
    avg = evaluate_method(dncnn, test_loader, device, "DnCNN")
    results["DnCNN"] = avg
    torch.save(dncnn.state_dict(), "baseline_results/dncnn.pt")

    # 3. Pix2Pix-like
    print("\n=== 训练 Pix2Pix-like ===")
    pix = Pix2PixGenerator(config).to(device)
    opt = torch.optim.Adam(pix.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs // 2):
        pix.train()
        for artea, cond in tqdm(train_loader):
            artea, cond = artea.to(device), cond.to(device)
            opt.zero_grad()
            pred = pix(cond)
            loss = F.l1_loss(pred, artea)
            loss.backward()
            opt.step()
    avg = evaluate_method(pix, test_loader, device, "Pix2Pix-like")
    results["Pix2Pix-like"] = avg
    torch.save(pix.state_dict(), "baseline_results/pix2pix_like.pt")

    # 4. 标准 DDPM（无 physics / adaptive）
    print("\n=== 训练 标准 DDPM ===")
    std_ddpm = ConditionalLatentDiffusion(config).to(device)
    opt = torch.optim.AdamW(std_ddpm.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    for epoch in range(config.num_epochs // 2):
        std_ddpm.train()
        for i, (artea, cond) in enumerate(tqdm(train_loader)):
            artea, cond = artea.to(device), cond.to(device)
            t = torch.randint(0, config.num_timesteps, (artea.shape[0],), device=device).long()
            diff_loss, _ = std_ddpm.p_losses(artea, t, cond)
            loss = diff_loss
            loss.backward()
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(std_ddpm.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
    avg = evaluate_method(std_ddpm, test_loader, device, "Standard DDPM")
    results["Standard DDPM"] = avg
    torch.save(std_ddpm.state_dict(), "baseline_results/std_ddpm.pt")

    # 5. PIFC-Diff（你的完整模型）
    print("\n=== 训练 PIFC-Diff（proposed） ===")
    pifc = ConditionalLatentDiffusion(config).to(device)
    adaptive = AdaptiveLoss().to(device)
    opt = torch.optim.AdamW(list(pifc.parameters()) + list(adaptive.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    for epoch in range(config.num_epochs):
        pifc.train()
        for i, (artea, cond) in enumerate(tqdm(train_loader)):
            artea, cond = artea.to(device), cond.to(device)
            t = torch.randint(0, config.num_timesteps, (artea.shape[0],), device=device).long()
            diff_loss, phys_loss = pifc.p_losses(artea, t, cond)
            total_loss = adaptive([diff_loss, phys_loss])
            (total_loss / config.gradient_accumulation_steps).backward()
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(pifc.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
    avg = evaluate_method(pifc, test_loader, device, "PIFC-Diff")
    results["PIFC-Diff"] = avg
    torch.save({'model': pifc.state_dict(), 'adaptive': adaptive.state_dict()}, "baseline_results/pifc_diff.pt")

    # 保存对比表格
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv("baseline_results/comparison_metrics.csv")
    print("\n🎉 全部对比完成！")
    print(df)
    print("\n📁 结果已保存至 baseline_results/ 文件夹，可直接用于后续成图。")

if __name__ == "__main__":
    main()