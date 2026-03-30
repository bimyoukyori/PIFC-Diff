import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import math
from dataclasses import dataclass

# =============================================================================
# Part 1: Configuration
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
    
    # 🔥 优化点 1: 显存利用最大化
    batch_size: int = 32                
    gradient_accumulation_steps: int = 2  
    
    num_epochs: int = 100
    
    wave_velocity: float = 0.1
    center_freq: float = 100e6
    sampling_rate: float = 1e9

# =============================================================================
# Part 2: Adaptive Loss
# =============================================================================

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

# =============================================================================
# Part 3: Physics-Guided Loss 
# =============================================================================

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
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_mag = pred_mag / (pred_mag.mean() + 1e-8)
        target_mag = target_mag / (target_mag.mean() + 1e-8)
        return F.mse_loss(pred_mag, target_mag)
    
    def forward(self, pred, target):
        wave_loss = torch.clamp(self.wave_equation_loss(pred, target), max=5.0)
        hyperbola_loss = torch.clamp(self.hyperbola_consistency_loss(pred, target), max=5.0)
        spectral_loss = self.spectral_consistency_loss(pred, target)
        physics_raw = (0.3 * wave_loss + 0.4 * hyperbola_loss + 0.3 * spectral_loss)
        return physics_raw

# =============================================================================
# Part 4: U-Net
# =============================================================================

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
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, time_emb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        time_emb_projected = self.time_emb_proj(time_emb)
        h = h + time_emb_projected.view(h.shape[0], h.shape[1], 1, 1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
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
            for block_idx in range(config.num_res_blocks):
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
            for block_idx in range(config.num_res_blocks + 1):
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

# =============================================================================
# Part 5: Diffusion Model
# =============================================================================

class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unet = UNet(config)
        self.physics_loss_fn = PhysicsGuidedLoss(config)
        
        self.register_buffer('betas', self._get_betas())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
    
    def _get_betas(self):
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_timesteps)
    
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
# Part 6: Dataset (HDF5 Version)
# =============================================================================

class GPRHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, patch_size=(256, 100)):
        self.h5_file_path = h5_file_path
        self.patch_size = patch_size
        self.index_cache = []
        
        print("Scanning HDF5 structure...")
        with h5py.File(h5_file_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
            total_patches = 0
            ph, pw = patch_size
            step_h, step_w = ph // 2, pw // 2 
            
            for i in range(self.num_samples):
                grp = f[f"sample_{i}"]
                h, w = grp['raw'].shape
                if h < ph or w < pw: continue
                n_h = (h - ph) // step_h + 1
                n_w = (w - pw) // step_w + 1
                if n_h * n_w > 0:
                    self.index_cache.append({
                        'sample_idx': i, 'n_w': n_w,
                        'step_h': step_h, 'step_w': step_w,
                        'start_idx': total_patches, 'count': n_h * n_w
                    })
                    total_patches += n_h * n_w
            self.total_patches = total_patches
            print(f"✅ Indexed {total_patches} patches from HDF5.")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        target_info = None
        # 二分查找加速
        l, r = 0, len(self.index_cache) - 1
        while l <= r:
            mid = (l + r) // 2
            info = self.index_cache[mid]
            if info['start_idx'] <= idx < info['start_idx'] + info['count']:
                target_info = info
                break
            elif idx < info['start_idx']:
                r = mid - 1
            else:
                l = mid + 1
        
        if target_info is None:
            raise IndexError("Index out of range")

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
# Part 7: Training Loop
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def train_model(model: ConditionalLatentDiffusion, dataloader: DataLoader, 
                config: ModelConfig, device: str = 'cuda'):

    model.to(device)
    
    adaptive_loss_fn = AdaptiveLoss(num_tasks=2).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adaptive_loss_fn.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    early_stopping = EarlyStopping(patience=15)
    accum_steps = config.gradient_accumulation_steps
    
    # 🔥 开启 BF16 提示
    print(f"🚀 Training Config: Batch={config.batch_size}, Accum={accum_steps}, Precision=BF16 (Speed Mode)")
    model.train()
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        epoch_stats = {'total': 0.0, 'diff': 0.0, 'phys': 0.0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        optimizer.zero_grad()
        
        for batch_idx, (artea_target, raw_condition) in enumerate(pbar):
            artea_target = artea_target.to(device)
            raw_condition = raw_condition.to(device)
            t = torch.randint(0, config.num_timesteps, (artea_target.shape[0],), device=device).long()
            
            # 🔥 优化点 2: BF16 上下文
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                diff_loss_raw, phys_loss_raw = model.p_losses(artea_target, t, raw_condition)
                total_loss = adaptive_loss_fn([diff_loss_raw, phys_loss_raw])
                loss_scaled = total_loss / accum_steps
            
            loss_scaled.backward()
            
            epoch_stats['total'] += total_loss.item()
            epoch_stats['diff'] += diff_loss_raw.item()
            epoch_stats['phys'] += phys_loss_raw.item()
            
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            vars = adaptive_loss_fn.log_vars.detach().cpu().numpy()
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.2f}",
                'Diff': f"{diff_loss_raw.item():.3f}",
                'Phys': f"{phys_loss_raw.item():.2f}",
                'W_Diff': f"{vars[0]:.2f}", 
                'W_Phys': f"{vars[1]:.2f}"  
            })

        # Epoch Summary
        avg_stats = {k: v / len(dataloader) for k, v in epoch_stats.items()}
        scheduler.step(avg_stats['total'])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_stats['total']:.4f}")
        print(f"  Raw Diff Loss: {avg_stats['diff']:.4f}")
        print(f"  Raw Phys Loss: {avg_stats['phys']:.4f}")
        print(f"  Learned Weights (LogVars): {adaptive_loss_fn.log_vars.data.cpu().numpy()}")
        
        if early_stopping(avg_stats['total'], epoch):
            print("🛑 Early stopping triggered")
            break
            
        if avg_stats['total'] < best_loss:
            best_loss = avg_stats['total']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss_state': adaptive_loss_fn.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'config': config
            }, 'best_model.pt')
            print("  ✅ Best model saved")

# =============================================================================
# Main
# =============================================================================

def main():
    # 🔥 确保路径正确
    h5_path = "/home/gpr_training_data.h5"
    
    if not os.path.exists(h5_path):
        print(f"❌ HDF5 file not found: {h5_path}")
        return

    config = ModelConfig()
    dataset = GPRHDF5Dataset(h5_path, patch_size=(256, 100))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        # 🔥 优化点 3: 增加 Worker
        num_workers=16,  
        pin_memory=True, 
        persistent_workers=True
    )
    
    model = ConditionalLatentDiffusion(config)
    
    try:
        train_model(model, dataloader, config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()