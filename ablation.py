
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
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric


try:
    from train import UNet, AdaptiveLoss, PhysicsGuidedLoss
except ImportError:
    raise ImportError("❌ 找不到 train.py！请务必将 ablation.py 放在与 train.py 相同的目录下！")

# =============================================================================
# Part 1: Configuration (支持消融控制)
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
    

    batch_size: int = 32                
    gradient_accumulation_steps: int = 2  
    num_epochs: int = 50                
    
    wave_velocity: float = 0.1
    center_freq: float = 100e6
    sampling_rate: float = 1e9

    use_physics: bool = True     
    cond_type: str = 'full'      

# =============================================================================
# Part 2: Conditional Diffusion (支持消融开关)
# =============================================================================

class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
       
        self.unet = UNet(config) 
        
        if self.config.use_physics:
            self.physics_loss_fn = PhysicsGuidedLoss(config)
        
        self.register_buffer('betas', torch.linspace(config.beta_start, config.beta_end, config.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
    
    def q_sample(self, x_start, t, noise):
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start + 
                self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)
    
    def _predict_x0_from_eps(self, x_t, t, eps):
        return (x_t - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * eps) / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)

    def p_losses(self, artea_clean, t, condition, noise=None):
        if noise is None: noise = torch.randn_like(artea_clean)
        artea_noisy = self.q_sample(artea_clean, t, noise)
        model_input = torch.cat([artea_noisy, condition], dim=1)
        predicted_noise = self.unet(model_input, t)
        
        diffusion_loss = F.l1_loss(predicted_noise, noise)
        
     
        if not self.config.use_physics:
            return diffusion_loss, torch.tensor(0.0, device=diffusion_loss.device)
            
        artea_recon = self._predict_x0_from_eps(artea_noisy, t, predicted_noise)
        physics_loss = self.physics_loss_fn(artea_recon, artea_clean)
        return diffusion_loss, physics_loss

    @torch.no_grad()
    def sample(self, condition, num_steps=50): 
        # 默认启用 DDIM 50步极速采样加速验证
        device = condition.device
        b, c_cond, h, w = condition.shape
        x = torch.randn((b, 1, h, w), device=device)
        times = torch.linspace(-1, self.config.num_timesteps - 1, steps=num_steps + 1).long().to(device)
        times = list(reversed(times.tolist()))
        
        for i in range(len(times) - 1):
            t = times[i]; t_next = times[i + 1]
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            pred_noise = self.unet(torch.cat([x, condition], dim=1), t_tensor)
            
            a_t = self.alphas_cumprod[t] if t >= 0 else torch.tensor(1.0, device=device)
            a_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            x0_pred = (x - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)
            dir_xt = torch.sqrt(1 - a_next) * pred_noise
            x = torch.sqrt(a_next) * x0_pred + dir_xt
        return x

# =============================================================================
# Part 3: Dataset (支持 Train/Test 分割和动态 Cond)
# =============================================================================

class GPRHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, patch_size=(256, 100), split='train', test_ratio=0.2, cond_type='full'):
        self.h5_file_path = h5_file_path
        self.patch_size = patch_size
        self.cond_type = cond_type
        self.index_cache = []
        
        with h5py.File(h5_file_path, 'r') as f:
            num_samples = f.attrs['num_samples']
            ph, pw = patch_size
            step_h, step_w = ph // 2, pw // 2
            temp_cache = []
            total_patches = 0
            for i in range(num_samples):
                grp = f[f"sample_{i}"]
                h, w = grp['raw'].shape
                if h < ph or w < pw: continue
                n_h = (h - ph) // step_h + 1
                n_w = (w - pw) // step_w + 1
                if n_h * n_w > 0:
                    temp_cache.append({
                        'sample_idx': i, 'n_w': n_w, 'step_h': step_h, 'step_w': step_w,
                        'start_idx': total_patches, 'count': n_h * n_w
                    })
                    total_patches += n_h * n_w
                    
            split_idx = int(len(temp_cache) * (1 - test_ratio))
            self.index_cache = temp_cache[:split_idx] if split == 'train' else temp_cache[split_idx:]
            
            self.total_patches = 0
            for info in self.index_cache:
                info['start_idx'] = self.total_patches
                self.total_patches += info['count']

    def __len__(self): return self.total_patches

    def __getitem__(self, idx):
        l, r = 0, len(self.index_cache) - 1
        target_info = None
        while l <= r:
            mid = (l + r) // 2
            info = self.index_cache[mid]
            if info['start_idx'] <= idx < info['start_idx'] + info['count']:
                target_info = info; break
            elif idx < info['start_idx']: r = mid - 1
            else: l = mid + 1

        local_idx = idx - target_info['start_idx']
        y = (local_idx // target_info['n_w']) * target_info['step_h']
        x = (local_idx % target_info['n_w']) * target_info['step_w']
        ph, pw = self.patch_size
        
        with h5py.File(self.h5_file_path, 'r') as f:
            grp = f[f"sample_{target_info['sample_idx']}"]
            artea = torch.from_numpy(grp['artea'][y:y+ph, x:x+pw]).unsqueeze(0).float()
            raw = torch.from_numpy(grp['raw'][y:y+ph, x:x+pw]).unsqueeze(0).float()
            
           
            if self.cond_type == 'raw_only':
                return artea, raw
            else:
                gt = torch.from_numpy(grp['grad_t'][y:y+ph, x:x+pw]).unsqueeze(0).float()
                gx = torch.from_numpy(grp['grad_x'][y:y+ph, x:x+pw]).unsqueeze(0).float()
                return artea, torch.cat([raw, gt, gx], dim=0)

# =============================================================================
# Part 4: Training & Evaluation Pipeline
# =============================================================================

def train_and_eval_variant(variant_name, config, train_loader, test_loader, device):
    print(f"\n{'='*50}\n🚀 启动消融实验: {variant_name}\n{'='*50}")
    print(f"配置 -> Physics: {config.use_physics} | Cond Type: {config.cond_type} | In_channels: {config.in_channels}")
    
    model = ConditionalLatentDiffusion(config).to(device)
    
    # 动态配置优化器
    if config.use_physics:
        adaptive_loss_fn = AdaptiveLoss(num_tasks=2).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(adaptive_loss_fn.parameters()), lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
    model.train()
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader, desc=f'[{variant_name}] Epoch {epoch+1}/{config.num_epochs}', leave=False)
        for artea_target, condition in pbar:
            artea_target, condition = artea_target.to(device), condition.to(device)
            t = torch.randint(0, config.num_timesteps, (artea_target.shape[0],), device=device).long()
            
            optimizer.zero_grad()
           
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                diff_loss, phys_loss = model.p_losses(artea_target, t, condition)
                if config.use_physics:
                    total_loss = adaptive_loss_fn([diff_loss, phys_loss])
                else:
                    total_loss = diff_loss
                    
                loss_scaled = total_loss / config.gradient_accumulation_steps
                
            loss_scaled.backward()
            
            if pbar.n % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})

    # ========== 评估环节 ==========
    print(f"\n📊 开始定量评估: {variant_name}")
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    
    with torch.no_grad():
        for artea_target, condition in tqdm(test_loader, desc="Evaluating"):
            artea_target, condition = artea_target.to(device), condition.to(device)
            
            pred = model.sample(condition, num_steps=50) # DDIM 极速验证
            
            pred_norm = torch.clamp(pred, -1, 1)
            target_norm = torch.clamp(artea_target, -1, 1)
            
            mse = F.mse_loss(pred_norm, target_norm).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-8))
            
            pred_np = pred_norm.cpu().numpy()
            target_np = target_norm.cpu().numpy()
            ssim_val = sum(ssim_metric(target_np[i, 0], pred_np[i, 0], data_range=2.0) for i in range(pred_np.shape[0])) / pred_np.shape[0]
            
            mse_list.append(mse); psnr_list.append(psnr); ssim_list.append(ssim_val)

    avg_metrics = {'MSE': np.mean(mse_list), 'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}
    print(f"✅ {variant_name} 结果 -> MSE: {avg_metrics['MSE']:.4f} | PSNR: {avg_metrics['PSNR']:.2f} | SSIM: {avg_metrics['SSIM']:.4f}")
    
    os.makedirs("ablation_results", exist_ok=True)
    save_path = f"ablation_results/{variant_name.replace(' ', '_').replace('/', '_')}.pt"
    torch.save(model.state_dict(), save_path)
    
    return avg_metrics

# =============================================================================
# Main
# =============================================================================

def main():
    h5_path = "/home/gpr_training_data.h5" 
    if not os.path.exists(h5_path):
        print(f"❌ 找不到数据文件: {h5_path}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # ==========================================================
    # Variant A: w/o Physics (证明物理损失的必要性)
    # ==========================================================
    cfg_a = ModelConfig(use_physics=False, cond_type='full', in_channels=4)
    train_dl_a = DataLoader(GPRHDF5Dataset(h5_path, split='train', cond_type='full'), batch_size=cfg_a.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl_a = DataLoader(GPRHDF5Dataset(h5_path, split='test', cond_type='full'), batch_size=cfg_a.batch_size, shuffle=False, num_workers=4)
    res_a = train_and_eval_variant("Variant A (wo Physics)", cfg_a, train_dl_a, test_dl_a, device)
    results["Variant A (w/o Physics)"] = res_a

    # ==========================================================
    # Variant B: w/o Gradients (证明梯度引导的必要性)
    # ==========================================================
    cfg_b = ModelConfig(use_physics=True, cond_type='raw_only', in_channels=2) # 只有 noisy + raw
    train_dl_b = DataLoader(GPRHDF5Dataset(h5_path, split='train', cond_type='raw_only'), batch_size=cfg_b.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl_b = DataLoader(GPRHDF5Dataset(h5_path, split='test', cond_type='raw_only'), batch_size=cfg_b.batch_size, shuffle=False, num_workers=4)
    res_b = train_and_eval_variant("Variant B (wo Gradients)", cfg_b, train_dl_b, test_dl_b, device)
    results["Variant B (w/o Gradients)"] = res_b

    # ==========================================================
    # Variant C: PIFC-Diff Full (完全体)
    # ==========================================================
    cfg_c = ModelConfig(use_physics=True, cond_type='full', in_channels=4)
    res_c = train_and_eval_variant("Variant C (PIFC-Diff Full)", cfg_c, train_dl_a, test_dl_a, device)
    results["PIFC-Diff (Full)"] = res_c

    # ==========================================================
    # 汇总并保存表格
    # ==========================================================
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv("ablation_results/ablation_metrics.csv")
    print("\n🎉 所有消融实验运行完毕！结果已保存至 ablation_results/ablation_metrics.csv")
    print(df)

if __name__ == "__main__":
    main()