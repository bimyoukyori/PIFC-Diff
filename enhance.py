import os
import glob
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple

# =============================================================================
# 1. 导入模型定义 (从 train.py)
# =============================================================================
try:
    from train import (
        ModelConfig,
        ConditionalLatentDiffusion,
        AdaptiveLoss,  
        PhysicsGuidedLoss
    )
    print("✓ 成功从 train.py 导入模型定义")
except ImportError:
    print("❌ 错误：找不到 'train.py'")
    print("请确保此脚本与 'train.py' 在同一文件夹中")
    exit()

# =============================================================================
# 2. 工具函数
# =============================================================================

def load_gpr_from_out(file_path: str) -> np.ndarray:
    """从 .out 或 .h5 文件加载 GPR 数据"""
    print(f"正在加载文件: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 如果是训练用的 HDF5 (通常是 group 结构)
            if 'sample_0' in f:
                print("✓ 检测到训练集格式 HDF5，加载 sample_0/raw ...")
                return f['sample_0']['raw'][:]

            # 2. 如果是原始 .out 文件 (尝试常见字段)
            candidates = ['/rxs/rx1/Ez', 'Ez', '/rxs/rx1/Ex', 'Ex']
            data = None
            for key in candidates:
                if key in f:
                    data = f[key][:]
                    print(f"✓ 找到字段: {key}, 形状: {data.shape}")
                    break
            
            if data is None:
                print(f"❌ 未找到有效字段，文件包含: {list(f.keys())}")
                return None
            return data
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """归一化到 [-1, 1]"""
    d_min, d_max = data.min(), data.max()
    if abs(d_max - d_min) < 1e-9:
        return np.zeros_like(data), d_min, d_max
    norm_data = 2 * (data - d_min) / (d_max - d_min) - 1
    return norm_data, d_min, d_max

def denormalize_data(norm_data: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """反归一化"""
    return (norm_data + 1) / 2 * (d_max - d_min) + d_min

def visualize_results(results: Dict[str, np.ndarray], save_path: str):
    """画图并保存"""
    raw = results['raw']
    enhanced = results['enhanced']
    diff = results['diff']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    vmin = min(raw.min(), enhanced.min())
    vmax = max(raw.max(), enhanced.max())
    
    im0 = axes[0].imshow(raw, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Raw Data')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(enhanced, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Diffusion Enhanced')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(diff, cmap='coolwarm', aspect='auto')
    axes[2].set_title('Difference (Enhanced - Raw)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 结果图已保存: {save_path}")

# =============================================================================
# 3. 推理核心类
# =============================================================================

class GPREnhancer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.config = ModelConfig() 
        
        print(f"✓ 初始化模型 (Device: {device})...")
        self.model = ConditionalLatentDiffusion(self.config)
        
        print(f"✓ 加载权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint['model_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()
        
        
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            print("🚀 启用 BF16 加速推理")
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

    @torch.no_grad()
    def enhance(self, raw_data: np.ndarray, patch_size=(256, 100)) -> np.ndarray:
        h, w = raw_data.shape
        ph, pw = patch_size
        
        norm_raw, d_min, d_max = normalize_data(raw_data)
        
        enhanced_map = np.zeros_like(norm_raw)
        count_map = np.zeros_like(norm_raw)
        
        step_h = ph // 2
        step_w = pw // 2
        
        coords = []
        for i in range(0, h - ph + 1, step_h):
            for j in range(0, w - pw + 1, step_w):
                coords.append((i, j))
        
        
        batch_size = 32
        
        print(f"🚀 开始增强... (Total patches: {len(coords)})")
        
        for b in tqdm(range(0, len(coords), batch_size)):
            batch_coords = coords[b : b + batch_size]
            
            raw_patches = []
            grad_t_patches = []
            grad_x_patches = []
            
            for (i, j) in batch_coords:
                patch = norm_raw[i:i+ph, j:j+pw]
                
                raw_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                grad_t = torch.gradient(raw_tensor, dim=2)[0]
                grad_x = torch.gradient(raw_tensor, dim=3)[0]
                
                raw_patches.append(raw_tensor)
                grad_t_patches.append(grad_t)
                grad_x_patches.append(grad_x)
            
            raw_batch = torch.cat(raw_patches, dim=0).to(self.device)
            grad_t_batch = torch.cat(grad_t_patches, dim=0).to(self.device)
            grad_x_batch = torch.cat(grad_x_patches, dim=0).to(self.device)
            
            condition = torch.cat([raw_batch, grad_t_batch, grad_x_batch], dim=1)
            
            with torch.amp.autocast('cuda', dtype=self.dtype):
                enhanced_batch = self.model.sample(condition)
            
            enhanced_numpy = enhanced_batch.cpu().float().numpy()
            for idx, (i, j) in enumerate(batch_coords):
                enhanced_map[i:i+ph, j:j+pw] += enhanced_numpy[idx, 0]
                count_map[i:i+ph, j:j+pw] += 1.0
        
        mask = count_map > 0
        enhanced_map[mask] /= count_map[mask]
        enhanced_map[~mask] = norm_raw[~mask]
        
        final_result = denormalize_data(enhanced_map, d_min, d_max)
        return final_result

# =============================================================================
# 4. 主执行入口
# =============================================================================

def main():
    # 🔥 配置区
    CHECKPOINT_PATH = "best_model.pt"
    
    INPUT_DIR = "/home/test_data" 
    
    OUTPUT_DIR = "results"
    
    # ==========================================
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 找不到模型文件: {CHECKPOINT_PATH}")
        return
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入文件夹: {INPUT_DIR}")
        print("请在 enhance.py 中修改 INPUT_DIR 为正确的文件夹路径")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 查找所有支持的文件
    print(f"正在扫描文件夹: {INPUT_DIR} ...")
    # 递归查找所有 .out 和 .h5 文件
    patterns = [os.path.join(INPUT_DIR, "**", "*.out"), 
                os.path.join(INPUT_DIR, "**", "*.h5")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    
    # 过滤掉 training 数据 (可选)
    # files = [f for f in files if "training" not in f]
    
    files = sorted(list(set(files))) # 去重并排序
    print(f"📂 找到 {len(files)} 个待处理文件")
    
    if len(files) == 0:
        print("⚠️ 没有找到 .out 或 .h5 文件，请检查路径或上传数据。")
        return

    # 2. 初始化模型 (只加载一次)
    enhancer = GPREnhancer(CHECKPOINT_PATH)
    
    # 3. 批量循环
    success_count = 0
    
    for idx, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        # 获取不带扩展名的文件名
        file_stem = os.path.splitext(file_name)[0]
        
        # 结果保存路径
        save_path_npy = os.path.join(OUTPUT_DIR, f"{file_stem}_enhanced.npy")
        save_path_png = os.path.join(OUTPUT_DIR, f"{file_stem}_enhanced.png")
        
        # 跳过已存在的结果 (断点续传)
        if os.path.exists(save_path_npy):
            print(f"⏭️ [{idx+1}/{len(files)}] 跳过已存在: {file_name}")
            continue
            
        print(f"\n🚀 [{idx+1}/{len(files)}] 处理中: {file_name}")
        
        # 加载数据
        raw_data = load_gpr_from_out(file_path)
        if raw_data is None:
            print(f"⚠️ 跳过无法读取的文件: {file_name}")
            continue
            
        # 增强
        try:
            enhanced_data = enhancer.enhance(raw_data)
            
            # 保存结果图
            visualize_results({
                'raw': raw_data,
                'enhanced': enhanced_data,
                'diff': enhanced_data - raw_data
            }, save_path_png)
            
            # 保存结果数据
            np.save(save_path_npy, enhanced_data)
            print(f"✓ 完成: {file_name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败 {file_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 批量处理完成！成功: {success_count}/{len(files)}")

if __name__ == "__main__":
    main()