# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segyio
from tqdm import tqdm

# 导入你训练脚本中的网络结构和配置
# 假设你的训练代码文件名为 train.py (如果不一样请修改这里)
from train import ModelConfig, ConditionalLatentDiffusion

def read_sgy_data(sgy_path):
    """读取 SGY 文件并返回 2D numpy 数组 (Time_Samples, Traces)"""
    print(f"📖 正在读取 SGY 文件: {sgy_path}")
    # 使用 ignore_geometry=True 忽略复杂的 3D 几何信息，直接按 2D 剖面读取
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        # 将所有单道数据叠成一个矩阵
        data = np.stack([trace for trace in f.trace])
        data = data.T  # 转置为 (Time, Traces)
    print(f"✅ SGY 读取成功，原始数据维度: {data.shape}")
    return data

def extract_condition_features(raw_2d):
    """提取梯度特征，与训练集逻辑严格保持一致"""
    # 转换为 tensor
    raw_tensor = torch.from_numpy(raw_2d).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # 计算梯度
    grad_t = torch.gradient(raw_tensor, dim=2)[0]
    grad_x = torch.gradient(raw_tensor, dim=3)[0]
    
    # 拼接为 3 通道 condition [1, 3, H, W]
    condition = torch.cat([raw_tensor, grad_t, grad_x], dim=1)
    return condition

def create_hanning_window(h, w, device):
    """创建二维汉明窗，用于消除拼接缝隙"""
    window_h = np.hanning(h)
    window_w = np.hanning(w)
    window_2d = np.outer(window_h, window_w)
    return torch.from_numpy(window_2d).float().to(device)

@torch.no_grad()
def process_large_bscan(model, raw_img, config, device):
    """大图的 Overlap-and-Add 无缝拼接推断"""
    H_img, W_img = raw_img.shape
    patch_h, patch_w = 256, 100
    stride_h, stride_w = patch_h // 2, patch_w // 2  # 50% 重叠率
    
    # 计算需要 Pad 的尺寸，确保边缘能被整除
    pad_h = (stride_h - (H_img - patch_h) % stride_h) % stride_h
    pad_w = (stride_w - (W_img - patch_w) % stride_w) % stride_w
    
    # 对原图进行 Padding (边缘反射)
    raw_padded = np.pad(raw_img, ((0, pad_h), (0, pad_w)), mode='reflect')
    H_pad, W_pad = raw_padded.shape
    print(f"📐 经过 Padding 后的维度: ({H_pad}, {W_pad})")
    
    # 初始化空白画布和权重画布
    canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)
    weight_canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)
    window = create_hanning_window(patch_h, patch_w, device).view(1, 1, patch_h, patch_w)
    
    # 计算需要遍历的网格数量
    num_steps_h = (H_pad - patch_h) // stride_h + 1
    num_steps_w = (W_pad - patch_w) // stride_w + 1
    total_patches = num_steps_h * num_steps_w
    
    print(f"🧩 开始切块推断，共划分为 {total_patches} 个 Patches...")
    
    model.eval()
    patch_idx = 0
    # 逐块进行扩散去噪
    with tqdm(total=total_patches, desc="Patch Inference") as pbar:
        for i in range(num_steps_h):
            for j in range(num_steps_w):
                y = i * stride_h
                x = j * stride_w
                
                # 裁剪 Patch 并归一化 (局部归一化非常重要！)
                patch_raw = raw_padded[y:y+patch_h, x:x+patch_w]
                patch_mean, patch_std = patch_raw.mean(), patch_raw.std() + 1e-8
                patch_norm = (patch_raw - patch_mean) / patch_std
                
                # 提取特征并推断
                condition = extract_condition_features(patch_norm).to(device)
                # DDPM sample 是个漫长的过程，每个 patch 都要跑 1000 步
                pred_patch = model.sample(condition) 
                
                # 反归一化，恢复局部真实的能量振幅
                pred_patch = pred_patch * patch_std + patch_mean
                
                # 将预测结果乘上窗口权重，累加到画布上
                canvas[:, :, y:y+patch_h, x:x+patch_w] += pred_patch * window
                weight_canvas[:, :, y:y+patch_h, x:x+patch_w] += window
                
                patch_idx += 1
                pbar.update(1)
                
    # 消除重叠区域的权重累加影响
    final_output = canvas / (weight_canvas + 1e-8)
    
    # 裁掉之前 Padding 的部分，恢复原图尺寸
    final_output = final_output[:, :, :H_img, :W_img].cpu().numpy().squeeze()
    return final_output

def main():
    # ================= 配置路径 =================
    model_weight_path = "best_model.pt" # 你训练保存的权重
    sgy_file_path = "bridge_deck_400MHz.sgy" # 替换为你的 SGY 文件路径
    # ============================================
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ModelConfig()
    
    # 加载预训练模型
    print("⏳ 正在加载预训练 PIFC-Diff 模型...")
    model = ConditionalLatentDiffusion(config).to(device)
    checkpoint = torch.load(model_weight_path, map_location=device)
    
    # 兼容处理（如果你保存时存成了 dict）
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ 模型加载完毕！")
    
    # 读取实测数据
    raw_data = read_sgy_data(sgy_file_path)
    
    # 考虑到实测大图跑 DDPM 极其耗时，如果是测试，可以先截取一部分剖面
    # raw_data = raw_data[:, :500] # 测试时打开注释：只取前 500 道
    
    # 全局归一化保护
    raw_mean, raw_std = raw_data.mean(), raw_data.std() + 1e-8
    raw_norm_global = (raw_data - raw_mean) / raw_std
    
    # 执行无缝切块推断
    enhanced_data = process_large_bscan(model, raw_norm_global, config, device)
    
    # 可视化对比并保存
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    # 取 -2 到 2 个标准差来增强对比度
    vmin, vmax = np.percentile(raw_norm_global, [2, 98])
    plt.imshow(raw_norm_global, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title("Original Field GPR (Noisy)", fontsize=14)
    plt.xlabel("Trace Number")
    plt.ylabel("Time Sample")
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_data, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title("PIFC-Diff Enhanced (Zero-Shot)", fontsize=14)
    plt.xlabel("Trace Number")
    
    plt.tight_layout()
    plt.savefig("Field_Data_ZeroShot_Result.png", dpi=300)
    plt.show()
    print("实测数据增强完成！结果已保存为 Field_Data_ZeroShot_Result.png")

if __name__ == "__main__":
    main()