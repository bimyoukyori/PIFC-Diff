# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

# 从你的训练代码中导入模型和配置
from train import ConditionalLatentDiffusion, ModelConfig

# ==========================================================
# 1. 真实波动方程 RTM 算法 
# ==========================================================
def set_sci_style():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

set_sci_style()

@torch.no_grad()
def reverse_time_migration(b_scan, dt, dx, dz, velocity, device='cuda'):
    """
    基于声波方程的逆时偏移 (Post-stack RTM)
    """
    b_scan_tensor = torch.tensor(b_scan, dtype=torch.float32, device=device)
    nt, nx = b_scan_tensor.shape
    
    # 爆炸反射面模型：速度减半
    v = velocity / 2.0 
    
    # 【自动保护】CFL 稳定性检查与时间重采样
    cfl = v * dt / min(dx, dz)
    if cfl > 0.707:
        print(f"⚠️ CFL不满足 ({cfl:.2f} > 0.707)，正在内部自动增加时间采样点以确保稳定...")
        sub_steps = int(np.ceil(cfl / 0.5)) 
        dt_new = dt / sub_steps
        nt_new = nt * sub_steps
        # 在时间轴上插值
        b_scan_tensor = torch.nn.functional.interpolate(
            b_scan_tensor.unsqueeze(0).unsqueeze(0), 
            size=(nt_new, nx), mode='bilinear', align_corners=False
        ).squeeze()
        dt = dt_new
        nt = nt_new
        cfl = v * dt / min(dx, dz)
        print(f"✅ 已调整为: dt={dt:.2e}s, nt={nt}, 新CFL={cfl:.2f}")

    # 计算深度网格
    max_depth = nt * dt * v
    nz = int(max_depth / dz) + 20 
    
    p = torch.zeros((nz, nx), device=device)
    p_old = torch.zeros((nz, nx), device=device)
    p_new = torch.zeros((nz, nx), device=device)
    
    c_x = (v * dt / dx) ** 2
    c_z = (v * dt / dz) ** 2

    print(f"⏳ 正在执行波场逆推 (网格: {nz}x{nx}, 步数: {nt})...")
    for t in range(nt - 1, -1, -1):
        d2p_dx2 = torch.zeros_like(p)
        d2p_dz2 = torch.zeros_like(p)
        
        d2p_dx2[:, 1:-1] = p[:, :-2] - 2*p[:, 1:-1] + p[:, 2:]
        d2p_dz2[1:-1, :] = p[:-2, :] - 2*p[1:-1, :] + p[2:, :]
        
        p_new = 2 * p - p_old + c_x * d2p_dx2 + c_z * d2p_dz2
        
        # 在表面注入接收到的数据
        p_new[2, :] += b_scan_tensor[t, :]
        
        p_old = p
        p = p_new
        
    return p.cpu().numpy()

# ==========================================================
# 2. 核心对接：加载模型、生成数据、执行偏移
# ==========================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h5_path = "/home/gpr_training_data.h5"
    model_path = "best_model.pt" 
    
    # ------------------ 第一步：加载模型 ------------------
    config = ModelConfig()
    model = ConditionalLatentDiffusion(config).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
        print("✅ 模型加载成功！")
    else:
        print(f"❌ 找不到模型权重 {model_path}，请确认路径！")
        return
    model.eval()

    # ------------------ 第二步：设定要抽取的 3 个样本 ID ------------------
    with h5py.File(h5_path, 'r') as f:
        num_samples = f.attrs['num_samples']
        # 挑选最后测试集里的 3 个不同样本
        test_indices = [num_samples - 1, num_samples - 50, num_samples - 200] 

    # 创建一个 3行2列 的大画布 (宽8，高12，适合瘦高型矩阵数据展示)
    fig, axes = plt.subplots(len(test_indices), 2, figsize=(8, 12), sharex=True, sharey=True)
    
    # 物理参数
    dt = 1.0 / config.sampling_rate 
    velocity = config.wave_velocity * 1e9 
    dx = 0.05  
    dz = 0.05  
    display_start = 5
    display_end = 245  # 裁掉底部的无用白边

    global_max_limit = 0 # 用于统一百分比，确保 3 组图的 Colorbar 刻度一致

    print(f"📂 开始批量处理 {len(test_indices)} 组数据...")

    # ------------------ 第三步：循环处理这 3 组数据 ------------------
    for row_idx, sample_id in enumerate(test_indices):
        print(f"  -> 正在处理 Sample {sample_id}...")
        with h5py.File(h5_path, 'r') as f:
            grp = f[f"sample_{sample_id}"]
            raw = grp['raw'][:256, :100]      
            grad_t = grp['grad_t'][:256, :100]
            grad_x = grp['grad_x'][:256, :100]
            
        condition_tensor = torch.tensor(
            np.stack([raw, grad_t, grad_x]), dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pifc_tensor = model.sample(condition_tensor)
            pifc_data = pifc_tensor[0, 0].cpu().numpy() 

        # 归一化并翻转 (修复物理时间倒置，加上 .copy() 防止 strides 报错)
        raw_norm = np.flipud((raw - np.mean(raw)) / (np.std(raw) + 1e-8)).copy()
        pifc_norm = np.flipud((pifc_data - np.mean(pifc_data)) / (np.std(pifc_data) + 1e-8)).copy()

        # 执行 RTM
        mig_raw = reverse_time_migration(raw_norm, dt, dx, dz, velocity, device)
        mig_pifc = reverse_time_migration(pifc_norm, dt, dx, dz, velocity, device)

        # 截取有效深度，切除底部全白区域
        m_raw_disp = mig_raw[display_start:display_end, :]
        m_pifc_disp = mig_pifc[display_start:display_end, :]

        # 动态更新全局最大绝对值 (为了共用一个绝对对称的 Colorbar)
        current_limit = max(np.max(np.abs(m_raw_disp)), np.max(np.abs(m_pifc_disp))) * 0.8
        global_max_limit = max(global_max_limit, current_limit)

        # 缓存数据以便后面统一画图
        axes[row_idx, 0].m_data = m_raw_disp 
        axes[row_idx, 1].m_data = m_pifc_disp

        # 设置 Y 轴标签（每一行最左侧都有）
        axes[row_idx, 0].set_ylabel('Depth Sample')
        
        # X轴标签和底部的 (a), (b) 标记 (只在最后一行显示)
        if row_idx == len(test_indices) - 1:
            axes[row_idx, 0].set_xlabel('Trace Number')
            axes[row_idx, 1].set_xlabel('Trace Number')
            
            # 在底部图像的中央下方添加粗体 (a) 和 (b)
            # y=-0.35 确保文字在 X轴标签 'Trace Number' 的正下方
            axes[row_idx, 0].text(0.5, -0.25, '(a)', transform=axes[row_idx, 0].transAxes,
                                    ha='center', va='top', fontsize=20)
            axes[row_idx, 1].text(0.5, -0.25, '(b)', transform=axes[row_idx, 1].transAxes,
                                    ha='center', va='top', fontsize=20)

    # ------------------ 第四步：统一刷新所有子图的对称阈值并绘制 Colorbar ------------------
    for row_idx in range(len(test_indices)):
        # 彻底消灭了未使用的 im_raw 灰色变量提示
        axes[row_idx, 0].imshow(axes[row_idx, 0].m_data, cmap='seismic', aspect='auto', vmin=-global_max_limit, vmax=global_max_limit)
        im_pifc = axes[row_idx, 1].imshow(axes[row_idx, 1].m_data, cmap='seismic', aspect='auto', vmin=-global_max_limit, vmax=global_max_limit)

    # ------------------ 第五步：布局调整与共用 Colorbar ------------------
    # right=0.85 保证主图只占左边 85% 的宽度，给 colorbar 留位置；bottom=0.12 防止切掉 (a)(b)
    plt.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.92, wspace=0.1, hspace=0.1)
    
    # 把 Colorbar 放在最右侧的独立坐标系里
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7]) 
    cbar = fig.colorbar(im_pifc, cax=cbar_ax)
    cbar.set_label('Normalized Amplitude', fontsize=16, labelpad=10, fontweight='bold')

    # 保存结果
    SAVE_DIR = "paper_figures"
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path_png = os.path.join(SAVE_DIR, 'Fig5_Migration_Comparison_Multi.png')
    save_path_pdf = os.path.join(SAVE_DIR, 'Fig5_Migration_Comparison_Multi.pdf')
    
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    
    print(f"已完成: \n  - {save_path_png}\n  - {save_path_pdf}")

if __name__ == "__main__":
    main()