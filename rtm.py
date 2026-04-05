# -*- coding: utf-8 -*-
"""
RTM v10 - Clean Version without FK (FK filtering completely removed)
Focus: Enhanced iterative mean background removal + mild gain + PCA
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, laplace
from sklearn.decomposition import PCA

from train import ConditionalLatentDiffusion, ModelConfig


def set_sci_style():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14

set_sci_style()


# ─────────────────────────────────────────────
# Advanced Preprocessing (No FK)
# ─────────────────────────────────────────────
def advanced_preprocess(bscan, dt, n_pca=40, gain_factor=0.30):
    nt, nx = bscan.shape
    t = np.arange(nt) * dt

    print(f"  Preprocess: shape={bscan.shape}, n_pca={n_pca}, gain={gain_factor} (no FK)")

    # 1. Iterative horizontal mean background removal
    for _ in range(6):                     
        mean_bg = np.mean(bscan, axis=1, keepdims=True)
        bscan = bscan - 0.90 * mean_bg

    # 2. Mild time gain (prevents amplifying shallow noise)
    gain = np.exp(gain_factor * t / 12e-9)[:, np.newaxis]
    bscan = bscan * gain

    # 3. Bandpass filter
    b, a = signal.butter(4, [35e6, 170e6], btype='bandpass', fs=1.0/dt)
    bscan = signal.filtfilt(b, a, bscan, axis=0)

    # 4. Enhanced PCA low-rank background removal
    data = bscan.T
    pca = PCA(n_components=n_pca)
    low_rank = pca.fit_transform(data)
    bg = low_rank @ pca.components_
    bscan = bscan - bg.T

    # 5. Independent trace normalization
    bscan = bscan - bscan.mean(axis=0)
    bscan = bscan / (bscan.std(axis=0) + 1e-8)

    return bscan


# ─────────────────────────────────────────────
# RTM Core
# ─────────────────────────────────────────────
@torch.no_grad()
def reverse_time_migration(b_scan, dt, dx, dz, velocity_map, device='cuda', apply_laplacian=True):
    b_scan_t = torch.tensor(b_scan, dtype=torch.float32, device=device)
    v_map = torch.tensor(velocity_map, dtype=torch.float32, device=device)

    nt, nx = b_scan_t.shape
    nz, _ = v_map.shape
    v_max = v_map.max().item()

    cfl_factor = 0.45
    dt_stable_limit = cfl_factor * min(dx, dz) / v_max

    if dt > dt_stable_limit:
        sub_steps = int(np.ceil(dt / (dt_stable_limit * 0.9)))
        dt_new = dt / sub_steps
        nt_new = nt * sub_steps
        b_scan_t = F.interpolate(b_scan_t.T.unsqueeze(0), size=nt_new, mode='linear', align_corners=False).squeeze(0).T
        dt = dt_new
        nt = nt_new

    p = torch.zeros((nz, nx), device=device)
    p_old = torch.zeros((nz, nx), device=device)

    v_dt_sq = (v_map * dt) ** 2
    idx_x_sq = 1.0 / dx ** 2
    idx_z_sq = 1.0 / dz ** 2

    for t_idx in range(nt - 1, -1, -1):
        d2p_dx2 = torch.zeros_like(p)
        d2p_dz2 = torch.zeros_like(p)
        d2p_dx2[:, 1:-1] = (p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:]) * idx_x_sq
        d2p_dz2[1:-1, :] = (p[:-2, :] - 2 * p[1:-1, :] + p[2:, :]) * idx_z_sq

        p_new = 2 * p - p_old + v_dt_sq * (d2p_dx2 + d2p_dz2)
        p_new[0, :] += b_scan_t[t_idx, :]

        p_old, p = p, p_new

    mig = p.cpu().numpy()

    if apply_laplacian:
        mig = -laplace(mig) * 1.0   # Mild Laplacian

    return mig


# ─────────────────────────────────────────────
# Main Process
# ─────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    base_dir = "/home/rtm"
    eps_folder = os.path.join(base_dir, "1")
    bscan_folder = os.path.join(base_dir, "3")
    model_path = "best_model.pt"

    domain_x = 1.5
    domain_z = 0.5
    t_window = 20e-9
    nt_bscan = 256
    nz_eps = 200
    nx_eps = 600

    dt = t_window / nt_bscan
    dz = domain_z / nz_eps
    dx = domain_x / nx_eps
    c0 = 3e8

    print(f"Physical parameters: dt={dt*1e12:.2f}ps, dx={dx*1e3:.2f}mm, dz={dz*1e3:.2f}mm\n")

    config = ModelConfig()
    model = ConditionalLatentDiffusion(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    mat_files = ['2437.mat', '500.mat', '44.mat']
    scenarios = ["Scenario 1", "Scenario 2", "Scenario 3"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)

    for row, filename in enumerate(mat_files):
        print(f"🔄 Processing: {scenarios[row]} ({filename})")

        eps_data = sio.loadmat(os.path.join(eps_folder, filename))
        bscan_data = sio.loadmat(os.path.join(bscan_folder, filename))

        eps_r = eps_data.get('data1', list(eps_data.values())[-1]).squeeze()
        raw_bscan = bscan_data.get('mask', bscan_data.get('IM', list(bscan_data.values())[-1])).squeeze()

        if raw_bscan.shape[0] < raw_bscan.shape[1]:
            raw_bscan = raw_bscan.T

        if eps_r.shape != (nz_eps, nx_eps):
            eps_r = F.interpolate(torch.tensor(eps_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                  size=(nz_eps, nx_eps), mode='bilinear', align_corners=False).squeeze().numpy()

        if raw_bscan.shape[1] != nx_eps:
            raw_bscan = F.interpolate(torch.tensor(raw_bscan, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                      size=(nt_bscan, nx_eps), mode='bilinear', align_corners=False).squeeze().numpy()

        # PIFC-Diff Enhancement
        raw_tensor = torch.from_numpy(raw_bscan).float().unsqueeze(0).unsqueeze(0)
        grad_t = torch.gradient(raw_tensor, dim=2)[0]
        grad_x = torch.gradient(raw_tensor, dim=3)[0]
        cond = torch.cat([raw_tensor, grad_t, grad_x], dim=1).to(device)

        with torch.no_grad():
            enhanced_bscan = model.sample(cond).cpu().numpy().squeeze()

        if enhanced_bscan.shape != (nt_bscan, nx_eps):
            enhanced_bscan = F.interpolate(torch.tensor(enhanced_bscan, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                           size=(nt_bscan, nx_eps), mode='bilinear', align_corners=False).squeeze().numpy()

        # Preprocessing (No FK)
        raw_in = advanced_preprocess(raw_bscan, dt, n_pca=40, gain_factor=0.30)
        enh_in = advanced_preprocess(enhanced_bscan, dt, n_pca=40, gain_factor=0.30)

        # Velocity model
        eps_smooth = gaussian_filter(eps_r, sigma=(3, 5))
        vel_map = (c0 / np.sqrt(np.clip(eps_smooth, 1.0, None))) / 2.0

        print(f"  vel_map: {vel_map.min():.0f} ~ {vel_map.max():.0f} m/s")

        mig_raw = reverse_time_migration(raw_in, dt, dx, dz, vel_map, device)
        mig_enh = reverse_time_migration(enh_in, dt, dx, dz, vel_map, device)

        extent = [0, domain_x, domain_z, 0]
        x_coords = np.linspace(0, domain_x, nx_eps)
        z_coords = np.linspace(0, domain_z, nz_eps)

        levels = [np.median(eps_r) - 1.5 * np.std(eps_r), np.median(eps_r) + 1.5 * np.std(eps_r)]

        axes[row, 0].imshow(eps_r, cmap='turbo', aspect='auto', extent=extent)
        axes[row, 0].set_ylabel(f"{scenarios[row]}\nDepth (m)", fontweight='bold', labelpad=15)

        vmax_raw = np.percentile(np.abs(mig_raw), 98) or 1.0
        vmax_enh = np.percentile(np.abs(mig_enh), 98) or 1.0

        axes[row, 1].imshow(mig_raw, cmap='seismic', aspect='auto', vmin=-vmax_raw, vmax=vmax_raw, extent=extent)
        axes[row, 1].contour(x_coords, z_coords, eps_r, levels=levels, colors='magenta', linestyles='--', linewidths=1.2, alpha=0.85)

        axes[row, 2].imshow(mig_enh, cmap='seismic', aspect='auto', vmin=-vmax_enh, vmax=vmax_enh, extent=extent)
        axes[row, 2].contour(x_coords, z_coords, eps_r, levels=levels, colors='magenta', linestyles='--', linewidths=1.5, alpha=0.9)

        if row == 0:
            axes[row, 0].set_title("(a) True Model", fontweight='bold')
            axes[row, 1].set_title("(b) Raw RTM", fontweight='bold')
            axes[row, 2].set_title("(c) PIFC-Diff RTM", fontweight='bold')

    for ax in axes[-1, :]:
        ax.set_xlabel("Distance (m)", fontweight='bold')

    plt.tight_layout()
    out_path = "Fig6_Migration_v10_no_fk.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure generated: {out_path}")


if __name__ == "__main__":
    main()
