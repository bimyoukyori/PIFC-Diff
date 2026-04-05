# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from train import ModelConfig, ConditionalLatentDiffusion

# ================= SCI Plot Global Font Settings =================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['mathtext.fontset'] = 'stix'
# ========================================================

def read_sgy_data(sgy_path):
    try:
        import segyio
        with segyio.open(sgy_path, "r", ignore_geometry=True, strict=False) as f:
            data = np.stack([trace for trace in f.trace])
            return data.T
    except Exception as e:
        import obspy
        stream = obspy.read(sgy_path)
        base_length = len(stream[0].data)
        traces = []
        for tr in stream:
            tr_data = tr.data
            if len(tr_data) > base_length:
                traces.append(tr_data[:base_length])
            elif len(tr_data) < base_length:
                traces.append(np.pad(tr_data, (0, base_length - len(tr_data)), 'constant'))
            else:
                traces.append(tr_data)
        return np.stack(traces).T

def extract_condition_features(raw_2d):
    raw_tensor = torch.from_numpy(raw_2d).float().unsqueeze(0).unsqueeze(0)
    grad_t = torch.gradient(raw_tensor, dim=2)[0]
    grad_x = torch.gradient(raw_tensor, dim=3)[0]
    return torch.cat([raw_tensor, grad_t, grad_x], dim=1)

def create_hanning_window(h, w, device):
    window_2d = np.outer(np.hanning(h), np.hanning(w))
    return torch.from_numpy(window_2d).float().to(device)

@torch.no_grad()
def process_large_bscan_single(model, raw_img, device):
    """Single patch inference, returns stitched result (original logic)."""
    H_img, W_img = raw_img.shape
    patch_h, patch_w = 256, 100
    stride_h, stride_w = patch_h // 2, patch_w // 2

    pad_top = stride_h
    pad_bottom = stride_h + (stride_h - (H_img % stride_h)) % stride_h
    pad_left = stride_w
    pad_right = stride_w + (stride_w - (W_img % stride_w)) % stride_w

    raw_padded = np.pad(raw_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
    H_pad, W_pad = raw_padded.shape

    canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)
    weight_canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)
    window = create_hanning_window(patch_h, patch_w, device).view(1, 1, patch_h, patch_w)

    num_steps_h = (H_pad - patch_h) // stride_h + 1
    num_steps_w = (W_pad - patch_w) // stride_w + 1

    for i in range(num_steps_h):
        for j in range(num_steps_w):
            y, x = i * stride_h, j * stride_w
            patch_raw = raw_padded[y:y+patch_h, x:x+patch_w]
            condition = extract_condition_features(patch_raw).to(device)
            pred_patch = model.sample(condition)
            canvas[:, :, y:y+patch_h, x:x+patch_w] += pred_patch * window
            weight_canvas[:, :, y:y+patch_h, x:x+patch_w] += window

    final_output = canvas / (weight_canvas + 1e-8)
    final_cropped = final_output[:, :, pad_top:pad_top+H_img, pad_left:pad_left+W_img]
    return final_cropped.cpu().numpy().squeeze()


@torch.no_grad()
def process_large_bscan_ensemble(model, raw_img, device, n_ensemble=8):
    """
    Ensemble inference: Run n_ensemble times on the same input and average.
    
    Concept: Diffusion models have random start points, causing random artifacts 
    in single runs. Averaging cancels random noise (expected value 0) while 
    preserving deterministic physical structures.

    n_ensemble tips:
      - Quick check: 5
      - Paper quality: 8~10
      - Low VRAM: Start with 5
    """
    H_img, W_img = raw_img.shape
    patch_h, patch_w = 256, 100
    stride_h, stride_w = patch_h // 2, patch_w // 2

    pad_top = stride_h
    pad_bottom = stride_h + (stride_h - (H_img % stride_h)) % stride_h
    pad_left = stride_w
    pad_right = stride_w + (stride_w - (W_img % stride_w)) % stride_w

    raw_padded = np.pad(raw_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
    H_pad, W_pad = raw_padded.shape

    num_steps_h = (H_pad - patch_h) // stride_h + 1
    num_steps_w = (W_pad - patch_w) // stride_w + 1
    total_patches = num_steps_h * num_steps_w

    window = create_hanning_window(patch_h, patch_w, device).view(1, 1, patch_h, patch_w)

    # Accumulator: Float64 accumulation on CPU to reduce precision loss
    accum = np.zeros((H_pad, W_pad), dtype=np.float64)
    weight_map = np.zeros((H_pad, W_pad), dtype=np.float64)
    window_np = window.squeeze().cpu().numpy().astype(np.float64)

    model.eval()

    print(f"  🔁 Ensemble Inference (n={n_ensemble}), total {total_patches} patches × {n_ensemble} runs...")

    for run_idx in range(n_ensemble):
        canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)
        weight_canvas = torch.zeros((1, 1, H_pad, W_pad), device=device)

        with tqdm(total=total_patches,
                  desc=f"  Run {run_idx+1}/{n_ensemble}",
                  leave=False, colour='cyan') as pbar:
            for i in range(num_steps_h):
                for j in range(num_steps_w):
                    y, x = i * stride_h, j * stride_w
                    patch_raw = raw_padded[y:y+patch_h, x:x+patch_w]
                    condition = extract_condition_features(patch_raw).to(device)
                    pred_patch = model.sample(condition)
                    canvas[:, :, y:y+patch_h, x:x+patch_w] += pred_patch * window
                    weight_canvas[:, :, y:y+patch_h, x:x+patch_w] += window
                    pbar.update(1)

        single_result = (canvas / (weight_canvas + 1e-8)).cpu().numpy().squeeze()
        # Direct accumulation (unweighted), divide by n_ensemble later
        accum += single_result.astype(np.float64)

        # Free VRAM after each run
        del canvas, weight_canvas
        torch.cuda.empty_cache()

    # Calculate average
    ensemble_mean = (accum / n_ensemble).astype(np.float32)
    final_cropped = ensemble_mean[pad_top:pad_top+H_img, pad_left:pad_left+W_img]
    return final_cropped


def post_process_enhanced(enhanced_data):
    """Original post-processing logic, unchanged."""
    enhanced_data = enhanced_data - np.mean(enhanced_data, axis=0, keepdims=True)
    p_max = np.percentile(np.abs(enhanced_data), 99.0)
    enhanced_data = np.tanh(enhanced_data / (p_max + 1e-8)) * p_max
    enhanced_data = gaussian_filter(enhanced_data, sigma=(1.0, 0.5))
    return enhanced_data


def process_single_sgy(sgy_path, model, config, device, mute_cutoff=80, n_ensemble=8):
    print(f"\n🚀 Start processing: {sgy_path}")
    raw_data = read_sgy_data(sgy_path)[:, :800]

    bg_trace = raw_data.mean(axis=1, keepdims=True)
    raw_data = raw_data - bg_trace
    raw_data[:mute_cutoff, :] = 0.0

    row_rms = np.sqrt(np.mean(raw_data**2, axis=1))
    window_sz = 31
    padded_rms = np.pad(row_rms, (window_sz//2, window_sz//2), mode='edge')
    smoothed_rms = np.convolve(padded_rms, np.ones(window_sz)/window_sz, mode='valid').reshape(-1, 1)
    smoothed_rms = np.clip(smoothed_rms, a_min=1e-5, a_max=None)

    gained_data = raw_data / smoothed_rms
    gained_data[:mute_cutoff, :] = 0.0

    max_val = np.percentile(np.abs(gained_data), 99)
    raw_norm_global = np.clip(gained_data / max_val, -1.0, 1.0)

    H_orig, W_orig = raw_norm_global.shape
    tensor_img = torch.from_numpy(raw_norm_global).unsqueeze(0).unsqueeze(0).float()
    scaled_img = F.interpolate(tensor_img, size=(256, W_orig), mode='bicubic', align_corners=False)
    scaled_numpy = scaled_img.squeeze().numpy()

    # ✅ Replace single inference with ensemble inference
    enhanced_scaled = process_large_bscan_ensemble(model, scaled_numpy, device, n_ensemble=n_ensemble)

    enhanced_tensor = torch.from_numpy(enhanced_scaled).unsqueeze(0).unsqueeze(0).float()
    final_enhanced = F.interpolate(enhanced_tensor, size=(H_orig, W_orig), mode='bicubic', align_corners=False).squeeze().numpy()

    final_enhanced = post_process_enhanced(final_enhanced)

    return raw_norm_global[mute_cutoff:, :], final_enhanced[mute_cutoff:, :]


def main():
    model_weight_path = "best_model.pt"
    sgy_files = ["Path12.sgy", "Path2.sgy", "Path4.sgy"]

    # ✅ Adjust ensemble count here
    #    Start with 5 for quick check, use 8 for paper figures
    N_ENSEMBLE = 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ModelConfig()
    model = ConditionalLatentDiffusion(config).to(device)

    checkpoint = torch.load(model_weight_path, map_location=device, weights_only=False)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ Model loaded! Ensemble n={N_ENSEMBLE}")

    results_raw = []
    results_enh = []

    for sgy in sgy_files:
        crop_raw, crop_enh = process_single_sgy(
            sgy, model, config, device,
            mute_cutoff=80, n_ensemble=N_ENSEMBLE
        )
        results_raw.append(crop_raw)
        results_enh.append(crop_enh)
        torch.cuda.empty_cache()

    # =================================================================
    # Visualization (original logic, only changed percentile of vmax_row1)
    # =================================================================
    print("\n🎨 Rendering SCI-level high-res 2x3 grid...")
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    labels = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
    label_size = 15
    tick_size = 13
    title_size = 17

    vmax_row0 = np.max([np.percentile(np.abs(d), 94.0) for d in results_raw])
    # Changed 98.5 -> 96.0 to enhance bottom row contrast and clean background
    vmax_row1 = np.max([np.percentile(np.abs(d), 94.0) for d in results_enh])

    for row in range(2):
        vmax_val = vmax_row0 if row == 0 else vmax_row1

        for col in range(3):
            ax = axs[row, col]
            data_to_plot = results_raw[col] if row == 0 else results_enh[col]

            im = ax.imshow(data_to_plot, cmap='gray', aspect='auto',
                           vmin=-vmax_val, vmax=vmax_val)

            ax.text(0.03, 0.96, labels[row][col], transform=ax.transAxes,
                    fontsize=title_size, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))

            if row == 1:
                ax.set_xlabel("Trace Number", fontsize=label_size, fontweight='normal')
                ax.tick_params(axis='x', which='major', labelsize=tick_size)
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if col == 0:
                ax.set_ylabel("Time Sample", fontsize=label_size, fontweight='normal')
                ax.tick_params(axis='y', which='major', labelsize=tick_size)

        cbar = fig.colorbar(im, ax=axs[row, :], shrink=0.85, pad=0.015)
        cbar.ax.tick_params(labelsize=tick_size-2)
        cbar.set_label("Normalized Amp.", fontsize=label_size-2, fontweight='normal')

    fig.text(0.06, 0.72, 'Original\nData', va='center', ha='center', rotation='vertical',
             fontsize=title_size, fontweight='bold')
    fig.text(0.06, 0.30, 'PIFC-Diff\nEnhanced', va='center', ha='center', rotation='vertical',
             fontsize=title_size, fontweight='bold')

    output_name = "Field_Validation_2x3_Grid_ensemble"
    plt.savefig(f"{output_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_name}.pdf", bbox_inches='tight')
    print(f"🎉 Done! Saved as {output_name}.png / .pdf")


if __name__ == "__main__":
    main()
