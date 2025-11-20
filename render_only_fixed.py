#!/usr/bin/env python3
"""
Standalone renderer for your trained NeRF.
Saves renders to: <expname>/renders/render_XXXX.png

Usage:
    python render_only_fixed.py --dataset_path ./data/nerf_synthetic/lego --expname lego_exp --ckpt final.pth
"""
import os
import json
from glob import glob
from argparse import ArgumentParser

import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------
# Small helper loader (robust)
# ---------------------------
def load_blender_data(basedir):
    def load_json(p):
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
        return None

    meta = load_json(os.path.join(basedir, "transforms.json"))
    frames = []
    camera_angle_x = None
    H = W = focal = None

    if meta is not None:
        frames = meta.get("frames", [])
        camera_angle_x = meta.get("camera_angle_x", None)
        H = meta.get("h", None)
        W = meta.get("w", None)
        focal = meta.get("fl_x", None)
    else:
        split_files = [("transforms_train.json", "train"),
                       ("transforms_val.json", "val"),
                       ("transforms_test.json", "test")]
        for fname, folder in split_files:
            m = load_json(os.path.join(basedir, fname))
            if m is None:
                continue
            if camera_angle_x is None:
                camera_angle_x = m.get("camera_angle_x", None)
            if H is None and m.get("h") is not None:
                H = m.get("h")
            if W is None and m.get("w") is not None:
                W = m.get("w")
            if focal is None and m.get("fl_x") is not None:
                focal = m.get("fl_x")
            for fr in m.get("frames", []):
                base = os.path.basename(fr["file_path"])
                base_no_ext = os.path.splitext(base)[0]
                frames.append({**fr, "file_path": os.path.join(folder, base_no_ext)})

    if len(frames) == 0:
        raise RuntimeError("No transforms found in " + basedir)

    frames_dict = {fr["file_path"]: fr for fr in frames}
    all_frame_keys = list(frames_dict.keys())

    imgs = []
    poses = []
    i_split = []
    idx0 = 0

    # helper to infer hw from files
    def infer_hw():
        for s in ["train","val","test"]:
            d = os.path.join(basedir, s)
            if not os.path.exists(d):
                continue
            files = sorted(glob(d + "/*.png"))
            files = [f for f in files if "_depth" not in f]
            if len(files) > 0:
                im = imageio.imread(files[0])
                return im.shape[0], im.shape[1]
        return None, None

    for s in ["train","val","test"]:
        d = os.path.join(basedir, s)
        if not os.path.exists(d):
            i_split.append([])
            continue
        files = sorted(glob(d + "/*.png"))
        files = [f for f in files if "_depth" not in f]
        i_split.append(list(range(idx0, idx0 + len(files))))
        idx0 += len(files)
        for p in files:
            name = os.path.splitext(os.path.basename(p))[0]
            # try direct keys
            candidates = [os.path.join(s, name), name]
            found = None
            for c in candidates:
                if c in frames_dict:
                    found = c
                    break
            if found is None:
                for k in all_frame_keys:
                    if name.lower() in os.path.basename(k).lower():
                        found = k
                        break
            if found is None:
                raise RuntimeError(f"No frame entry for image {p}. Available keys sample: {all_frame_keys[:6]}")
            fr = frames_dict[found]
            imgs.append(imageio.imread(p))
            poses.append(np.array(fr["transform_matrix"], dtype=np.float32))

    imgs = np.array(imgs).astype(np.uint8)
    poses = np.array(poses).astype(np.float32)

    if H is None or W is None:
        h_w = infer_hw()
        if h_w[0] is None:
            H = imgs.shape[1]; W = imgs.shape[2]
        else:
            H, W = h_w

    if focal is None:
        if camera_angle_x is not None:
            focal = .5 * W / np.tan(.5 * float(camera_angle_x))
        else:
            focal = .5 * W / np.tan(.5 * 0.6911112070083618)

    return imgs, poses, [int(H), int(W), float(focal)], i_split

# ---------------------------
# Small NeRF utilities (match training)
# ---------------------------
def get_rays(H, W, focal, c2w):
    if not torch.is_tensor(c2w):
        c2w = torch.from_numpy(c2w).float()
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - W*0.5)/focal, -(j - H*0.5)/focal, -torch.ones_like(i)], -1).permute(1,0,2)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

class PositionalEncoding(nn.Module):
    def __init__(self, in_dims, num_freqs=10):
        super().__init__()
        self.in_dims = in_dims
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs-1, num_freqs)
        self.out_dim = in_dims * (2 * num_freqs) + in_dims
    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_dir=3, skips=[4], num_rgb_channels=3):
        super().__init__()
        self.D = D; self.W = W; self.skips = skips
        layers = []
        in_ch = input_ch
        for i in range(D):
            layers.append(nn.Linear(in_ch, W))
            layers.append(nn.ReLU(True))
            in_ch = W
            if i in skips:
                in_ch += input_ch
        self.pos_mlp = nn.Sequential(*layers)
        self.sigma_layer = nn.Linear(W, 1)
        self.feature_layer = nn.Linear(W, W)
        self.dir_layer_1 = nn.Linear(W + input_ch_dir, W//2)
        self.dir_layer_2 = nn.Linear(W//2, num_rgb_channels)
    def forward(self, x, d):
        h = x; inVec = x
        for i in range(self.D):
            lin = self.pos_mlp[2*i]; relu = self.pos_mlp[2*i+1]
            h = lin(h); h = relu(h)
            if i in self.skips:
                h = torch.cat([h, inVec], -1)
        sigma = self.sigma_layer(h)
        feat = self.feature_layer(h)
        h_dir = torch.cat([feat, d], -1)
        h_dir = F.relu(self.dir_layer_1(h_dir))
        rgb = torch.sigmoid(self.dir_layer_2(h_dir))
        return rgb, F.relu(sigma)

def volume_render_radiance_field(rgb, sigma, z_vals, rays_d):
    deltas = z_vals[...,1:] - z_vals[...,:-1]
    delta_last = 1e10*torch.ones_like(deltas[...,:1])
    deltas = torch.cat([deltas, delta_last], -1)
    d_norm = torch.norm(rays_d[..., None, :], dim=-1)
    deltas = deltas * d_norm
    alpha = 1.0 - torch.exp(-sigma[...,0] * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * trans
    comp_rgb = torch.sum(weights.unsqueeze(-1) * rgb, -2)
    return comp_rgb, weights

# ---------------------------
# Rendering function
# ---------------------------
def render_from_ckpt(dataset_path, expname, ckpt_path, split="test", out_dir=None, device=None, N_coarse=64, N_fine=128, near=2.0, far=6.0, N_pos_freqs=10, N_dir_freqs=4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Rendering on device:", device)
    imgs, poses, hwf, i_split = load_blender_data(dataset_path)
    H, W, focal = hwf
    pos_enc = PositionalEncoding(3, num_freqs=N_pos_freqs).to(device)
    dir_enc = PositionalEncoding(3, num_freqs=N_dir_freqs).to(device)
    input_ch = pos_enc.out_dim
    input_dir_ch = dir_enc.out_dim
    nerf_coarse = NeRF(input_ch=input_ch, input_ch_dir=input_dir_ch).to(device)
    nerf_fine = NeRF(input_ch=input_ch, input_ch_dir=input_dir_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # handle common key names
    if "coarse" in ckpt and "fine" in ckpt:
        nerf_coarse.load_state_dict(ckpt["coarse"])
        nerf_fine.load_state_dict(ckpt["fine"])
    elif "nerf_coarse" in ckpt and "nerf_fine" in ckpt:
        nerf_coarse.load_state_dict(ckpt["nerf_coarse"])
        nerf_fine.load_state_dict(ckpt["nerf_fine"])
    else:
        # try to load entire checkpoint keys if names differ
        for k in ckpt.keys():
            if "coarse" in k and isinstance(ckpt[k], dict):
                nerf_coarse.load_state_dict(ckpt[k]); break
        for k in ckpt.keys():
            if "fine" in k and isinstance(ckpt[k], dict):
                nerf_fine.load_state_dict(ckpt[k]); break
    nerf_coarse.eval(); nerf_fine.eval()
    split_idx = i_split[2] if split=="test" else (i_split[1] if split=="val" else i_split[0])
    out_dir = out_dir or os.path.join(expname, "renders")
    os.makedirs(out_dir, exist_ok=True)
    near = near; far = far
    for idx in split_idx:
        c2w = torch.from_numpy(poses[idx]).float().to(device)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1,3).to(device)
        rays_d = rays_d.reshape(-1,3).to(device)
        # coarse
        z_vals = torch.linspace(near, far, N_coarse, device=device).unsqueeze(0).expand(rays_o.shape[0], N_coarse)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        pts_flat = pts.reshape(-1, 3)
        dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        dirs_flat = dirs.unsqueeze(1).expand(-1, N_coarse, -1).reshape(-1, 3)
        with torch.no_grad():
            pts_enc = pos_enc(pts_flat)
            dirs_enc = dir_enc(dirs_flat)
            rgb_c, sigma_c = nerf_coarse(pts_enc, dirs_enc)
            rgb_c = rgb_c.reshape(-1, N_coarse, 3)
            sigma_c = sigma_c.reshape(-1, N_coarse, 1)
            comp_rgb_c, weights = volume_render_radiance_field(rgb_c, sigma_c, z_vals, rays_d)
            if N_fine > 0:
                z_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                samples = None
                # sample_pdf implementation (deterministic sampling for render)
                weights_pdf = weights[...,1:-1] if weights.shape[1] > 2 else weights[..., :1]
                weights_pdf = weights_pdf + 1e-5
                pdf = weights_pdf / torch.sum(weights_pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
                u = torch.linspace(0.0, 1.0, steps=N_fine, device=device).expand(list(cdf.shape[:-1]) + [N_fine])
                inds = torch.searchsorted(cdf, u, right=True)
                below = torch.clamp(inds - 1, min=0)
                above = torch.clamp(inds, max=cdf.shape[-1]-1)
                inds_g = torch.stack([below, above], -1)
                cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*inds_g.shape[:-1], cdf.shape[-1]), -1, inds_g)
                bins_g = torch.gather(z_mid.unsqueeze(-2).expand(*inds_g.shape[:-1], z_mid.shape[-1]), -1, inds_g)
                denom = (cdf_g[...,1] - cdf_g[...,0])
                denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
                t = (u - cdf_g[...,0]) / denom
                samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
                z_vals_combined, _ = torch.sort(torch.cat([z_vals, samples], -1), -1)
                pts_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_combined.unsqueeze(-1)
                pts_fine_flat = pts_fine.reshape(-1, 3)
                dirs_fine_flat = dirs.unsqueeze(1).expand(-1, z_vals_combined.shape[1], -1).reshape(-1, 3)
                pts_fine_enc = pos_enc(pts_fine_flat)
                dirs_fine_enc = dir_enc(dirs_fine_flat)
                rgb_f, sigma_f = nerf_fine(pts_fine_enc, dirs_fine_enc)
                rgb_f = rgb_f.reshape(-1, z_vals_combined.shape[1], 3)
                sigma_f = sigma_f.reshape(-1, z_vals_combined.shape[1], 1)
                comp_rgb_f, _ = volume_render_radiance_field(rgb_f, sigma_f, z_vals_combined, rays_d)
                comp = comp_rgb_f
            else:
                comp = comp_rgb_c
        comp_img = (comp.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(out_dir, f'render_{idx:04d}.png')
        imageio.imwrite(out_path, comp_img)
        print("Saved", out_path)
    print("All renders saved to:", out_dir)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--expname", required=True)
    p.add_argument("--ckpt", default="final.pth")
    p.add_argument("--device", default=None)
    p.add_argument("--N_coarse", type=int, default=64)
    p.add_argument("--N_fine", type=int, default=128)
    p.add_argument("--near", type=float, default=2.0)
    p.add_argument("--far", type=float, default=6.0)
    p.add_argument("--N_pos_freqs", type=int, default=10)
    p.add_argument("--N_dir_freqs", type=int, default=4)
    args = p.parse_args()

    ckpt_path = os.path.join(args.expname, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise SystemExit("Checkpoint not found: " + ckpt_path)

    render_from_ckpt(args.dataset_path, args.expname, ckpt_path, split="test", out_dir=os.path.join(args.expname, "renders"),
                     device=args.device, N_coarse=args.N_coarse, N_fine=args.N_fine, near=args.near, far=args.far,
                     N_pos_freqs=args.N_pos_freqs, N_dir_freqs=args.N_dir_freqs)
