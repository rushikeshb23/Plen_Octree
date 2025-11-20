#!/usr/bin/env python3
"""
Minimal NeRF implementation (coarse + fine) FULLY FIXED
Supports your dataset:
   r_0.png
   r_0_depth_0001.png  ← IGNORED
   r_1.png
   r_1_depth_0001.png  ← IGNORED
"""

import os
import json
import math
from glob import glob
from argparse import ArgumentParser

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def img2float32(img):
    return (np.array(img).astype(np.float32) / 255.0)


# ===========================================================
#  LOAD BLENDER DATA (fixed for depth filenames)
# ===========================================================
def load_blender_data(basedir):
    """
    Robust loader that accepts:
      - transforms.json
      - transforms_train.json / transforms_val.json / transforms_test.json
    Handles cases where JSONs are missing 'h', 'w', or 'fl_x' by:
      - reading an actual image to infer H,W
      - computing focal from camera_angle_x if available
    Ignores depth images matching '*_depth*'
    Returns: imgs, poses, [H,W,focal], i_split
    """
    import os, json
    from glob import glob
    import imageio.v2 as imageio
    import numpy as np

    def load_json(p):
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return None

    # try single transforms.json first
    meta = load_json(os.path.join(basedir, "transforms.json"))
    frames = []
    camera_angle_x = None
    H = W = focal = None

    if meta is not None:
        frames = meta.get("frames", [])
        camera_angle_x = meta.get("camera_angle_x", None)
        if meta.get("h") is not None:
            try:
                H = int(meta.get("h"))
            except Exception:
                H = None
        if meta.get("w") is not None:
            try:
                W = int(meta.get("w"))
            except Exception:
                W = None
        if meta.get("fl_x") is not None:
            try:
                focal = float(meta.get("fl_x"))
            except Exception:
                focal = None
    else:
        # try split jsons
        split_files = [("transforms_train.json", "train"),
                       ("transforms_val.json", "val"),
                       ("transforms_test.json", "test")]
        for fname, folder in split_files:
            m = load_json(os.path.join(basedir, fname))
            if m is None:
                continue
            # adopt camera params only if not set already
            if camera_angle_x is None and m.get("camera_angle_x") is not None:
                camera_angle_x = m.get("camera_angle_x")
            if H is None and m.get("h") is not None:
                try:
                    H = int(m.get("h"))
                except Exception:
                    H = None
            if W is None and m.get("w") is not None:
                try:
                    W = int(m.get("w"))
                except Exception:
                    W = None
            if focal is None and m.get("fl_x") is not None:
                try:
                    focal = float(m.get("fl_x"))
                except Exception:
                    focal = None
            for fr in m.get("frames", []):
                orig_fp = fr.get("file_path", "")
                base = os.path.basename(orig_fp)
                base_no_ext = os.path.splitext(base)[0]
                frames.append({**fr, "file_path": os.path.join(folder, base_no_ext)})

    if len(frames) == 0:
        raise RuntimeError(f"No transforms.json or transforms_*.json found in {basedir}")

    # Build quick lookup
    frames_dict = {fr["file_path"]: fr for fr in frames}
    all_frame_keys = list(frames_dict.keys())

    # Collect image files (ignore depth images)
    imgs = []
    poses = []
    i_split = []
    idx0 = 0

    # helper to pick a sample image to infer H,W if needed
    def infer_hw_from_files():
        for split in ["train", "val", "test"]:
            d = os.path.join(basedir, split)
            if not os.path.exists(d):
                continue
            files = sorted(glob(os.path.join(d, "*.png")))
            files = [f for f in files if "_depth" not in f]
            if len(files) > 0:
                im = imageio.imread(files[0])
                return im.shape[0], im.shape[1]
        return None, None

    for s in ["train", "val", "test"]:
        d = os.path.join(basedir, s)
        if not os.path.exists(d):
            i_split.append([])
            continue
        files = sorted(glob(os.path.join(d, "*.png")))
        # filter out depth maps
        files = [f for f in files if "_depth" not in f]
        i_split.append(list(range(idx0, idx0 + len(files))))
        idx0 += len(files)
        for p in files:
            name = os.path.splitext(os.path.basename(p))[0]  # e.g., r_0
            # try direct normalized keys
            candidates = [
                os.path.join(s, name),
                os.path.join(s, name + ".png"),
                "./" + os.path.join(s, name),
                "./" + os.path.join(s, name + ".png"),
                name,
                name + ".png"
            ]
            found_key = None
            for c in candidates:
                if c in frames_dict:
                    found_key = c
                    break
            if found_key is None:
                # substring match on basename
                for k in all_frame_keys:
                    if name.lower() in os.path.basename(k).lower() or os.path.basename(k).lower() in name.lower():
                        found_key = k
                        break
            if found_key is None:
                # final fallback: match prefix before underscore
                pref = name.split("_")[0]
                for k in all_frame_keys:
                    if pref.lower() in os.path.basename(k).lower():
                        found_key = k
                        break
            if found_key is None:
                sample_keys = all_frame_keys[:20]
                raise RuntimeError(f"Couldn't find a matching frame for image '{p}'. Tried candidates {candidates}. Sample keys: {sample_keys}")
            fr = frames_dict[found_key]
            pose = np.array(fr["transform_matrix"], dtype=np.float32)
            imgs.append(imageio.imread(p))
            poses.append(pose)

    imgs = np.array(imgs).astype(np.uint8)
    poses = np.array(poses).astype(np.float32)

    # If H/W not known from JSONs, infer from actual image files
    if H is None or W is None:
        h_w = infer_hw_from_files()
        if h_w[0] is None:
            # last resort: infer from loaded imgs numpy
            if imgs.size == 0:
                raise RuntimeError("No images found to infer H/W.")
            H = imgs.shape[1]
            W = imgs.shape[2]
        else:
            H, W = h_w

    # Compute focal if missing: prefer fl_x, else compute from camera_angle_x, else fallback
    if focal is None:
        if camera_angle_x is not None:
            focal = 0.5 * W / np.tan(0.5 * float(camera_angle_x))
        else:
            # fallback to a reasonable default focal
            focal = 0.5 * W / np.tan(0.5 * 0.6911112070083618)

    hwf = [int(H), int(W), float(focal)]
    return imgs, poses, hwf, i_split


# ===========================================================
#  RAYS (GPU SAFE)
# ===========================================================
def get_rays(H, W, focal, c2w):

    if not torch.is_tensor(c2w):
        c2w = torch.from_numpy(c2w).float()

    device = c2w.device

    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], -1).permute(1, 0, 2)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


# ===========================================================
#  POSITIONAL ENCODING
# ===========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, in_dims, num_freqs=10):
        super().__init__()
        self.freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.out_dim = in_dims * (1 + 2 * num_freqs)

    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out.append(torch.sin(x * f))
            out.append(torch.cos(x * f))
        return torch.cat(out, -1)


# ===========================================================
#  NeRF MLP
# ===========================================================
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_dir=3):
        super().__init__()

        layers = []
        in_ch = input_ch
        skip = 4

        for i in range(D):
            layers.append(nn.Linear(in_ch, W))
            layers.append(nn.ReLU(True))
            in_ch = W
            if i == skip:
                in_ch += input_ch

        self.mlp = nn.Sequential(*layers)

        self.sigma_fc = nn.Linear(W, 1)
        self.feature_fc = nn.Linear(W, W)

        self.dir_fc1 = nn.Linear(W + input_ch_dir, W // 2)
        self.dir_fc2 = nn.Linear(W // 2, 3)

    def forward(self, x, d):
        h = x
        x_in = x

        for layer_idx in range(0, len(self.mlp), 2):
            h = self.mlp[layer_idx](h)
            h = self.mlp[layer_idx + 1](h)
            if layer_idx // 2 == 4:
                h = torch.cat([h, x_in], -1)

        sigma = F.relu(self.sigma_fc(h))
        feat = self.feature_fc(h)

        h_dir = F.relu(self.dir_fc1(torch.cat([feat, d], -1)))
        rgb = torch.sigmoid(self.dir_fc2(h_dir))

        return rgb, sigma


# ===========================================================
#  VOLUME RENDER
# ===========================================================
def volume_render(rgb, sigma, z_vals, rays_d):

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], -1)

    d_norm = torch.norm(rays_d[:, None, :], dim=-1)
    deltas *= d_norm

    alpha = 1 - torch.exp(-sigma[..., 0] * deltas)

    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1),
        -1
    )[:, :-1]

    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, -2)

    return rgb_map, weights


# ===========================================================
#  DATASET
# ===========================================================
class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, basedir):
        imgs, poses, hwf, i_split = load_blender_data(basedir)
        self.H, self.W, self.focal = hwf

        self.poses = poses
        self.imgs = imgs
        self.idx = i_split[0]  # only train

        all_imgs = []

        for i in self.idx:
            im = img2float32(imgs[i])
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[..., :3]
            if im.ndim == 2:
                im = np.stack([im, im, im], -1)
            all_imgs.append(im)

        self.all_rgbs = np.stack(all_imgs).reshape(-1, 3)

        rays_o, rays_d = [], []

        for i in self.idx:
            c2w = torch.from_numpy(poses[i]).float()
            r_o, r_d = get_rays(self.H, self.W, self.focal, c2w)
            rays_o.append(r_o.numpy().reshape(-1, 3))
            rays_d.append(r_d.numpy().reshape(-1, 3))

        self.all_rays_o = np.concatenate(rays_o, axis=0)
        self.all_rays_d = np.concatenate(rays_d, axis=0)

        print("Loaded dataset:", self.all_rays_o.shape, self.all_rgbs.shape)

    def __len__(self):
        return len(self.all_rays_o)

    def __getitem__(self, idx):
        return (
            self.all_rays_o[idx],
            self.all_rays_d[idx],
            self.all_rgbs[idx]
        )


# ===========================================================
#  TRAINING
# ===========================================================
def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = NeRFDataset(args.dataset_path)

    H, W, focal = train_ds.H, train_ds.W, train_ds.focal

    pos_enc = PositionalEncoding(3, args.N_pos).to(device)
    dir_enc = PositionalEncoding(3, args.N_dir).to(device)

    nerf_c = NeRF(input_ch=pos_enc.out_dim, input_ch_dir=dir_enc.out_dim).to(device)
    nerf_f = NeRF(input_ch=pos_enc.out_dim, input_ch_dir=dir_enc.out_dim).to(device)

    optimizer = torch.optim.Adam(
        list(nerf_c.parameters()) + list(nerf_f.parameters()),
        lr=args.lr
    )

    rays_o = torch.tensor(train_ds.all_rays_o, dtype=torch.float32).to(device)
    rays_d = torch.tensor(train_ds.all_rays_d, dtype=torch.float32).to(device)
    rgbs = torch.tensor(train_ds.all_rgbs, dtype=torch.float32).to(device)

    N = len(train_ds)

    for it in tqdm(range(args.iters)):

        sel = torch.randint(0, N, (args.N_rand,), device=device)

        ro = rays_o[sel]
        rd = rays_d[sel]
        target = rgbs[sel]

        # Coarse sampling
        z_vals = torch.linspace(args.near, args.far, args.Nc, device=device)
        z_vals = z_vals.unsqueeze(0).repeat(args.N_rand, 1)

        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], 1)
        lower = torch.cat([z_vals[:, :1], mids], 1)
        z_vals = lower + torch.rand_like(z_vals) * (upper - lower)

        pts = ro[:, None, :] + rd[:, None, :] * z_vals[..., None]
        dirs = rd / torch.norm(rd, dim=-1, keepdim=True)

        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs[:, None, :].expand_as(pts).reshape(-1, 3)

        pts_enc = pos_enc(pts_flat)
        dirs_enc = dir_enc(dirs_flat)

        rgb_c, sigma_c = nerf_c(pts_enc, dirs_enc)
        rgb_c = rgb_c.reshape(args.N_rand, args.Nc, 3)
        sigma_c = sigma_c.reshape(args.N_rand, args.Nc, 1)

        comp_c, weights = volume_render(rgb_c, sigma_c, z_vals, rd)

        # Fine sampling
        z_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        weights2 = weights[:, 1:-1] + 1e-5

        pdf = weights2 / torch.sum(weights2, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)

        u = torch.rand(args.N_rand, args.Nf, device=device)
        inds = torch.searchsorted(cdf, u, right=True)

        below = torch.clamp(inds - 1, 0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)

        inds_g = torch.stack([below, above], -1)

        cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, args.Nf, -1), 2, inds_g)
        bins_g = torch.gather(z_mid.unsqueeze(1).expand(-1, args.Nf, -1), 2, inds_g)

        denom = torch.where(
            (cdf_g[..., 1] - cdf_g[..., 0]) < 1e-5,
            torch.ones_like(cdf_g[..., 0]),
            cdf_g[..., 1] - cdf_g[..., 0]
        )

        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_final = torch.sort(torch.cat([z_vals, samples], -1), -1)[0]

        pts = ro[:, None, :] + rd[:, None, :] * z_final[..., None]
        dirs = rd / torch.norm(rd, dim=-1, keepdim=True)

        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs[:, None, :].expand_as(pts).reshape(-1, 3)

        pts_enc = pos_enc(pts_flat)
        dirs_enc = dir_enc(dirs_flat)

        rgb_f, sigma_f = nerf_f(pts_enc, dirs_enc)
        rgb_f = rgb_f.reshape(args.N_rand, z_final.shape[1], 3)
        sigma_f = sigma_f.reshape(args.N_rand, z_final.shape[1], 1)

        comp_f, _ = volume_render(rgb_f, sigma_f, z_final, rd)

        loss = F.mse_loss(comp_c, target) + F.mse_loss(comp_f, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % args.i_print == 0:
            psnr = -10 * torch.log10(F.mse_loss(comp_f, target))
            print(f"Iter {it}  Loss {loss.item():.4f}  PSNR {psnr.item():.2f}")

    print("Training finished.")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--N_pos", type=int, default=10)
    parser.add_argument("--N_dir", type=int, default=4)
    parser.add_argument("--N_rand", type=int, default=1024)
    parser.add_argument("--Nc", type=int, default=64)
    parser.add_argument("--Nf", type=int, default=128)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--i_print", type=int, default=100)

    args = parser.parse_args()
    train(args)
