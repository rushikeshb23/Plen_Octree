# render_only.py
import os, argparse
import train_nerf
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True)
parser.add_argument("--expname", required=True)
parser.add_argument("--ckpt", default="final.pth")
parser.add_argument("--outdir", default=None)
args = parser.parse_args()

final_ckpt = os.path.join(args.expname, args.ckpt)
if not os.path.exists(final_ckpt):
    raise SystemExit("Checkpoint not found: " + final_ckpt)

# build args namespace for render_test
a = Namespace(
    dataset_path=args.dataset_path,
    expname=args.expname,
    N_pos_freqs=10,
    N_dir_freqs=4,
    D=8, W=256,
    N_rand=1024, N_coarse=64, N_fine=128,
    near=2.0, far=6.0,
    N_pos=10, N_dir=4,
    Nc=64, Nf=128
)

out = args.outdir or os.path.join(args.expname, "renders")
os.makedirs(out, exist_ok=True)
train_nerf.render_test(a, final_ckpt, split="test", out_dir=out)
print("Rendered frames to", out)
