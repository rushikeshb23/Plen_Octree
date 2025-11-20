import imageio
import os
from glob import glob

# Always compute absolute path to renders folder
renders_dir = os.path.abspath("lego_exp/renders")
print("Rendering from:", renders_dir)

frames = sorted(glob(os.path.join(renders_dir, "render_*.png")))

if len(frames) == 0:
    print("ERROR: No render_*.png frames found in:", renders_dir)
    print("Check if training actually saved renders.")
    exit()

out_path = os.path.abspath(os.path.join(renders_dir, "video.mp4"))
print("Saving video to:", out_path)

writer = imageio.get_writer(out_path, fps=24)

for f in frames:
    img = imageio.imread(f)
    writer.append_data(img)

writer.close()

print("\nDONE!")
print("Your video is saved at:")
print(out_path)
