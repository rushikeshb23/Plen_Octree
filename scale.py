from PIL import Image
import glob

for p in glob.glob("./data/nerf_synthetic/lego/val/*.png"):
    img = Image.open(p).resize((800,800), Image.LANCZOS)
    img.save(p)

# Repeat for val and test
