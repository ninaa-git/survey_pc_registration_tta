import os
from PIL import Image, ImageChops

folder = "/home/nbodelot/projects/def-egranger/nbodelot/registration/pc-registration/silico/PARENet/output_evaluate/my3DMatch_TTA/viz"

for fname in os.listdir(folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(folder, fname)
        img  = Image.open(path).convert('RGB')  # force RGB
        bg   = Image.new('RGB', img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            img.crop(bbox).save(path)
            print(f"Cropped: {fname}")
        else:
            print(f"Skipped (blank?): {fname}")