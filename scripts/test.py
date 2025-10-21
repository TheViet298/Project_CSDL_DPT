from pathlib import Path
import re, cv2
from PIL import Image
import numpy as np
from collections import Counter

DATA_DIR = Path("data/non_elderly")  
age_re = re.compile(r"^(\d+)_")

def parse_age(name):
    m = age_re.match(name)
    return int(m.group(1)) if m else None

def lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_false_colormap(arr, thresh=0.25):
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    r,g,b = arr[...,0],arr[...,1],arr[...,2]
    bad = (r < 25) & ((g > 140) | (b > 140))
    return bad.mean() > thresh

ages = []
blurry = heat = total = 0

for p in DATA_DIR.iterdir():
    if p.suffix.lower().endswith("jpg"):
        total += 1
        age = parse_age(p.name)
        if age: ages.append(age)
        img = np.array(Image.open(p).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if lap_var(gray) < 75:
            blurry += 1
        if is_false_colormap(img):
            heat += 1

print("Tổng ảnh:", total)
print("Mờ (blur):", blurry)
print("Giả màu (heatmap):", heat)
print("Phân bố tuổi:", Counter(ages))
