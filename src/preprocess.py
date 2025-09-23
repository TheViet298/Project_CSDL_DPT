import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Load config từ .env
load_dotenv()
SELECTED_DIR = Path(os.getenv("SELECTED_DIR", "./data/selected/elderly"))
ALIGNED_DIR = Path(os.getenv("ALIGNED_DIR", "./data/aligned"))
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (224, 224)  # resize về 224x224

def preprocess_images():
    files = list(SELECTED_DIR.glob("*.jpg*"))
    print(f"Tổng số ảnh cần xử lý: {len(files)}")

    for f in tqdm(files, desc="Resizing"):
        try:
            img = Image.open(f).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)  # LANCZOS = chất lượng cao
            out_path = ALIGNED_DIR / f.name
            img.save(out_path)
        except Exception as e:
            print(f"Lỗi {f}: {e}")

if __name__ == "__main__":
    preprocess_images()
    print(f"✔ Đã resize toàn bộ ảnh sang {TARGET_SIZE[0]}x{TARGET_SIZE[1]} và lưu ở {ALIGNED_DIR}")

