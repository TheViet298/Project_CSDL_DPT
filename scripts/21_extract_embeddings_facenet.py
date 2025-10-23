# scripts/21_extract_embeddings_facenet.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings_facenet import FaceNetEmbedder, load_image_any

import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

# === đường dẫn ===
PROJECT_ROOT = Path(".")
MANI_DIR = PROJECT_ROOT / "data/manifests"
INDEX_DIR = PROJECT_ROOT / "data/index"; INDEX_DIR.mkdir(parents=True, exist_ok=True)

def run_one(split: str, max_rows: int | None = None):
    csv_in = MANI_DIR / f"{split}.csv"
    df = pd.read_csv(csv_in)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    fb = FaceNetEmbedder()
    feats, kept = [], []

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split}"):
        try:
            img = load_image_any(r["path"], project_root=PROJECT_ROOT)
            vec = fb.embed_pil(img)     # (512,)
            feats.append(vec)
            kept.append(r)
        except Exception as e:
            print("skip:", r.get("path"), e)

    X = np.stack(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)
    np.save(INDEX_DIR / f"{split}_feats_facenet.npy", X)
    pd.DataFrame(kept).to_csv(INDEX_DIR / f"{split}_meta.csv", index=False)
    print(f"[{split}] saved feats shape:", X.shape)

if __name__ == "__main__":
    # max_rows=None để lấy hết; đặt 2000 nếu muốn chạy nhanh lần đầu
    run_one("train", max_rows=None)
    run_one("val",   max_rows=None)
    run_one("test",  max_rows=None)
