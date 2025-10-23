# src/extract_features.py
import os, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# ====== đường dẫn ======
DATA_ROOT = Path(os.getenv("DATA_DIR", "./data"))
CLEAN_DIR = Path(os.getenv("ALIGNED_DIR", "./data/aligned"))  
CLEAN_DIR = Path("./data/aligned_clean")  # dùng bộ đã làm sạch
INDEX_DIR = DATA_ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ====== LBP ======
def extract_lbp_features(img: Image.Image, P=8, R=1):
    # dùng skimage cho gọn
    from skimage.feature import local_binary_pattern
    import numpy as np

    gray = img.convert("L")
    arr = np.array(gray)
    lbp = local_binary_pattern(arr, P=P, R=R, method="uniform")
    # số bins = P + 2 theo 'uniform'
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype("float32")

# ====== FaceNet ======
_facenet_model = None
def load_facenet(device="cpu"):
    from facenet_pytorch import InceptionResnetV1
    global _facenet_model
    if _facenet_model is None:
        _facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _facenet_model

def extract_facenet_features(img: Image.Image, device="cpu"):
    import torch
    import torchvision.transforms as T
    model = load_facenet(device)

    # InceptionResnetV1 kỳ vọng ảnh 160x160, chuẩn hóa [-1,1]
    tfm = T.Compose([
        T.Resize((160,160)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)  # (1,512)
    v = emb.cpu().numpy().reshape(-1).astype("float32")
    # chuẩn hóa vector (cosine-friendly)
    v = v / (np.linalg.norm(v) + 1e-10)
    return v

def extract_all(method="lbp", batch_device="cpu"):
    paths = [p for p in CLEAN_DIR.glob("*.jpg*")]
    paths.sort()
    feats = []
    print(f"Tìm thấy {len(paths)} ảnh trong {CLEAN_DIR}")

    for p in tqdm(paths, desc=f"Extracting [{method}]"):
        try:
            img = Image.open(p).convert("RGB")
            if method == "lbp":
                v = extract_lbp_features(img, P=8, R=1)
            elif method == "facenet":
                v = extract_facenet_features(img, device=batch_device)
            else:
                raise ValueError("method phải là 'lbp' hoặc 'facenet'")
            feats.append(v)
        except Exception as e:
            print("Lỗi ảnh:", p.name, e)

    feats = np.vstack(feats).astype("float32")
    ids_path = INDEX_DIR / f"ids_{method}.csv"
    npy_path = INDEX_DIR / f"features_{method}.npy"
    meta_path = INDEX_DIR / f"meta_{method}.json"

    # lưu
    np.save(npy_path, feats)
    with ids_path.open("w", encoding="utf-8") as fp:
        fp.write("image_id,path\n")
        for p in paths:
            fp.write(f"{p.name},{p.as_posix()}\n")

    meta = {
        "method": method,
        "clean_dir": str(CLEAN_DIR),
        "feature_shape": list(feats.shape)
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f" Saved features: {npy_path}")
    print(f" Saved ids:      {ids_path}")
    print(f" Saved meta:     {meta_path}")

if __name__ == "__main__":
    import argparse, torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lbp", "facenet"], default="lbp")
    parser.add_argument("--device", default="cpu")  # "cuda" nếu có GPU
    args = parser.parse_args()
    # nếu chọn facenet và có GPU:
    if args.method == "facenet" and args.device == "cuda" and not torch.cuda.is_available():
        print(" Không thấy CUDA, chuyển sang CPU.")
        args.device = "cpu"
    extract_all(method=args.method, batch_device=args.device)
