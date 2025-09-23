# src/query.py
import csv, argparse
from pathlib import Path
from PIL import Image
import numpy as np

INDEX_DIR = Path("./data/index")
REPORT_DIR = Path("./report"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LBP ----------
def extract_lbp(img: Image.Image):
    from skimage.feature import local_binary_pattern
    gray = img.convert("L")
    arr = np.array(gray)
    P, R = 8, 1
    lbp = local_binary_pattern(arr, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    v = hist.astype("float32")
    v /= (np.linalg.norm(v) + 1e-10)
    return v

# ---------- FaceNet ----------
_facenet = None
def get_facenet(device="cpu"):
    global _facenet
    if _facenet is None:
        from facenet_pytorch import InceptionResnetV1
        import torch
        _facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _facenet

def extract_facenet(img: Image.Image, device="cpu"):
    import torchvision.transforms as T
    model = get_facenet(device)
    tfm = T.Compose([
        T.Resize((160,160)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    x = tfm(img.convert("RGB")).unsqueeze(0).to(next(model.parameters()).device)
    with np.errstate(all='ignore'):
        import torch
        with torch.no_grad():
            emb = model(x)            # (1,512)
    v = emb.cpu().numpy().reshape(-1).astype("float32")
    v /= (np.linalg.norm(v) + 1e-10)
    return v

# ---------- utils ----------
def load_index(method):
    feats = np.load(INDEX_DIR / f"features_{method}.npy").astype("float32")
    ids, paths = [], []
    with open(INDEX_DIR / f"ids_{method}.csv", "r", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            ids.append(row["image_id"]); paths.append(row["path"])
    return feats, ids, paths

def cosine_topk(q, M, k=3):
    q = q / (np.linalg.norm(q) + 1e-10)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
    sims = M @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def save_grid(query_path, hit_paths, out_path):
    from PIL import Image, ImageDraw, ImageFont
    SZ = 224
    q = Image.open(query_path).convert("RGB").resize((SZ,SZ))
    imgs = [Image.open(p).convert("RGB").resize((SZ,SZ)) for p in hit_paths]
    W, H = SZ*4, SZ
    canvas = Image.new("RGB", (W, H), (255,255,255))
    canvas.paste(q, (0,0))
    for i, im in enumerate(imgs, 1):
        canvas.paste(im, (i*SZ, 0))
    canvas.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="đường dẫn ảnh query (jpg/png)")
    ap.add_argument("--method", choices=["lbp","facenet"], default="facenet")
    ap.add_argument("--device", default="cpu", help="cpu/cuda (cho facenet)")
    ap.add_argument("--save-grid", action="store_true", help="lưu 1 ảnh ghép query + top3")
    args = ap.parse_args()

    feats, ids, paths = load_index(args.method)
    img = Image.open(args.image).convert("RGB")

    if args.method == "lbp":
        q = extract_lbp(img)
    else:
        q = extract_facenet(img, device=args.device)

    top_idx, top_sim = cosine_topk(q, feats, k=3)

    print("\nTop-3 ảnh giống nhất:")
    for rank, (i, s) in enumerate(zip(top_idx, top_sim), 1):
        print(f"{rank}. {ids[i]}  |  cos={s:.4f}  |  path={paths[i]}")

    if args.save_grid:
        out = REPORT_DIR / f"top3_{Path(args.image).stem}_{args.method}.jpg"
        save_grid(args.image, [paths[i] for i in top_idx], out)
        print(f"\nĐã lưu ảnh ghép: {out}")

if __name__ == "__main__":
    main()
