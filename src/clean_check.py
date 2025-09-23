# src/clean_check.py
import os, csv, hashlib
from pathlib import Path
from PIL import Image, ImageStat
import cv2
from tqdm import tqdm

ALIGNED_DIR   = Path("./data/aligned")         # input 224x224
CLEAN_DIR     = Path("./data/aligned_clean")   # output ảnh sạch
MANIFEST_DIR  = Path("./data/manifests")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (224, 224)

# ===== THAM SỐ Balanced profile =====
BLUR_THRES = 95.0              # <95 => mờ
USE_DEDUP  = True              # khử trùng lặp pHash
REJECT_WATERMARK = True

# watermark ở GÓC
CORNER_FRAC = 0.15
BRIGHT_THRESH = 230
DARK_THRESH   = 25
EDGE_DENSITY_CORNER = 0.12
BRIGHT_DARK_RATIO   = 0.25

# watermark PHỦ / TRUNG TÂM
EDGE_DENSITY_FULL   = 0.20
EDGE_DENSITY_CENTER = 0.14

# MSER text-like
USE_MSER = True
MSER_DELTA = 5
MSER_MIN_AREA = 30
MSER_MAX_AREA = 2000
MSER_MAX_COMPONENTS = 120

def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png")
    return [p for p in root.rglob("*")
            if p.is_file() and (p.suffix.lower() in exts or p.name.lower().endswith(".jpg.chip.jpg"))]

def is_image_ok(p: Path) -> bool:
    try:
        img = Image.open(p); img.verify()
        img = Image.open(p).convert("RGB")
        return img.size == TARGET_SIZE
    except Exception:
        return False

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean() / 255.0

def has_watermark_goc(img_bgr):
    h, w = img_bgr.shape[:2]
    ch, cw = int(h * CORNER_FRAC), int(w * CORNER_FRAC)
    corners = [
        img_bgr[0:ch, 0:cw], img_bgr[0:ch, w-cw:w],
        img_bgr[h-ch:h, 0:cw], img_bgr[h-ch:h, w-cw:w]
    ]
    for patch in corners:
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        bright = (g >= BRIGHT_THRESH).mean()
        dark   = (g <= DARK_THRESH).mean()
        ed = edge_density(g)
        if (bright >= BRIGHT_DARK_RATIO or dark >= BRIGHT_DARK_RATIO) and (ed >= EDGE_DENSITY_CORNER):
            return True
    return False

def has_watermark_center(gray):
    h, w = gray.shape
    if edge_density(gray) >= EDGE_DENSITY_FULL:
        return True
    cy, cx = int(h*0.25), int(w*0.25)
    center = gray[cy:h-cy, cx:w-cx]
    if edge_density(center) >= EDGE_DENSITY_CENTER:
        return True
    return False

def has_too_many_mser(gray):
    mser = cv2.MSER_create(MSER_DELTA, MSER_MIN_AREA, MSER_MAX_AREA)
    regions, _ = mser.detectRegions(gray)
    cnt = 0
    for pts in regions:
        x, y, w0, h0 = cv2.boundingRect(pts)
        if w0*h0 < MSER_MIN_AREA or w0*h0 > MSER_MAX_AREA:
            continue
        ar = w0 / max(1, h0)
        if 0.3 <= ar <= 10:    # ngưỡng hợp lý cho chữ
            cnt += 1
    return cnt > MSER_MAX_COMPONENTS

def phash(p: Path) -> str:
    img = Image.open(p).convert("L").resize((32, 32))
    avg = ImageStat.Stat(img).mean[0]
    bits = "".join('1' if px > avg else '0' for px in img.getdata())
    return hashlib.sha1(bits.encode()).hexdigest()[:16]

def main():
    files = list_images(ALIGNED_DIR)
    print(f"Tổng file kiểm tra: {len(files)}")
    print(f"- BLUR={BLUR_THRES} | WATERMARK Balanced | DEDUP={USE_DEDUP}")

    kept, dropped = 0, 0
    rows_clean, rows_reject, preview = [], [], []
    seen = set()

    for p in tqdm(files):
        reason = None
        img = cv2.imread(str(p))
        if img is None:
            reason = "load_fail"
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not is_image_ok(p):
                reason = "invalid_or_size"
            elif variance_of_laplacian(gray) < BLUR_THRES:
                reason = "blur"
            else:
                wm_corner = has_watermark_goc(img)
                wm_center = has_watermark_center(gray)
                mser_bad  = USE_MSER and has_too_many_mser(gray)
                if REJECT_WATERMARK and (wm_corner or (wm_center and mser_bad)):
                    reason = "watermark"
                elif USE_DEDUP:
                    h = phash(p)
                    if h in seen:
                        reason = "duplicate"
                    else:
                        seen.add(h)

        if reason:
            dropped += 1
            rows_reject.append({"image_id": p.name, "path": str(p), "reason": reason})
            if len(preview) < 10:
                preview.append((reason, p.name))
            continue

        out = CLEAN_DIR / p.name
        Image.open(p).convert("RGB").save(out)
        rows_clean.append({"image_id": p.name, "path": str(out), "status": "clean"})
        kept += 1

    # Xuất manifest
    clean_csv  = MANIFEST_DIR / "elderly_clean_manifest.csv"
    reject_csv = MANIFEST_DIR / "elderly_reject_manifest.csv"
    with clean_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_id","path","status"])
        w.writeheader(); w.writerows(rows_clean)
    with reject_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_id","path","reason"])
        w.writeheader(); w.writerows(rows_reject)

    print(f"✔ Giữ lại: {kept} | ✖ Loại: {dropped}")
    print(f"Ảnh sạch: {CLEAN_DIR}")
    print(f"Manifest sạch:  {clean_csv}")
    print(f"Manifest loại:  {reject_csv}")
    if preview:
        print("Ví dụ ảnh bị loại (tối đa 10):")
        for r, n in preview:
            print(f"  - {r}: {n}")

if __name__ == "__main__":
    main()
