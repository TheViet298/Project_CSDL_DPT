# scripts/01_build_non_elderly_v2.py
from pathlib import Path
import re, random
from collections import defaultdict, deque
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

# ====== Cấu hình ======
RAW_DIR   = Path("data/raw/utkface")     # nơi chứa ảnh UTKFace gốc
ELDER_DIR = Path("data/aligned_clean")   # ảnh elderly (>=60) bạn đã có
OUT_DIR   = Path("data/non_elderly")     # nơi sẽ ghi non-elderly (0..59)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tỉ lệ muốn đạt: elderly : non = 1 : 2
RATIO_NON_TO_ELDERLY = 2.0
MAX_CAP = 20000  # trần an toàn

# Trọng số theo bin (để học ranh giới gần elderly tốt hơn)
# 0-19  -> 0.8x   ; 20-39 -> 1.0x ; 40-59 -> 1.2x
BIN_WEIGHTS = {(0,19):0.8, (20,39):1.0, (40,59):1.2}

SEED = 42
random.seed(SEED)

# Detector/aligner
mtcnn = MTCNN(
    image_size=160, margin=20, keep_all=False, select_largest=True,
    post_process=False, min_face_size=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

age_re = re.compile(r"^(\d+)_")
def parse_age(name: str):
    m = age_re.match(name)
    return int(m.group(1)) if m else None

def iter_candidates(root: Path):
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower()==".jpg":
            yield p

def count_elderly():
    n=0
    if not ELDER_DIR.exists(): return 700  # fallback
    for p in ELDER_DIR.glob("*.jpg"):
        a = parse_age(p.name)
        if a is not None and a >= 60: n+=1
    return max(n,1)

def bin_of(age:int):
    for (lo,hi),w in BIN_WEIGHTS.items():
        if lo <= age <= hi: return (lo,hi)
    return None

if __name__ == "__main__":
    # 1) Quyết định tổng số cần lưu cho non-elderly
    n_elder = count_elderly()
    target_total = int(min(MAX_CAP, RATIO_NON_TO_ELDERLY * n_elder))
    print(f"Elderly count≈{n_elder} → target non-elderly = {target_total}")

    # 2) Gom ảnh 0..59 theo từng tuổi + đếm theo bin
    by_age = defaultdict(list)
    by_bin = defaultdict(list)
    total_scanned = 0
    for p in iter_candidates(RAW_DIR):
        total_scanned += 1
        age = parse_age(p.name)
        if age is None or age > 59: 
            continue
        b = bin_of(age)
        if b is None: 
            continue
        by_age[age].append(p); by_bin[b].append(p)

    # 3) Chuẩn bị quota theo bin-weight
    #    - Phân bổ theo trọng số BIN_WEIGHTS
    #    - Sau đó dàn đều quota của mỗi bin cho từng tuổi trong bin
    weight_sum = sum(BIN_WEIGHTS.values())
    bin_quota = {}
    for b,w in BIN_WEIGHTS.items():
        bin_quota[b] = int(round(target_total * (w/weight_sum)))

    # Bảo đảm tổng đúng target_total
    diff = target_total - sum(bin_quota.values())
    # Đổ phần dư vào bin 40-59 trước, rồi 20-39, rồi 0-19
    pref = [(40,59),(20,39),(0,19)]
    i=0
    while diff!=0:
        b = pref[i%len(pref)]
        bin_quota[b] += 1 if diff>0 else -1
        diff += -1 if diff>0 else 1
        i+=1

    print("Bin quota:", bin_quota)

    # Shuffle để tránh thiên vị
    for a in list(by_age.keys()):
        random.shuffle(by_age[a])

    saved_total, skipped = 0, 0
    saved_per_age = {a:0 for a in range(0,60)}
    idx = {a:0 for a in range(0,60)}

    # Tạo danh sách tuổi theo từng bin (round-robin)
    pbar = tqdm(total=target_total, desc="Build non-elderly 0..59 (weighted bins)")
    for b, q in bin_quota.items():
        lo,hi = b
        ages = list(range(lo,hi+1))
        # Loại bỏ tuổi không có ảnh
        ages = [a for a in ages if len(by_age[a])>0]
        if not ages: continue
        dq = deque(ages)
        while q>0 and dq:
            a = dq.popleft()
            # lấy 1 ảnh từ tuổi a nếu còn
            while idx[a] < len(by_age[a]) and q>0:
                p = by_age[a][idx[a]]; idx[a]+=1
                try:
                    img = Image.open(p).convert("RGB")
                    aligned = mtcnn(img, save_path=str(OUT_DIR / p.name))
                    if aligned is None:
                        skipped += 1; continue
                    saved_per_age[a]+=1; saved_total+=1; q-=1; pbar.update(1)
                    break
                except Exception:
                    skipped += 1
                    break
            # nếu tuổi a còn ảnh và bin q >0 thì vòng sau lại push cuối hàng
            if idx[a] < len(by_age[a]) and q>0:
                dq.append(a)

    pbar.close()

    # Nếu vẫn thiếu vì cạn ảnh → đổ bù từ phần còn lại 0..59
    if saved_total < target_total:
        leftovers = []
        for a in range(0,60):
            leftovers.extend(by_age[a][idx[a]:])
        random.shuffle(leftovers)
        for p in leftovers:
            if saved_total >= target_total: break
            try:
                img = Image.open(p).convert("RGB")
                aligned = mtcnn(img, save_path=str(OUT_DIR / p.name))
                if aligned is None: 
                    skipped += 1; continue
                saved_total+=1
            except Exception:
                skipped += 1

    # Thống kê
    by_decade = {}
    for d_lo in [0,10,20,30,40,50]:
        d_hi = d_lo+9
        by_decade[f"{d_lo}-{d_hi}"] = sum(saved_per_age[a] for a in range(d_lo, min(d_hi,59)+1))
    print("Saved per-decade:", by_decade)
    print(f"Total scanned : {total_scanned}")
    print(f"Saved total   : {saved_total} (target {target_total})")
    print(f"Skipped       : {skipped}")
    print(f"Output dir    : {OUT_DIR.resolve()}")
