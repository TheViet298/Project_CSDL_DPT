# scripts/01_build_non_elderly.py
from pathlib import Path
import re, random
from collections import defaultdict, deque
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

RAW_DIR = Path("data/raw/utkface")
OUT_DIR = Path("data/non_elderly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== tổng số ảnh non-elderly muốn lưu (1:2 ~ 700 elderly -> 1400 non) ======
MAX_SAVE = 1400

age_re = re.compile(r"^(\d+)_")
AGES = list(range(20, 60))        # 20..59 (40 tuổi)
random.seed(42)

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    post_process=False,           # đầu ra [0,1] -> màu chuẩn
    device="cuda" if torch.cuda.is_available() else "cpu",
    keep_all=False,
    select_largest=True,
    min_face_size=10
)

def parse_age(name: str):
    m = age_re.match(name)
    return int(m.group(1)) if m else None

def iter_candidates(root: Path):
    for p in root.iterdir():
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".jpg") or n.endswith(".jpg.chip.jpg"):
            yield p

def main():
    # Gom file theo từng tuổi
    by_age = defaultdict(list)
    total_files = 0
    for p in iter_candidates(RAW_DIR):
        total_files += 1
        age = parse_age(p.name)
        if age is None or age not in set(AGES):
            continue
        by_age[age].append(p)

    # xáo trộn trong từng tuổi để tránh lệch đầu danh sách
    for a in AGES:
        random.shuffle(by_age[a])

    # quota đều cho mỗi tuổi + chia phần dư
    base = MAX_SAVE // len(AGES)
    rem  = MAX_SAVE - base * len(AGES)
    target_per_age = {a: base for a in AGES}
    for a in AGES[:rem]:
        target_per_age[a] += 1

    saved_per_age = {a: 0 for a in AGES}
    saved_total, skipped = 0, 0

    # Hàng đợi các tuổi còn thiếu quota
    need_queue = deque([a for a in AGES if target_per_age[a] > 0])

    # Tạo con trỏ duyệt cho từng tuổi
    idx = {a: 0 for a in AGES}

    pbar = tqdm(total=MAX_SAVE, desc="Build non-elderly (even 20..59)")
    while need_queue and saved_total < MAX_SAVE:
        age = need_queue.popleft()

        # duyệt ảnh của tuổi này cho tới khi đạt quota hoặc hết ảnh
        while saved_per_age[age] < target_per_age[age] and idx[age] < len(by_age[age]) and saved_total < MAX_SAVE:
            p = by_age[age][idx[age]]
            idx[age] += 1
            try:
                img = Image.open(p).convert("RGB")
                aligned = mtcnn(img, save_path=str(OUT_DIR / p.name))
                if aligned is None:
                    skipped += 1
                    continue
                saved_per_age[age] += 1
                saved_total += 1
                pbar.update(1)
            except Exception:
                skipped += 1

        # Nếu chưa đủ quota mà vẫn còn ảnh → đưa lại vào queue để thử tiếp vòng sau
        if saved_per_age[age] < target_per_age[age] and idx[age] < len(by_age[age]) and saved_total < MAX_SAVE:
            need_queue.append(age)

        # Nếu tuổi này đã hết ảnh nhưng chưa đủ quota → ghi nhận shortfall
        # (phần thiếu sẽ tự động được lấp ở cuối vòng bằng các tuổi còn dư)

        # Khi tất cả tuổi trong queue đều đã cạn ảnh, vòng while sẽ thoát vì không append thêm

    pbar.close()

    # Nếu vẫn thiếu so với MAX_SAVE (vì nhiều tuổi hết ảnh), đổ nốt từ các tuổi còn ảnh
    if saved_total < MAX_SAVE:
        leftovers = []
        for a in AGES:
            leftovers.extend(by_age[a][idx[a]:])  # phần còn lại chưa dùng
        random.shuffle(leftovers)
        for p in leftovers:
            if saved_total >= MAX_SAVE:
                break
            try:
                img = Image.open(p).convert("RGB")
                aligned = mtcnn(img, save_path=str(OUT_DIR / p.name))
                if aligned is None:
                    skipped += 1
                    continue
                saved_total += 1
            except Exception:
                skipped += 1

    # ---- thống kê ----
    print(f"Total files scanned      : {total_files}")
    print(f"✔ Saved total            : {saved_total}")
    print(f"✖ Skipped (no face/error): {skipped}")
    print("Per-age saved (sample 20..29):",
          {a: saved_per_age[a] for a in range(20, 30)})
    print(f"Output dir               : {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
