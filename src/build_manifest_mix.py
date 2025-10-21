# src/build_manifest_mix.py
import csv, re, random
from pathlib import Path

random.seed(42)

ROOT = Path(".")
ELDER_DIR = ROOT/"data/aligned_clean"     # > 700 ảnh ≥60 (đã clean)
NON_DIR   = ROOT/"data/non_elderly"       # 1400 ảnh 20–59 (đã align MTCNN)
OUT_DIR   = ROOT/"data/manifests"; OUT_DIR.mkdir(parents=True, exist_ok=True)

age_re = re.compile(r"^(\d+)_")

def parse_age(name:str):
    m = age_re.match(name)
    return int(m.group(1)) if m else None

def list_items(d: Path, elderly_flag: int):
    rows=[]
    for p in d.glob("*.jpg"):
        age = parse_age(p.name)
        if age is None: 
            continue
        rows.append({"path": str(p), "age": age, "elderly": elderly_flag})
    return rows

elder = list_items(ELDER_DIR, 1)  # ≥60
non   = list_items(NON_DIR,   0)  # 20–59

print(f"Elderly clean: {len(elder)} | Non-elderly pool: {len(non)}")

# ====== sampling strategy ======
TARGET_RATIO = 2   # 1:2  (elderly : non-elderly). Đổi 1 nếu muốn 1:1
k_non = min(len(non), TARGET_RATIO * len(elder))

# Optional: cân theo thập kỷ 20s/30s/40s/50s
bins = [(20,29),(30,39),(40,49),(50,59)]
by_bin = {b:[] for b in bins}
for r in non:
    for lo,hi in bins:
        if lo <= r["age"] <= hi:
            by_bin[(lo,hi)].append(r); break

each = k_non // len(bins)
picked=[]
for b in bins:
    candidates = by_bin[b]
    random.shuffle(candidates)
    picked.extend(candidates[:each])
# phần dư
remain = k_non - len(picked)
others = [r for b in bins for r in by_bin[b][each:]]
random.shuffle(others); picked.extend(others[:remain])

non_sel = picked
print(f"Sampled non-elderly: {len(non_sel)}")

data = elder + non_sel
random.shuffle(data)

# ====== split 80/10/10 stratified theo elderly ======
def split_stratified(rows):
    pos=[r for r in rows if r["elderly"]==1]
    neg=[r for r in rows if r["elderly"]==0]
    def split(arr):
        n=len(arr); tr=int(0.8*n); va=int(0.9*n)
        return arr[:tr], arr[tr:va], arr[va:]
    random.shuffle(pos); random.shuffle(neg)
    tr1,va1,te1 = split(pos)
    tr0,va0,te0 = split(neg)
    train = tr1+tr0; val = va1+va0; test = te1+te0
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train,val,test

train,val,test = split_stratified(data)

for name,rows in [("train.csv",train),("val.csv",val),("test.csv",test)]:
    with open(OUT_DIR/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["path","age","elderly"])
        w.writeheader(); w.writerows(rows)
    print(name, "→", len(rows))
