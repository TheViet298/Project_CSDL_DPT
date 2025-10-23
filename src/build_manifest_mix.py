# scripts/build_manifest_mix_v2.py
import csv, re, random
from pathlib import Path

random.seed(42)

ROOT       = Path(".")
ELDER_DIR  = ROOT/"data/aligned_clean"   # elderly (>=60)
NON_DIR    = ROOT/"data/non_elderly"     # 0..59 đã align
OUT_DIR    = ROOT/"data/manifests"; OUT_DIR.mkdir(parents=True, exist_ok=True)

age_re = re.compile(r"^(\d+)_")

def parse_age(name:str):
    m = age_re.match(name); return int(m.group(1)) if m else None

def list_items(d: Path, elderly_flag: int):
    rows=[]
    for p in d.glob("*.jpg"):
        a = parse_age(p.name)
        if a is None: continue
        rows.append({"path": str(p), "age": a, "elderly": elderly_flag})
    return rows

elder = list_items(ELDER_DIR, 1)  # >=60
non   = list_items(NON_DIR,   0)  # 0..59

print(f"Elderly: {len(elder)} | Non pool: {len(non)}")

# === sampling 1:2 theo elderly count, nhưng bảo toàn đủ tuổi 0..59 ===
TARGET_RATIO = 2
k_non = min(len(non), TARGET_RATIO * len(elder))

# chia non theo decade 0-9,10-19,...,50-59 với trọng số nhẹ ưu tiên 40-59
decades = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59)]
weights = { (0,9):0.7, (10,19):0.9, (20,29):1.0, (30,39):1.0, (40,49):1.1, (50,59):1.2 }
bucket = {d:[] for d in decades}
for r in non:
    for lo,hi in decades:
        if lo <= r["age"] <= hi:
            bucket[(lo,hi)].append(r); break

# quota theo weight
S = sum(weights.values())
quota = {d: int(round(k_non * (weights[d]/S))) for d in decades}
# fix lệch do round
diff = k_non - sum(quota.values())
order = [(50,59),(40,49),(30,39),(20,29),(10,19),(0,9)]
i=0
while diff!=0:
    d = order[i%len(order)]
    quota[d] += 1 if diff>0 else -1
    diff += -1 if diff>0 else 1
    i+=1

picked=[]
for d in decades:
    cand = bucket[d]; random.shuffle(cand)
    take = min(len(cand), quota[d])
    picked.extend(cand[:take])

# nếu thiếu do không đủ ảnh ở 1 số decade → đổ bù
remain = k_non - len(picked)
if remain > 0:
    rest = [r for d in decades for r in bucket[d][quota[d]:]]
    random.shuffle(rest); picked.extend(rest[:remain])

non_sel = picked
print(f"Sampled non-elderly: {len(non_sel)}")

data = elder + non_sel
random.shuffle(data)

# === split 80/10/10 – stratified theo elderly ===
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
