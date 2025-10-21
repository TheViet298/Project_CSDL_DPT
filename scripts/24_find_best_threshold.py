from pathlib import Path
import csv, joblib, numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True, parents=True)

# load val split
Xv  = np.load(INDEX_DIR / "val_feats_facenet.npy")
yv  = []
with open(INDEX_DIR / "val_meta.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        yv.append(int(r["elderly"]))
yv = np.array(yv)

# load classifier
clf = joblib.load(MODEL_DIR / "elderly_clf_facenet_rf.joblib")
p = clf.predict_proba(Xv)[:,1]

# tìm threshold tối ưu theo F1 của lớp elderly=1
prec, rec, thr = precision_recall_curve(yv, p, pos_label=1)
f1s = 2*prec*rec/(prec+rec+1e-9)
best_idx = int(np.nanargmax(f1s))
best_thr = float(thr[max(0, best_idx-1)]) if best_idx < len(thr) else 0.5

out = {
    "best_threshold_F1_on_val": best_thr,
    "best_F1": float(np.nanmax(f1s)),
    "val_support": int(len(yv))
}
print(out)

# lưu ra file để inference tự đọc
(MODEL_DIR / "elderly_threshold.txt").write_text(str(best_thr))
print("saved:", MODEL_DIR / "elderly_threshold.txt")
