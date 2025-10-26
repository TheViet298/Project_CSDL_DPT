from pathlib import Path
import csv, joblib, numpy as np
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_split(name):
    X = np.load(INDEX_DIR / f"{name}_feats_facenet.npy")
    y = []
    with open(INDEX_DIR / f"{name}_meta.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y.append(int(r["elderly"]))
    return X, np.array(y, dtype=np.int64)

if __name__ == "__main__":
    # dùng train để fit base clf, val để calibrate
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")

    base = joblib.load(MODEL_DIR / "elderly_clf_facenet_rf.joblib")
    # method="isotonic" mượt hơn nếu data đủ; "sigmoid" thì nhanh & ổn định
    calib = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calib.fit(Xva, yva)

    joblib.dump(calib, MODEL_DIR / "elderly_clf_facenet_rf_calib.joblib")
    print("saved:", MODEL_DIR / "elderly_clf_facenet_rf_calib.joblib")
