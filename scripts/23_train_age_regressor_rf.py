# scripts/23_train_age_regressor_rf.py
import csv, joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str):
    X = np.load(INDEX_DIR / f"{name}_feats_facenet.npy")
    ages, elderly = [], []
    with open(INDEX_DIR / f"{name}_meta.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ages.append(int(r["age"]))
            elderly.append(int(r["elderly"]))
    return X, np.array(ages, dtype=np.float32), np.array(elderly, dtype=np.int64)

if __name__ == "__main__":
    Xtr1, atr1, otr1 = load_split("train")
    Xtr2, atr2, otr2 = load_split("val")
    Xtr = np.concatenate([Xtr1, Xtr2], axis=0)
    atr = np.concatenate([atr1, atr2], axis=0)
    otr = np.concatenate([otr1, otr2], axis=0)

    # chỉ lấy elderly cho training regressor
    idx_tr = np.where(otr == 1)[0]
    Xtr_e = Xtr[idx_tr]; atr_e = atr[idx_tr]
    print("Train elderly count:", len(idx_tr))

    # Model
    reg = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(Xtr_e, atr_e)
    joblib.dump(reg, MODEL_DIR / "age_regressor_facenet_rf.joblib")
    print("saved:", MODEL_DIR / "age_regressor_facenet_rf.joblib")

    # Đánh giá trên elderly của test
    Xte, ate, ote = load_split("test")
    idx_te = np.where(ote == 1)[0]
    if len(idx_te) > 0:
        pred = reg.predict(Xte[idx_te])
        mae = mean_absolute_error(ate[idx_te], pred)
        r2  = r2_score(ate[idx_te], pred)
        print(f"ELDERLY test MAE = {mae:.3f} | R² = {r2:.3f} | n={len(idx_te)}")
    else:
        print("Không có mẫu elderly trong test để đánh giá.")
