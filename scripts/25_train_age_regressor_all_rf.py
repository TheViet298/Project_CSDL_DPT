"""
Train thêm regressor toàn dải tuổi (all ages)

để có tuổi ước lượng cho cả non-elderly
"""
from pathlib import Path
import csv, joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str):
    X = np.load(INDEX_DIR / f"{name}_feats_facenet.npy")
    ages = []
    with open(INDEX_DIR / f"{name}_meta.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ages.append(int(r["age"]))
    return X, np.array(ages, dtype=np.float32)

if __name__ == "__main__":
    Xtr1, atr1 = load_split("train")
    Xtr2, atr2 = load_split("val")
    Xtr = np.concatenate([Xtr1, Xtr2], axis=0)
    atr = np.concatenate([atr1, atr2], axis=0)

    reg = RandomForestRegressor(
        n_estimators=800, n_jobs=-1, random_state=42
    )
    reg.fit(Xtr, atr)
    joblib.dump(reg, MODEL_DIR / "age_regressor_facenet_rf_all.joblib")
    print("saved:", MODEL_DIR / "age_regressor_facenet_rf_all.joblib")

    # evaluate on full test set
    Xte, ate = load_split("test")
    pred = reg.predict(Xte)
    mae = mean_absolute_error(ate, pred)
    r2  = r2_score(ate, pred)
    print(f"ALL-AGES test MAE = {mae:.3f} | R² = {r2:.3f} | n={len(ate)}")
