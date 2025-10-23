from pathlib import Path
import csv, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str):
    X = np.load(INDEX_DIR / f"{name}_feats_facenet.npy")
    ages = []
    with open(INDEX_DIR / f"{name}_meta.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ages.append(int(r["age"]))
    ages = np.array(ages, dtype=np.int32)
    # bins: young < 30, middle 30–59, elderly 60+
    ygroup = np.where(ages >= 60, 2, np.where(ages >= 30, 1, 0)).astype(np.int64)
    return X, ygroup

if __name__ == "__main__":
    Xtr1, y1 = load_split("train")
    Xtr2, y2 = load_split("val")
    Xtr = np.concatenate([Xtr1, Xtr2], axis=0)
    ytr = np.concatenate([y1, y2], axis=0)

    Xte, yte = load_split("test")

    clf = RandomForestClassifier(
        n_estimators=600, n_jobs=-1, random_state=42
    )
    clf.fit(Xtr, ytr)

    yhat = clf.predict(Xte)
    print("Confusion matrix (rows=GT, cols=Pred) 0:young 1:middle 2:elderly")
    print(confusion_matrix(yte, yhat))
    print(classification_report(yte, yhat, digits=4))

    joblib.dump(clf, MODEL_DIR / "agegroup_clf_facenet_rf.joblib")
    print("saved:", MODEL_DIR / "agegroup_clf_facenet_rf.joblib")
