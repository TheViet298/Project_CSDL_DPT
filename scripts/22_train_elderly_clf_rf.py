# scripts/22_train_elderly_clf_rf.py
import csv, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str):
    X = np.load(INDEX_DIR / f"{name}_feats_facenet.npy")
    y = []
    with open(INDEX_DIR / f"{name}_meta.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y.append(int(r["elderly"]))
    return X, np.array(y, dtype=np.int64)

if __name__ == "__main__":
    # gộp train + val để train cuối
    Xtr1, ytr1 = load_split("train")
    Xtr2, ytr2 = load_split("val")
    Xtr = np.concatenate([Xtr1, Xtr2], axis=0)
    ytr = np.concatenate([ytr1, ytr2], axis=0)

    Xte, yte = load_split("test")

    # RF mạnh, ít cần chuẩn hoá, cho cả prob
    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(Xtr, ytr)

    yhat = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1]
    print("Confusion matrix:\n", confusion_matrix(yte, yhat))
    print(classification_report(yte, yhat, digits=4))
    print("ROC-AUC:", roc_auc_score(yte, yprob))

    joblib.dump(clf, MODEL_DIR / "elderly_clf_facenet_rf.joblib")
    print("saved:", MODEL_DIR / "elderly_clf_facenet_rf.joblib")
