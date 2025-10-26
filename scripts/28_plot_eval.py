# scripts/28_plot_eval.py
"""
Vẽ & xuất báo cáo:
- Confusion matrix 3 lớp (young 0-29, middle 30-59, elderly 60+)
- PR curve & ROC curve cho bài toán Elderly vs Not (nhị phân)
- Bảng classification report (CSV + PNG) cho cả 3-lớp và nhị phân

Yêu cầu: matplotlib, numpy, pandas, scikit-learn, joblib
"""

from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # render không cần GUI
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)

PROJECT_ROOT = Path(".")
INDEX_DIR = PROJECT_ROOT / "data/index"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def save_table_png(df: pd.DataFrame, out_path: Path, title: str = "", fontsize: int = 10):
    """Vẽ DataFrame thành ảnh PNG bằng matplotlib."""
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 1.4), max(2, df.shape[0] * 0.6)))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=12)
    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   rowLabels=df.index if df.index.name or any(df.index) else None,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='Ground truth', xlabel='Predicted', title=title)

    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title="Precision-Recall (Elderly)"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, where='post')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f"{title} — AP={ap:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title="ROC (Elderly)"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def to_age_groups(ages: np.ndarray) -> np.ndarray:
    # 0: young 0–29, 1: middle 30–59, 2: elderly 60+
    return np.where(ages >= 60, 2, np.where(ages >= 30, 1, 0)).astype(np.int64)


# --------------------------
# Load test data
# --------------------------
X_test = np.load(INDEX_DIR / "test_feats_facenet.npy")

paths, ages, elders = [], [], []
with open(INDEX_DIR / "test_meta.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        paths.append(r["path"])
        ages.append(int(r["age"]))
        elders.append(int(r["elderly"]))
y_age = np.array(ages, dtype=np.int32)
y_elder = np.array(elders, dtype=np.int64)
y_groups = to_age_groups(y_age)

# --------------------------
# Load models
# --------------------------
clf_elder = joblib.load(MODEL_DIR / "elderly_clf_facenet_rf.joblib")
clf_group = joblib.load(MODEL_DIR / "agegroup_clf_facenet_rf.joblib")

# --------------------------
# 3-class evaluation (age groups)
# --------------------------
y_pred_group = clf_group.predict(X_test)
cm_group = confusion_matrix(y_groups, y_pred_group)
report_group = classification_report(y_groups, y_pred_group, output_dict=True, digits=4)
df_group = pd.DataFrame(report_group).transpose()

# Save confusion matrix (PNG)
plot_confusion_matrix(
    cm_group,
    class_names=["young (0-29)", "middle (30-59)", "elderly (60+)"],
    out_path=REPORT_DIR / "confusion_matrix_agegroup.png",
    title="Age Group Confusion Matrix"
)

# Save classification report (CSV + PNG)
df_group.to_csv(REPORT_DIR / "agegroup_classification_report.csv", index=True)
save_table_png(
    df_group.round(4),
    out_path=REPORT_DIR / "agegroup_classification_report.png",
    title="Age Group Classification Report",
    fontsize=10
)

# --------------------------
# Binary evaluation (elderly vs not) for PR/ROC
# --------------------------
y_score_elder = clf_elder.predict_proba(X_test)[:, 1]  # prob of class=elderly
# PR curve
plot_pr_curve(
    y_true=y_elder, y_score=y_score_elder,
    out_path=REPORT_DIR / "pr_curve_elderly.png",
    title="Precision–Recall Curve (Elderly vs Not)"
)
# ROC curve
plot_roc_curve(
    y_true=y_elder, y_score=y_score_elder,
    out_path=REPORT_DIR / "roc_curve_elderly.png",
    title="ROC Curve (Elderly vs Not)"
)

# Binary classification table at default threshold 0.5 (report thêm)
y_pred_elder_05 = (y_score_elder >= 0.5).astype(int)
report_elder_05 = classification_report(y_elder, y_pred_elder_05, output_dict=True, digits=4)
df_elder_05 = pd.DataFrame(report_elder_05).transpose()
df_elder_05.to_csv(REPORT_DIR / "elderly_binary_report.csv", index=True)
save_table_png(
    df_elder_05.round(4),
    out_path=REPORT_DIR / "elderly_binary_report.png",
    title="Elderly vs Not — Classification Report (thr=0.50)",
    fontsize=10
)

# (Optional) report thêm theo threshold tối ưu nếu có
thr_file = MODEL_DIR / "elderly_threshold.txt"
if thr_file.exists():
    try:
        thr = float(thr_file.read_text().strip())
        y_pred_elder_thr = (y_score_elder >= thr).astype(int)
        report_elder_thr = classification_report(y_elder, y_pred_elder_thr, output_dict=True, digits=4)
        df_elder_thr = pd.DataFrame(report_elder_thr).transpose()
        df_elder_thr.to_csv(REPORT_DIR / f"elderly_binary_report_thr_{thr:.3f}.csv", index=True)
        save_table_png(
            df_elder_thr.round(4),
            out_path=REPORT_DIR / f"elderly_binary_report_thr_{thr:.3f}.png",
            title=f"Elderly vs Not — Classification Report (thr={thr:.3f})",
            fontsize=10
        )
    except Exception:
        pass

# --------------------------
# Save a small JSON summary pointer
# --------------------------
summary = {
    "files": {
        "confusion_matrix_agegroup_png": str((REPORT_DIR / "confusion_matrix_agegroup.png").resolve()),
        "pr_curve_elderly_png": str((REPORT_DIR / "pr_curve_elderly.png").resolve()),
        "roc_curve_elderly_png": str((REPORT_DIR / "roc_curve_elderly.png").resolve()),
        "agegroup_report_csv": str((REPORT_DIR / "agegroup_classification_report.csv").resolve()),
        "agegroup_report_png": str((REPORT_DIR / "agegroup_classification_report.png").resolve()),
        "elderly_binary_report_csv": str((REPORT_DIR / "elderly_binary_report.csv").resolve()),
        "elderly_binary_report_png": str((REPORT_DIR / "elderly_binary_report.png").resolve()),
    }
}
with open(REPORT_DIR / "plots_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Saved report to:", REPORT_DIR.resolve())
