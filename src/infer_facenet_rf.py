# src/infer_facenet_rf.py
from pathlib import Path
import sys

# --- ensure project root on sys.path BEFORE importing src.* ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../elderly-face-retrieval
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings_facenet import FaceNetEmbedder, load_image_any

import json
import joblib
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

MODEL_DIR = PROJECT_ROOT / "models"


def image_quality_flags(img: Image.Image, blur_cut: float = 90.0):
    """
    Kiểm tra chất lượng ảnh:
      - blurry: độ sắc nét thấp (Laplacian variance)
      - too_dark / too_bright: trung bình kênh xám quá thấp/cao
    Nếu thiếu OpenCV, trả về cờ mặc định (không chặn pipeline).
    """
    try:
        import cv2  # pip install opencv-python
        g = np.array(img.convert("L"))
        fm = cv2.Laplacian(g, cv2.CV_64F).var()  # focus measure
        mean = g.mean()
        flags = {
            "blurry": fm < float(blur_cut),   # chỉnh bằng tham số
            "too_dark": mean < 60,
            "too_bright": mean > 200,
        }
        stats = {"laplacian_var": float(fm), "gray_mean": float(mean)}
    except Exception:
        # Fallback nếu không có cv2: không đánh dấu mờ/tối/sáng
        flags = {"blurry": False, "too_dark": False, "too_bright": False}
        stats = {"laplacian_var": None, "gray_mean": None}
    return flags, stats


class ElderlyAgePipeline:
    def __init__(
        self,
        clf_path: Path = MODEL_DIR / "elderly_clf_facenet_rf.joblib",
        reg_elder_path: Path = MODEL_DIR / "age_regressor_facenet_rf.joblib",
        reg_all_path: Path = MODEL_DIR / "age_regressor_facenet_rf_all.joblib",
        agegroup_path: Path = MODEL_DIR / "agegroup_clf_facenet_rf.joblib",
        threshold: float | None = None,
        force_crop: bool = True,
        blur_cut: float = 90.0,
    ):
        self.embedder = FaceNetEmbedder()
        # elderly classifier
        clf = joblib.load(clf_path)
        calib_path = MODEL_DIR / "elderly_clf_facenet_rf_calib.joblib"
        if calib_path.exists():
            try:
                clf = joblib.load(calib_path)
            except Exception:
                pass
        self.clf_elder = clf

        self.reg_elder = joblib.load(reg_elder_path)
        self.reg_all = joblib.load(reg_all_path)
        self.agegroup_clf = joblib.load(agegroup_path)

        # auto-threshold từ file nếu có
        thr_file = MODEL_DIR / "elderly_threshold.txt"
        if threshold is None and thr_file.exists():
            try:
                threshold = float(thr_file.read_text().strip())
            except Exception:
                threshold = 0.5
        self.threshold = 0.5 if threshold is None else float(threshold)

        # detector để đếm/chọn mặt
        self.detector = MTCNN(keep_all=True, device="cpu")

        # cấu hình inference
        self.force_crop = bool(force_crop)
        self.blur_cut = float(blur_cut)

    def _crop_primary_face(self, img: Image.Image, boxes, probs) -> Image.Image:
        """
        Crop mặt có xác suất cao nhất, pad nhẹ cho an toàn.
        Nếu không crop được, trả lại ảnh gốc.
        """
        try:
            b = np.asarray(boxes, dtype=np.float32)
            p = np.asarray(probs, dtype=np.float32)
            if b.ndim != 2 or len(b) == 0:
                return img
            idx = int(np.nanargmax(p))
            x1, y1, x2, y2 = b[idx]
            # nới khung
            w = x2 - x1
            h = y2 - y1
            pad = 0.12
            x1 = max(0.0, x1 - pad * w)
            y1 = max(0.0, y1 - pad * h)
            x2 = x2 + pad * w
            y2 = y2 + pad * h
            # clamp theo kích thước ảnh
            W, H = img.size
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(W, int(round(x2)))
            y2 = min(H, int(round(y2)))
            if x2 > x1 and y2 > y1:
                return img.crop((x1, y1, x2, y2))
            return img
        except Exception:
            return img

    def predict_image(self, img_path: str):
        img = load_image_any(img_path, project_root=PROJECT_ROOT)

        # 0) kiểm tra số mặt (robust)
        boxes, probs = self.detector.detect(img)
        if boxes is None:
            n_faces = 0
        else:
            try:
                arr = np.asarray(boxes, dtype=np.float32)
                n_faces = int((~np.isnan(arr).any(axis=1)).sum())
            except Exception:
                n_faces = len(boxes) if hasattr(boxes, "__len__") else 1

        if n_faces == 0:
            return {"ok": False, "reason": "no_face_detected"}

        # luôn crop mặt chính nếu bật force_crop
        if boxes is not None and probs is not None and n_faces >= 1 and self.force_crop:
            img = self._crop_primary_face(img, boxes, probs)

        # 1) chất lượng ảnh
        flags, qstats = image_quality_flags(img, self.blur_cut)

        # 2) embedding
        vec = self.embedder.embed_pil(img).reshape(1, -1)  # (1, 512)

        # 3) elderly decision
        p_old = float(self.clf_elder.predict_proba(vec)[:, 1][0])
        is_old = bool(p_old >= self.threshold)

        # 4) age group (0:young 0-29, 1:middle 30-59, 2:elderly 60+)
        group_idx = int(self.agegroup_clf.predict(vec)[0])
        group_map = {0: "young(0-29)", 1: "middle(30-59)", 2: "elderly(60+)"}
        group_label = group_map.get(group_idx, "unknown")
        # Ép hiển thị nhất quán với quyết định elderly
        if is_old:
            group_label = "elderly(60+)"

        # 5) age estimates
        age_all = float(self.reg_all.predict(vec)[0])          # gợi ý cho mọi ảnh
        age_elder = float(self.reg_elder.predict(vec)[0]) if is_old else None  # tuổi chính thức khi elderly

        # 6) xuất kết quả
        return {
            "ok": True,
            "faces_detected": int(n_faces),
            "quality_flags": {
                "blurry": bool(flags["blurry"]),
                "too_dark": bool(flags["too_dark"]),
                "too_bright": bool(flags["too_bright"]),
            },
            "quality_stats": qstats,
            "prob_old": float(p_old),
            "is_old": bool(is_old),
            "age_group": group_label,
            "pred_age_official": None if age_elder is None else float(age_elder),
            "pred_age_all_hint": float(age_all),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/infer_facenet_rf.py path/to/image.jpg")
        raise SystemExit(1)

    pipe = ElderlyAgePipeline()
    out = pipe.predict_image(sys.argv[1])

    # Chuyển mọi kiểu numpy -> kiểu Python thuần để dump JSON an toàn
    def convert_np(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(np, "bool_") and isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    print(json.dumps(out, ensure_ascii=False, indent=2, default=convert_np))
