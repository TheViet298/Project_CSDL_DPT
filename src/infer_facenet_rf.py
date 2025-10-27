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
import cv2
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

        # elderly classifier (ưu tiên bản calibrated nếu có)
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

        # cấu hình inference mặc định
        self.force_crop = bool(force_crop)
        self.blur_cut = float(blur_cut)

    # --------- helpers ----------
    def _clip_box(self, box, w, h, pad=0):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 <= x1: x2 = min(w, x1 + 1)
        if y2 <= y1: y2 = min(h, y1 + 1)
        return (x1, y1, x2, y2)

    # --------- main predict ----------
    def predict_image(self,
                      img_path: str,
                      threshold: float | None = None,
                      blur_cut: float | None = None,
                      force_crop: bool | None = None):
        """
        Trả về:
          {
            ok, faces_detected, primary_idx, select_strategy, faces: [...],
            # các trường top-level tương thích UI: quality_flags, prob_old, ...
          }
        """
        # backup & override tạm thời
        _thr, _blur, _crop = self.threshold, self.blur_cut, self.force_crop
        if threshold is not None: self.threshold = float(threshold)
        if blur_cut  is not None: self.blur_cut  = float(blur_cut)
        if force_crop is not None: self.force_crop = bool(force_crop)

        try:
            img = load_image_any(img_path, project_root=PROJECT_ROOT)

            # 1) detect tất cả mặt
            boxes, probs = self.detector.detect(img)
            if boxes is None or len(boxes) == 0:
                return {"ok": False, "reason": "no_face_detected"}

            boxes = np.asarray(boxes, dtype=np.float32)
            keep = ~np.isnan(boxes).any(axis=1)
            boxes = boxes[keep]
            if probs is not None:
                probs = np.asarray(probs)[keep]

            n_faces = int(len(boxes))
            if n_faces == 0:
                return {"ok": False, "reason": "no_face_detected"}

            W, H = img.size
            faces_out = []
            areas = []

            # 2) tính kết quả cho từng mặt (crop nhẹ 10px)
            for i, b in enumerate(boxes):
                crop_box = self._clip_box(b, W, H, pad=10)
                face_img = img.crop(crop_box) if self.force_crop else img

                flags, qstats = image_quality_flags(face_img, self.blur_cut)
                vec = self.embedder.embed_pil(face_img).reshape(1, -1)

                p_old = float(self.clf_elder.predict_proba(vec)[:, 1][0])
                is_old = bool(p_old >= self.threshold)

                group_idx = int(self.agegroup_clf.predict(vec)[0])
                group_map = {0: "young(0-29)", 1: "middle(30-59)", 2: "elderly(60+)"}
                group_label = "elderly(60+)" if is_old else group_map.get(group_idx, "unknown")

                age_all = float(self.reg_all.predict(vec)[0])
                age_elder = float(self.reg_elder.predict(vec)[0]) if is_old else None

                x1, y1, x2, y2 = crop_box
                area = max(1, (x2 - x1) * (y2 - y1))
                areas.append(area)

                faces_out.append({
                    "index": int(i),
                    "box": [int(x) for x in crop_box],
                    "det_prob": None if probs is None else float(probs[i]),
                    "quality_flags": {
                        "blurry": bool(flags["blurry"]),
                        "too_dark": bool(flags["too_dark"]),
                        "too_bright": bool(flags["too_bright"]),
                    },
                    "quality_stats": qstats,
                    "prob_old": p_old,
                    "is_old": is_old,
                    "age_group": group_label,
                    "pred_age_official": None if age_elder is None else float(age_elder),
                    "pred_age_all_hint": float(age_all),
                })

            # 3) chọn primary: mặt lớn nhất
            primary_idx = int(np.argmax(np.asarray(areas)))
            strategy = "largest_area"

            # 4) back-compat: copy primary lên top-level
            pf = faces_out[primary_idx]
            out = {
                "ok": True,
                "faces_detected": n_faces,
                "primary_idx": primary_idx,
                "select_strategy": strategy,
                "faces": faces_out,

                "quality_flags": pf["quality_flags"],
                "quality_stats": pf["quality_stats"],
                "prob_old": pf["prob_old"],
                "is_old": pf["is_old"],
                "age_group": pf["age_group"],
                "pred_age_official": pf["pred_age_official"],
                "pred_age_all_hint": pf["pred_age_all_hint"],
            }
            return out

        finally:
            # restore tham số gốc
            self.threshold, self.blur_cut, self.force_crop = _thr, _blur, _crop


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
