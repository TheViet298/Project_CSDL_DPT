# src/infer_facenet_rf.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings_facenet import FaceNetEmbedder, load_image_any

import joblib, json
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(".")
MODEL_DIR = PROJECT_ROOT / "models"

class ElderlyAgePipeline:
    def __init__(self,
                 clf_path: Path = MODEL_DIR / "elderly_clf_facenet_rf.joblib",
                 reg_path: Path = MODEL_DIR / "age_regressor_facenet_rf.joblib",
                 threshold: float | None = None):
        self.embedder = FaceNetEmbedder()
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)
        thr_file = MODEL_DIR / "elderly_threshold.txt"
        if threshold is None and thr_file.exists():
            try:
                threshold = float(thr_file.read_text().strip())
            except Exception:
                threshold = 0.5
        self.threshold = 0.5 if threshold is None else float(threshold)

    def predict_image(self, img_path: str):
        img = load_image_any(img_path, project_root=PROJECT_ROOT)
        vec = self.embedder.embed_pil(img).reshape(1, -1)  # (1,512)
        p_old = float(self.clf.predict_proba(vec)[:, 1][0])
        if p_old >= self.threshold:
            age = float(self.reg.predict(vec)[0])
            return {"is_old": True, "prob_old": p_old, "pred_age": age}
        else:
            return {"is_old": False, "prob_old": p_old, "pred_age": None}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer_facenet_rf.py path/to/image.jpg")
        raise SystemExit(1)
    pipe = ElderlyAgePipeline()
    out = pipe.predict_image(sys.argv[1])
    print(json.dumps(out, ensure_ascii=False, indent=2))
