# Elderly Face Retrieval

Dự án phát hiện **người cao tuổi (60+)** và **ước lượng tuổi** từ ảnh chân dung bằng **FaceNet (InceptionResnetV1, vggface2)** + **RandomForest**.  
Bao gồm **CLI** và **Streamlit GUI** (đa khuôn mặt, chọn mặt, điều chỉnh ngưỡng trực tiếp).

---

## 1) Kiến trúc & Ý tưởng

- **Trích xuất embedding**: `FaceNet (InceptionResnetV1, pretrained='vggface2')` → vector 512-d.
- **Phân loại elderly (60+)**: `RandomForestClassifier` trên embedding.
- **Ước lượng tuổi**:
  - `age_regressor_facenet_rf.joblib` (chỉ khi đã là elderly) → *pred_age_official*.
  - `age_regressor_facenet_rf_all.joblib` (cho mọi ảnh) → *pred_age_all_hint* (chỉ hiển thị ở **Chế độ Chuyên gia**).
- **Nhóm tuổi 3 lớp**: young (0–29) / middle (30–59) / elderly (60+) để tham khảo.
- **Đa khuôn mặt**: detect tất cả, mặc định chọn **mặt lớn nhất**, cho phép **chọn lại** trong UI.
- **Ngưỡng elderly**: có thể:
  - Lấy tự động từ tập *val* (`models/elderly_threshold.txt`) — ví dụ: `0.3183…`
  - Hoặc chỉnh live trong app (slider).

---

## 2) Cấu trúc thư mục
```bash
elderly-face-retrieval/
├─ app.py # Streamlit GUI
├─ README.md
├─ models/ # .joblib và files ngưỡng
│ ├─ elderly_clf_facenet_rf.joblib
│ ├─ elderly_clf_facenet_rf_calib.joblib (nếu có)
│ ├─ age_regressor_facenet_rf.joblib
│ ├─ age_regressor_facenet_rf_all.joblib
│ ├─ agegroup_clf_facenet_rf.joblib
│ └─ elderly_threshold.txt
├─ data/
│ ├─ raw/utkface/ # UTKFace gốc
│ ├─ aligned/ # ảnh đã align (nếu có)
│ ├─ aligned_clean/ # elderly (>=60) đã clean
│ ├─ non_elderly/ # 0–59 đã align/crop bằng MTCNN
│ ├─ index/ # npy/csv index & features
│ └─ manifests/ # manifest train/val/test
├─ scripts/
│ ├─ 01_build_non_elderly.py # lấy và align/crop 0–59 cân bằng
│ ├─ 21_extract_embeddings_facenet.py
│ ├─ 22_train_elderly_clf_rf.py
│ ├─ 23_train_age_regressor_rf.py
│ ├─ 24_find_best_threshold.py
│ ├─ 26_train_agegroup_clf_rf.py
│ └─ 28_plot_eval.py
└─ src/
├─ embeddings_facenet.py # FaceNetEmbedder + load_image_any
└─ infer_facenet_rf.py # pipeline inference (đa mặt)
```
---

## 3) Thiết lập môi trường
Yêu cầu: 
- **Python 3.11**
- CUDA (nếu dùng GPU)
- pip

```powershell
# Tạo môi trường ảo
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1

# Cài gói
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # hoặc cpu
pip install facenet-pytorch scikit-learn numpy pandas pillow opencv-python joblib tqdm streamlit matplotlib
```
- **Nếu lỗi khi kích hoạt .ps1**, chạy:
```
Set-ExecutionPolicy RemoteSigned
```
---

## 4) Chuẩn bị dữ liệu
- Elderly (>=60): để ở data/aligned_clean/ (đã clean/align).
- Non-elderly (0–59): sinh từ UTKFace gốc:
```bash
# dựng non-elderly 0..59, cân bằng theo tuổi (mặc định ~1400 ảnh)
python scripts/01_build_non_elderly.py
```
- Tự động crop & cân bằng dữ liệu theo tuổi.
- Output: data/non_elderly/ (ảnh 160×160 .jpg).
---

## 5) Tạo manifest & trích xuất embedding
- Tạo manifest train/val/test theo cấu trúc(hoặc dùng script có sẵn)
- Trích xuất embedding:
```bash
python scripts/21_extract_embeddings_facenet.py
```
- Tạo:
```bash
data/index/train_feats_facenet.npy + train_meta.csv
data/index/val_feats_facenet.npy   + val_meta.csv
data/index/test_feats_facenet.npy  + test_meta.csv
```
---

## 6) Huấn luyện mô hình
**a. Elderly Classifier (2 lớp: elderly vs non-elderly)**
```bash
python scripts/22_train_elderly_clf_rf.py
```
**b. Age Regressor cho elderly**
```bash
python scripts/23_train_age_regressor_rf.py
```
**c. Tìm threshold tối ưu**
```bash
python scripts/24_find_best_threshold.py
```
**d. Age Group Classifier (0–1–2)**
```bash
python scripts/26_train_agegroup_clf_rf.py
```
**e. Đánh giá mô hình**
```bash
python scripts/28_plot_eval.py
```
- Xuất confusion matrix, PR/ROC curves, classification report.

---
## 7) Chạy suy luận (CLI)
```bash
python src/infer_facenet_rf.py "path/to/image.jpg"
```
- Ví dụ output:
```bash
{
  "ok": true,
  "faces_detected": 2,
  "primary_idx": 1,
  "select_strategy": "largest_area",
  "prob_old": 0.815,
  "is_old": true,
  "age_group": "elderly(60+)",
  "pred_age_official": 66.8,
  "pred_age_all_hint": 57.1,
  "faces": [
    { "index": 0, "prob_old": 0.41, ... },
    { "index": 1, "prob_old": 0.82, ... }
  ]
}
```
---
## 8) Chạy giao diện GUI (Streamlit)
```bash
streamlit run app.py
```
**Giao diện chính:**
- Upload ảnh → tự detect tất cả khuôn mặt
- Chọn mặt → xem kết quả chi tiết
- Sidebar:
  - **Threshold elderly:** chỉnh ngưỡng dự đoán
  - **Blur threshold:** cảnh báo ảnh mờ
  - **Crop trước khi embed**
  - **Chế độ hiển thị:** Chuẩn đề bài (ẩn tuổi non-elderly) hoặc Chuyên gia (hiện tuổi gợi ý)
- Hiển thị mức độ chắc chắn + cảnh báo chất lượng ảnh
---
## 9) Kết quả mẫu (tham khảo)
```bash
| Task                   | Metric   | Result       |
| ---------------------- | -------- | ------------ |
| Elderly vs Non-elderly | Accuracy | ~0.87        |
| Age Group (0/1/2)      | Accuracy | ~0.818       |
| Age Regression Elderly | MAE / R² | ~5.46 / 0.59 |
```
- Kết quả có thể thay đổi tuỳ theo cách chia dữ liệu và seed ngẫu nhiên.
---
## 10) Troubleshooting
- No module named 'src' → chạy từ thư mục gốc dự án.
- TypeError: bool_ not serializable → đã fix trong code inference.
- Ảnh không phát hiện khuôn mặt → thử ảnh rõ nét, chính diện.
- OpenCV lỗi → pip install opencv-python. Nếu không có, hệ thống sẽ fallback an toàn.
- Ảnh nhiều mặt → UI mới cho phép chọn mặt chính xác.
---
## License
- Chỉ sử dụng cho mục đích học thuật và demo nội bộ.
- Dữ liệu huấn luyện lấy từ UTKFace — tuân thủ giấy phép dữ liệu gốc.
- Mô hình embedding sử dụng facenet-pytorch (MIT License).
