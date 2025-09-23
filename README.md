# Elderly Face Retrieval

Bài toán **tìm kiếm ảnh mặt người cao tuổi** (Content-Based Image Retrieval – CBIR).  
Đầu vào: 1 ảnh query (jpg/png) → Đầu ra: **Top-3 ảnh giống nhất** trong tập dữ liệu.  

---

## 📌 Pipeline hiện tại

1. **Tiền xử lý (Preprocess)**
   - Align + resize ảnh khuôn mặt về kích thước chuẩn (224×224).
   - Loại bỏ ảnh mờ, chứa watermark, trùng lặp → lưu vào `data/aligned_clean`.

2. **Trích rút đặc trưng (Feature Extraction)**
   - Hỗ trợ 2 phương pháp:
     - `lbp`: Local Binary Pattern (truyền thống).
     - `facenet`: Embedding 512 chiều từ mạng **InceptionResnetV1 (VGGFace2)**.
   - Kết quả lưu tại thư mục `data/index/`:
     - `features_<method>.npy` – ma trận đặc trưng.
     - `ids_<method>.csv` – id + path ảnh.
     - `meta_<method>.json` – thông tin meta.

3. **Truy vấn (Query)**
   - Tải index đã có.
   - Trích đặc trưng cho ảnh query.
   - Tính **cosine similarity** với toàn bộ index.
   - Xuất **Top-3 ảnh giống nhất** và (tuỳ chọn) ghép grid để xem.

---

## Cấu trúc thư mục
```bash
elderly-face-retrieval/
│
├─ data/
│ ├─ raw/ # ảnh gốc
│ ├─ aligned/ # ảnh sau align
│ ├─ aligned_clean/ # ảnh đã lọc sạch
│ └─ index/ # file index (features, ids, meta)
│
├─ src/
│ ├─ preprocess.py # detect/align/resize
│ ├─ clean_check.py # lọc blur/watermark/duplicate
│ ├─ extract_features.py # trích đặc trưng + lưu index
│ └─ query.py # tìm kiếm Top-3
│
├─ report/
│ └─ top3_xxx.jpg # ảnh minh hoạ query + kết quả
│
├─ requirements.txt
└─ README.md

```
---

## Cài đặt môi trường

```bash
# clone repo
git clone https://github.com/<your-username>/elderly-face-retrieval.git
cd elderly-face-retrieval

# tạo venv
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate # Linux/Mac

# cài dependencies
pip install -r requirements.txt

```
## Chạy thử
1. **Tiền xử lý + lọc ảnh**
```bash
python src/preprocess.py
python src/clean_check.py
```
2. **Trích rút đặc trưng**
```bash
# Facenet (deep features)
python src/extract_features.py --method facenet --device cpu

# LBP (truyền thống)
python src/extract_features.py --method lbp
```
3. **Truy vấn Top-3 ảnh giống nhất**
```bash
python src/query.py --image "data/aligned_clean/<your_image>.jpg" --method facenet --device cpu --save-grid
```
Kết quả:
 - In ra Top-3 ảnh giống nhất + độ tương đồng (cosine).
 - Nếu có --save-grid thì lưu ảnh minh hoạ vào thư mục report/.

---
## License
Chỉ sử dụng cho mục đích học thuật / demo nội bộ.
