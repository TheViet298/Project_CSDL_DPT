# 🧩 6. Đánh giá và So sánh Kết quả

## 6.1 Phương pháp và Thiết lập Thí nghiệm

Hệ thống thử nghiệm hai phương pháp trích xuất đặc trưng:

| Phương pháp | Mô tả | Kích thước vector | Thư viện sử dụng |
|--------------|--------|-------------------|------------------|
| **LBP (Local Binary Pattern)** | Đặc trưng texture (độ sáng – tối cục bộ) bằng histogram mẫu nhị phân. | 10 chiều (P=8, R=1, uniform pattern) | `skimage.feature.local_binary_pattern` |
| **FaceNet (InceptionResnetV1)** | Đặc trưng học sâu từ mạng nhận diện khuôn mặt pretrained trên **VGGFace2**, cho embedding 512 chiều. | 512 chiều | `facenet-pytorch` |

Mỗi ảnh trong tập `data/aligned_clean/` được:
- Resize về kích thước **224×224**.
- Trích đặc trưng và lưu tại `data/index/`.
- Khi truy vấn, tính **cosine similarity** giữa vector query và toàn bộ index, sau đó trả về **Top-3 ảnh giống nhất**.

---

## 6.2 Kết quả Truy vấn Minh họa

**Ảnh truy vấn:**  
`60_0_0_20170103182716210.jpg.chip.jpg`

| Phương pháp | Top-3 ảnh trả về | Nhận xét |
|--------------|------------------|-----------|
| **LBP** | <img src="../report/top3_60_0_0_20170103182716210.jpg.chip_lbp.jpg" width="500"> | Một số ảnh không cùng giới tính hoặc ánh sáng khác biệt. LBP nhạy với texture và ánh sáng, dễ sai khi phông nền hoặc contrast thay đổi. |
| **FaceNet** | <img src="../report/top3_60_0_0_20170103182716210.jpg.chip_facenet.jpg" width="500"> | Ảnh trả về có độ tương đồng cao về **tuổi, hướng mặt, nếp nhăn và tông màu da**, thể hiện khả năng nhận diện hình dạng tốt của mô hình học sâu. |

---

## 6.3 Đánh giá Định tính

| Tiêu chí | LBP | FaceNet |
|-----------|-----|----------|
| Độ chính xác trực quan | Trung bình | Cao |
| Ảnh hưởng bởi ánh sáng | Rất nhạy | Ổn định |
| Ổn định với góc mặt | Kém | Tốt |
| Tốc độ trích xuất | Rất nhanh (~0.01s/ảnh) | Trung bình (~0.2s/ảnh, CPU) |
| Kích thước vector | Nhỏ | Lớn (512 float) |
| Ứng dụng phù hợp | Minh họa thuật toán texture | Ứng dụng thực tế, face retrieval |

---

## 6.4 Nhận xét Tổng hợp

- **FaceNet** cho kết quả chính xác và ổn định hơn rõ rệt, đặc biệt khi ảnh có góc nhìn hoặc ánh sáng khác nhau.  
- **LBP** có tốc độ nhanh, dễ cài đặt nhưng chỉ phù hợp cho mục đích minh họa, không đủ mạnh cho nhận dạng thực tế.  
- Với quy mô dữ liệu nhỏ (~500 ảnh), **linear scan với cosine similarity** vẫn đảm bảo thời gian truy vấn <1s.

---

## 6.5 Hướng Mở Rộng

- Kết hợp **LBP + Deep Embedding** (feature fusion) để tăng robust.
- Dùng **FAISS** hoặc **pgvector** để tăng tốc tìm kiếm cho dữ liệu lớn.
- Bổ sung **Top-1 / Top-3 accuracy** nếu có nhãn người (`person_id`).
- Phát triển **UI Streamlit** để người dùng upload ảnh và xem kết quả trực quan.

---

✳️ *Tổng kết:*  
So sánh giữa LBP và FaceNet cho thấy sự khác biệt rõ ràng giữa **đặc trưng thủ công (handcrafted)** và **đặc trưng học sâu (deep feature embedding)**.  
Trong bài toán *Elderly Face Retrieval*, đặc trưng học sâu thể hiện ưu thế vượt trội về khả năng biểu diễn khuôn mặt người cao tuổi trong các điều kiện khác nhau.
