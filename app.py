# app.py — Streamlit GUI (multi-face + compact + confidence bar + warnings)
import os, sys
from pathlib import Path
import streamlit as st
from PIL import Image

# === đảm bảo src có thể import ===
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Elderly Age Detection", page_icon="🧓", layout="wide")

# ====== CSS tùy chỉnh ======
st.markdown("""
<style>
.block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 1rem; }
img { max-width: 500px !important; height: auto; display: block; margin: auto; }
.big-score { font-size: 30px; font-weight: 700; }
.ok-chip  { background:#e6ffed; color:#037a24; padding:4px 10px; border-radius:10px; display:inline-block; font-size:14px; }
.bad-chip { background:#ffe9e9; color:#b00020; padding:4px 10px; border-radius:10px; display:inline-block; font-size:14px; }
.muted { color:#6b7280; }
.stTabs [role="tablist"] { gap: 8px; }
.stTabs [role="tab"] { font-size: 14px; padding: 4px 10px; }
.card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🧓 Elderly Age Detection & Estimation")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Thiết lập")
    thr = st.slider("Threshold elderly (cho inference)", 0.10, 0.90, 0.50, 0.01)
    blur_cut = st.slider("Ngưỡng 'mờ' (Laplacian < …)", 10, 200, 90, 1)
    force_crop = st.checkbox("Crop mặt trước khi embed", value=True)
    mode = st.radio(
        "Chế độ hiển thị",
        ["Chuẩn đề bài", "Chuyên gia"],
        index=0,
        help="Chuẩn đề bài: chỉ hiện tuổi khi elderly. Chuyên gia: hiện thêm gợi ý tuổi cho mọi ảnh."
    )
    st.divider()
    st.caption(f"Working dir: `{os.getcwd()}`")
    st.caption(f"Python: `{sys.version.split()[0]}`")

@st.cache_resource(show_spinner=False)
def load_pipe():
    from src.infer_facenet_rf import ElderlyAgePipeline, MODEL_DIR
    pipe = ElderlyAgePipeline()  # cache mô hình 1 lần
    thr_file = (MODEL_DIR / "elderly_threshold.txt")
    thr_auto = None
    if thr_file.exists():
        try: thr_auto = float(thr_file.read_text().strip())
        except: pass
    return pipe, thr_auto

pipe, thr_auto = load_pipe()

with st.sidebar:
    if thr_auto is not None:
        st.caption("Threshold elderly (tự động từ val):")
        st.code(str(thr_auto))

# ========== UPLOAD ==========
uploaded = st.file_uploader("Tải ảnh chân dung (JPG/PNG)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("👆 Chọn 1 ảnh để bắt đầu.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
with st.container():
    st.image(img, caption="Ảnh đã chọn", width=500)

tmp_path = PROJECT_ROOT / "_tmp_upload.jpg"
img.save(tmp_path)

# ========== INFERENCE ==========
with st.spinner("Đang phân tích..."):
    # truyền tham số từ sidebar, không reload model
    out = pipe.predict_image(str(tmp_path),
                             threshold=thr,
                             blur_cut=blur_cut,
                             force_crop=force_crop)

if not out.get("ok", False):
    st.error(f"❌ Không phát hiện khuôn mặt. Lý do: {out.get('reason','unknown')}")
    st.stop()

# ========== CHỌN KHUÔN MẶT (nếu có nhiều mặt) ==========
selected_idx = out.get("primary_idx", 0)
faces = out.get("faces", [])

if out.get("faces_detected", 1) > 1 and faces:
    st.info(f"Phát hiện **{out['faces_detected']}** khuôn mặt. "
            f"Mặc định chọn mặt **{selected_idx}** theo chiến lược **{out.get('select_strategy')}**.")
    cols = st.columns(min(5, out["faces_detected"]))
    for i, face in enumerate(faces):
        bx = face["box"]
        thumb = img.crop((bx[0], bx[1], bx[2], bx[3])).resize((120, 120))
        with cols[i % len(cols)]:
            st.image(thumb, caption=f"Face {i}", use_container_width=True)
            if st.button(f"Chọn {i}", key=f"pick_{i}"):
                selected_idx = i

    # đồng bộ lại top-level cho UI dưới
    pf = faces[selected_idx]
    for k in ["quality_flags","quality_stats","prob_old","is_old","age_group","pred_age_official","pred_age_all_hint"]:
        out[k] = pf[k]
    out["primary_idx"] = selected_idx

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["✅ Kết quả", "🔎 Chi tiết", "🧪 JSON"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("Xác suất elderly")
        st.markdown(f"<div class='big-score'>{out['prob_old']*100:.1f}%</div>", unsafe_allow_html=True)
        st.caption("Elderly?")
        st.markdown(f"<span class='ok-chip'>Có</span>" if out["is_old"] else f"<span class='bad-chip'>Không</span>", unsafe_allow_html=True)
        st.caption("Nhóm tuổi")
        st.markdown(f"**{out['age_group']}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        pred_official = out["pred_age_official"]          # tuổi khi elderly
        pred_hint = int(round(out["pred_age_all_hint"]))  # gợi ý từ reg_all

        st.caption("Tuổi (elderly)")
        if out["is_old"] and pred_official is not None:
            st.markdown(f"<div class='big-score'>{int(round(pred_official))}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>—</div>", unsafe_allow_html=True)

        if mode == "Chuyên gia":
            st.caption("Gợi ý tuổi (mọi ảnh)")
            st.markdown(f"**{pred_hint}**")

        st.caption("Số khuôn mặt")
        st.markdown(f"**{out.get('faces_detected',1)}**")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========== CONFIDENCE BAR + WARNING ==========
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔒 Độ chắc chắn")
    p = float(out["prob_old"])
    thr_now = float(thr)
    st.progress(int(round(p * 100)))
    st.write(f"**{p*100:.1f}%**  •  Ngưỡng hiện tại: **{thr_now:.2f}**")

    uncertain_margin = 0.07
    if abs(p - thr_now) <= uncertain_margin:
        st.warning("⚠️ Xác suất nằm gần ngưỡng — kết quả có thể chưa chắc chắn.")

    qf = out["quality_flags"]
    low_quality_msgs = []
    if qf.get("blurry"): low_quality_msgs.append("Ảnh mờ (blurry)")
    if qf.get("too_dark"): low_quality_msgs.append("Ảnh tối (too dark)")
    if qf.get("too_bright"): low_quality_msgs.append("Ảnh quá sáng (too bright)")
    if low_quality_msgs:
        st.warning("⚠️ " + " • ".join(low_quality_msgs) + " → nên chọn ảnh rõ, sáng, chính diện hơn.")

    delta = abs(p - thr_now)
    if delta >= 0.25:
        st.success("Mức tin cậy: **Cao**")
    elif delta >= 0.12:
        st.info("Mức tin cậy: **Trung bình**")
    else:
        st.warning("Mức tin cậy: **Thấp**")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("⚠️ Chất lượng ảnh")
    qf = out["quality_flags"]; qs = out["quality_stats"]
    c1, c2 = st.columns(2)
    with c1:
        st.write(
            f"- **Blurry**: `{qf['blurry']}`\n"
            f"- **Too dark**: `{qf['too_dark']}`\n"
            f"- **Too bright**: `{qf['too_bright']}`"
        )
    with c2:
        st.write(
            f"- **Laplacian Var**: `{qs.get('laplacian_var')}`\n"
            f"- **Gray Mean**: `{qs.get('gray_mean')}`"
        )

with tab3:
    st.json(out)

st.success("✅ Xong! Chuyển tab để xem chi tiết hoặc chọn ảnh khác.")
