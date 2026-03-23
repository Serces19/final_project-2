"""
app.py — ScopeSearch Streamlit UI
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = Path("vector_store/index.faiss")
PATHS_PATH = Path("vector_store/image_paths.json")

st.set_page_config(
    page_title="ScopeSearch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
.score-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; padding: 2px 8px; border-radius: 99px; font-size: 0.75rem; font-weight: 600;
}
.rank-badge { color: #94a3b8; font-size: 0.72rem; }
.stTextInput > div > div > input {
    background: #1e293b !important; border: 1px solid #6366f1 !important;
    border-radius: 10px !important; color: white !important;
    font-size: 1rem !important; padding: 12px 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load model & index (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⬇️  Loading CLIP model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device

@st.cache_resource(show_spinner="📂 Loading vector store...")
def load_index():
    if not INDEX_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(PATHS_PATH) as f:
        image_paths = json.load(f)
    return index, image_paths

# ─── Embed helpers ────────────────────────────────────────────────────────────
def embed_text(query, model, processor, device):
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embed = model.text_projection(text_out.pooler_output)
        embed = F.normalize(embed, p=2, dim=-1)
    return embed.cpu().numpy().astype(np.float32)

def embed_image(pil_image, model, processor, device):
    pixel_values = processor(images=pil_image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        embed = model.visual_projection(vision_out.pooler_output)
        embed = F.normalize(embed, p=2, dim=-1)
    return embed.cpu().numpy().astype(np.float32)

# ─── Core search ──────────────────────────────────────────────────────────────
def search(embed, index, image_paths, top_k):
    q = embed.copy()
    faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    return [
        {"path": image_paths[idx], "score": float(score)}
        for idx, score in zip(indices[0], distances[0])
        if idx < len(image_paths)
    ]

# ─── Results grid ─────────────────────────────────────────────────────────────
def show_results(results, label):
    st.markdown(f"### Results for *{label}*")
    cols = st.columns(5)
    for i, r in enumerate(results):
        with cols[i % 5]:
            try:
                img = Image.open(r["path"]).convert("RGB")
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<p class="rank-badge">#{i+1} &nbsp;'
                    f'<span class="score-badge">{r["score"]:.3f}</span></p>',
                    unsafe_allow_html=True,
                )
                st.caption(Path(r["path"]).name)
            except Exception:
                st.warning("Could not load image")

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 ScopeSearch</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Semantic image retrieval · CLIP · FAISS · Text & Image queries</p>', unsafe_allow_html=True)

model, processor, device = load_model()
index, image_paths = load_index()

if index is None:
    st.error("❌ Vector store not found. Run `index_images.py` first.", icon="🚫")
    st.stop()

st.markdown(f"**{index.ntotal:,}** images indexed · device: `{device}`")

tab_text, tab_image = st.tabs(["📝 Text Search", "🖼️ Image Search"])

# ── Tab 1: Text → Image ───────────────────────────────────────────────────────
with tab_text:
    col_q, col_k = st.columns([5, 1])
    with col_q:
        query = st.text_input("", placeholder="✏️  Describe an image... e.g. 'a dog running on a beach'",
                              label_visibility="collapsed", key="text_query")
    with col_k:
        top_k_text = st.selectbox("Results", [5, 10, 20], index=1, key="top_k_text")

    if query:
        with st.spinner("🔎 Searching..."):
            embed = embed_text(query, model, processor, device)
            results = search(embed, index, image_paths, top_k_text)
        show_results(results, f'"{query}"')
    else:
        st.markdown("""<div style="text-align:center;padding:60px 0;color:#475569;">
            <div style="font-size:3rem;">📝</div>
            <p style="margin-top:12px;">Type a description above to search.</p>
        </div>""", unsafe_allow_html=True)

# ── Tab 2: Image → Image ──────────────────────────────────────────────────────
with tab_image:
    col_up, col_k2 = st.columns([5, 1])
    with col_up:
        uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                    label_visibility="visible")
    with col_k2:
        top_k_img = st.selectbox("Results", [5, 10, 20], index=1, key="top_k_img")

    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        col_preview, col_results = st.columns([1, 4])
        with col_preview:
            st.image(query_img, caption="Query image", use_container_width=True)
        with col_results:
            with st.spinner("🔎 Finding similar images..."):
                embed = embed_image(query_img, model, processor, device)
                results = search(embed, index, image_paths, top_k_img)
            show_results(results, "uploaded image")
    else:
        st.markdown("""<div style="text-align:center;padding:60px 0;color:#475569;">
            <div style="font-size:3rem;">🖼️</div>
            <p style="margin-top:12px;">Upload an image to find visually similar assets.</p>
        </div>""", unsafe_allow_html=True)
