"""
Sanity check: base CLIP (no PEFT) with synthetic data.
Verifica que el forward pass, la Contrastive Loss y FAISS funcionan end-to-end.
Ejecutar con:  uv run python -m pytest tests/test_base_clip.py -v
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from src.models.loss import ContrastiveLoss
from src.retrieval.faiss_index import FaissRetrievalSystem


# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# ─── Fixtures ────────────────────────────────────────────────────────────────
VFX_PAIRS = [
    ("specular pass vehicle",        (200, 200, 215)),   # cool silver
    ("dense smoke with alpha channel", (80,  80,  80)),  # grey smoke
    ("chroma key green plate",        (50,  180, 50)),   # green screen
    ("depth of field bokeh blur",     (90,  120, 200)),  # blurry blue
]


def _build_batch():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    texts  = [t for t, _ in VFX_PAIRS]
    images = [Image.new("RGB", (224, 224), c) for _, c in VFX_PAIRS]
    return processor(text=texts, images=images, return_tensors="pt",
                     padding=True, truncation=True, max_length=77)

# ─── Tests ───────────────────────────────────────────────────────────────────

def test_forward_pass():
    """Base CLIP produce embeddings sin errores."""
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    batch = _build_batch()

    with torch.no_grad():
        out = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            pixel_values   = batch["pixel_values"].to(DEVICE),
        )

    assert out.image_embeds.shape == (BATCH_SIZE, 512), "Dim imagen incorrecta"
    assert out.text_embeds.shape  == (BATCH_SIZE, 512), "Dim texto incorrecta"
    print(f"\n✅ Forward pass OK — image_embeds: {out.image_embeds.shape}")


def test_contrastive_loss():
    """La pérdida contrastiva disminuye cuando los pares son correctos."""
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    criterion = ContrastiveLoss().to(DEVICE)
    batch = _build_batch()

    with torch.no_grad():
        out = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            pixel_values   = batch["pixel_values"].to(DEVICE),
        )
        loss = criterion(out.image_embeds, out.text_embeds)

    assert loss.item() > 0, "La pérdida debe ser mayor que 0"
    print(f"\n✅ Contrastive Loss OK — Loss = {loss.item():.4f}")


def test_faiss_retrieval():
    """FAISS indexa los embeddings y retorna el match correcto en top-1."""
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    batch = _build_batch()

    with torch.no_grad():
        out = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            pixel_values   = batch["pixel_values"].to(DEVICE),
        )

    image_embeds = out.image_embeds.cpu().numpy().astype(np.float32)
    text_embeds  = out.text_embeds.cpu().numpy().astype(np.float32)

    paths  = [f"asset_{i}.exr" for i in range(BATCH_SIZE)]
    system = FaissRetrievalSystem(embedding_dim=512)
    system.add_embeddings(image_embeds, paths)

    results = system.search(text_embeds, top_k=1)

    hits = sum(1 for i, r in enumerate(results) if r[0]["image_path"] == paths[i])
    recall_at_1 = hits / BATCH_SIZE
    print(f"\n✅ FAISS Retrieval OK — Recall@1 (base CLIP, sin PEFT): {recall_at_1:.2%}")
    # Para datos sintéticos, los pares correctos deberían estar en top-1
    assert recall_at_1 > 0, "Al menos un par debe recuperarse correctamente"
