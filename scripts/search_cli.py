"""
scripts/search_cli.py
Interactive text search against the FAISS vector store.
No training required — uses the pre-built index from index_images.py.

Usage (one-shot):
    uv run python scripts/search_cli.py --query "smoke with alpha channel" --top_k 5

Usage (interactive REPL):
    uv run python scripts/search_cli.py
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = Path("vector_store/index.faiss")
PATHS_PATH = Path("vector_store/image_paths.json")


def load_artifacts():
    if not INDEX_PATH.exists() or not PATHS_PATH.exists():
        raise FileNotFoundError(
            "❌ Vector store not found. Run first:\n"
            "   uv run python scripts/index_images.py --metadata <your_csv>"
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(PATHS_PATH) as f:
        image_paths = json.load(f)
    return index, image_paths


def embed_text(query, model, processor, device):
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        embed = model.text_projection(text_out.pooler_output)
        embed = F.normalize(embed, p=2, dim=-1)
    return embed.cpu().numpy().astype(np.float32)


def embed_image(image_path_or_pil, model, processor, device):
    """Encode a local image path or a PIL Image into a CLIP embedding."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path_or_pil).convert("RGB") if isinstance(image_path_or_pil, (str, Path)) else image_path_or_pil
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        embed = model.visual_projection(vision_out.pooler_output)
        embed = F.normalize(embed, p=2, dim=-1)
    return embed.cpu().numpy().astype(np.float32)


def search(embed, index, image_paths, top_k=5):
    """Generic search: accepts a pre-computed embedding (text or image)."""
    faiss.normalize_L2(embed)
    distances, indices = index.search(embed, top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < len(image_paths):
            results.append({
                "rank": rank,
                "score": float(score),
                "path": image_paths[idx]
            })
    return results


def print_results(query, results):
    print(f"\n🔍 Query: \"{query}\"")
    print("─" * 60)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        print(f"  #{r['rank']}  [{bar:<20}]  {r['score']:.4f}  →  {r['path']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ScopeSearch: Text/Image → Image retrieval")
    parser.add_argument("--query",      type=str, help="Text query (text-to-image mode)")
    parser.add_argument("--image_path", type=str, help="Path to query image (image-to-image mode)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    print(f"⬇️  Loading model: {MODEL_NAME}")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("📂 Loading vector store...")
    index, image_paths = load_artifacts()
    print(f"✅ Index loaded — {index.ntotal} images indexed\n")

    if args.image_path:
        print(f"🖼️  Query image: {args.image_path}")
        embed = embed_image(args.image_path, model, processor, device)
        results = search(embed, index, image_paths, args.top_k)
        print_results(f"[image] {args.image_path}", results)

    elif args.query:
        embed = embed_text(args.query, model, processor, device)
        results = search(embed, index, image_paths, args.top_k)
        print_results(args.query, results)

    else:
        print("💬 Interactive mode — type a query, or 'img:<path>' for image search. Ctrl+C to exit.\n")
        while True:
            try:
                raw = input("🔎 Query > ").strip()
                if not raw:
                    continue
                if raw.startswith("img:"):
                    path = raw[4:].strip()
                    embed = embed_image(path, model, processor, device)
                    label = f"[image] {path}"
                else:
                    embed = embed_text(raw, model, processor, device)
                    label = raw
                results = search(embed, index, image_paths, args.top_k)
                print_results(label, results)
            except KeyboardInterrupt:
                print("\n\n👋 Exiting.")
                break


if __name__ == "__main__":
    main()
