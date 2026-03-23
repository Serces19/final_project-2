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
        # Explicit path: works across all transformers versions
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        embed = model.text_projection(text_out.pooler_output)
        embed = F.normalize(embed, p=2, dim=-1)
    return embed.cpu().numpy().astype(np.float32)


def search(query, index, image_paths, model, processor, device, top_k=5):
    embed = embed_text(query, model, processor, device)
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
    parser = argparse.ArgumentParser(description="ScopeSearch: Text → Image retrieval")
    parser.add_argument("--query", type=str, help="Text query. If omitted, enters interactive mode.")
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

    if args.query:
        # One-shot mode
        results = search(args.query, index, image_paths, model, processor, device, args.top_k)
        print_results(args.query, results)
    else:
        # Interactive REPL
        print("💬 Interactive mode — type a query and press ENTER (Ctrl+C to exit)\n")
        while True:
            try:
                query = input("🔎 Query > ").strip()
                if not query:
                    continue
                results = search(query, index, image_paths, model, processor, device, args.top_k)
                print_results(query, results)
            except KeyboardInterrupt:
                print("\n\n👋 Exiting.")
                break


if __name__ == "__main__":
    main()
