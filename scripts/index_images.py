"""
scripts/index_images.py
Encodes all images in a metadata CSV with base CLIP (no training required)
and saves a FAISS index for fast text search.

Usage:
    uv run python scripts/index_images.py --metadata data/processed/coco_val.csv
    uv run python scripts/index_images.py --image_dir data/raw/my_vfx_assets/  (no CSV needed)
"""
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME  = "openai/clip-vit-base-patch32"
EMBED_DIM   = 512
BATCH_SIZE  = 64  # increase on GPU
INDEX_OUT   = Path("vector_store/index.faiss")
PATHS_OUT   = Path("vector_store/image_paths.json")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_images_from_csv(metadata_path):
    import pandas as pd
    df = pd.read_csv(metadata_path, on_bad_lines="skip")
    return df["image_path"].tolist()

def load_images_from_dir(image_dir):
    return [str(p) for p in Path(image_dir).rglob("*") if p.suffix.lower() in SUPPORTED]

def encode_images(paths, model, processor, device, batch_size):
    all_embeds = []
    valid_paths = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_paths = paths[i:i + batch_size]
        images = []
        batch_valid = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                batch_valid.append(p)
            except Exception as e:
                print(f"\n⚠️  Skipping {p}: {e}")

        if not images:
            continue

        pixel_values = processor(images=images, return_tensors="pt").to(device)["pixel_values"]
        with torch.no_grad():
            # Explicit path: works across all transformers versions
            vision_out = model.vision_model(pixel_values=pixel_values)
            embeds = model.visual_projection(vision_out.pooler_output)
            embeds = F.normalize(embeds, p=2, dim=-1)

        all_embeds.append(embeds.cpu().numpy().astype(np.float32))
        valid_paths.extend(batch_valid)

    return np.concatenate(all_embeds, axis=0), valid_paths


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from images")
    parser.add_argument("--metadata",  type=str, help="Path to CSV with 'image_path' column")
    parser.add_argument("--image_dir", type=str, help="Alternative: folder of images (no CSV required)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not args.metadata and not args.image_dir:
        parser.error("Provide --metadata or --image_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    print(f"⬇️  Loading model: {MODEL_NAME}")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    paths = load_images_from_csv(args.metadata) if args.metadata else load_images_from_dir(args.image_dir)
    print(f"🖼️  Found {len(paths)} images to index")

    embeddings, valid_paths = encode_images(paths, model, processor, device, args.batch_size)
    print(f"✅ Encoded {len(valid_paths)} images  ({embeddings.shape})")

    # Build and save FAISS index
    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_OUT))
    print(f"💾 FAISS index saved: {INDEX_OUT}  ({index.ntotal} vectors)")

    with open(PATHS_OUT, "w") as f:
        json.dump(valid_paths, f)
    print(f"💾 Image paths saved: {PATHS_OUT}")

    print("\n🎉 Indexing complete! Run search with:")
    print('   uv run python scripts/search_cli.py --query "your text query"')


if __name__ == "__main__":
    main()
