"""
scripts/index_images.py
Encodes all images using CLIP (base or LoRA fine-tuned) and saves a FAISS index.

Usage — base CLIP:
    uv run python scripts/index_images.py --metadata data/processed/coco_val.csv

Usage — with fine-tuned LoRA checkpoint:
    uv run python scripts/index_images.py \
        --metadata data/processed/vfx_val.csv \
        --checkpoint checkpoints/
"""
import argparse
import json
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
BATCH_SIZE  = 64
INDEX_OUT   = Path("vector_store/index.faiss")
PATHS_OUT   = Path("vector_store/image_paths.json")
META_OUT    = Path("vector_store/meta.json")   # stores which model was used

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_model(checkpoint=None, device="cuda"):
    """Load base CLIP or LoRA fine-tuned model."""
    base = CLIPModel.from_pretrained(MODEL_NAME)
    if checkpoint and Path(checkpoint).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()  # fuse LoRA into weights for fast inference
        kind = f"CLIP + LoRA ({checkpoint})"
    else:
        model = base
        kind = "CLIP base (no fine-tuning)"
    print(f"✅ Model loaded: {kind}")
    return model.to(device).eval(), kind


def load_images_from_csv(metadata_path):
    import pandas as pd
    df = pd.read_csv(metadata_path, on_bad_lines="skip")
    return df["image_path"].tolist()

def load_images_from_dir(image_dir):
    return [str(p) for p in Path(image_dir).rglob("*") if p.suffix.lower() in SUPPORTED]

def encode_images(paths, model, processor, device, batch_size, fp16=False):
    all_embeds = []
    valid_paths = []
    dtype = torch.float16 if fp16 else torch.float32

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

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], enabled=fp16):
            vision_out = model.vision_model(pixel_values=pixel_values)
            embeds = model.visual_projection(vision_out.pooler_output)
            embeds = F.normalize(embeds.float(), p=2, dim=-1)

        all_embeds.append(embeds.cpu().numpy().astype(np.float32))
        valid_paths.extend(batch_valid)

    return np.concatenate(all_embeds, axis=0), valid_paths


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from images")
    parser.add_argument("--metadata",   type=str, help="Path to CSV with 'image_path' column")
    parser.add_argument("--image_dir",  type=str, help="Alt: folder of images (no CSV required)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint dir (e.g. checkpoints/). Omit for base CLIP.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 half-precision for 2x speedup on GPU (requires CUDA)")
    args = parser.parse_args()

    if not args.metadata and not args.image_dir:
        parser.error("Provide --metadata or --image_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fp16 and device == "cpu":
        print("⚠️  --fp16 ignored on CPU — requires CUDA")
        args.fp16 = False
    print(f"🖥️  Device: {device}{'  [FP16]' if args.fp16 else ''}")

    model, model_kind = load_model(args.checkpoint, device)
    if args.fp16:
        model = model.half()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    paths = load_images_from_csv(args.metadata) if args.metadata else load_images_from_dir(args.image_dir)
    print(f"🖼️  Found {len(paths)} images to index")

    embeddings, valid_paths = encode_images(paths, model, processor, device, args.batch_size, args.fp16)
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

    # Save metadata so app.py knows which model was used
    with open(META_OUT, "w") as f:
        import json as _json
        _json.dump({"model": model_kind, "checkpoint": args.checkpoint}, f)

    print(f"\n🎉 Indexing complete! Model used: {model_kind}")
    print('   uv run python scripts/search_cli.py --query "your query"')
    if args.checkpoint:
        print(f'   (or pass --checkpoint {args.checkpoint} to search_cli.py)')


if __name__ == "__main__":
    main()
