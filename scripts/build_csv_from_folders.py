"""
scripts/build_csv_from_folders.py

Scans your VFX image folders and auto-generates a labeled CSV
with category-specific captions. Supports augmentation via multiple
caption templates per image.

Usage:
    uv run python scripts/build_csv_from_folders.py \
        --root data/raw/vfx_assets/ \
        --output data/processed/vfx_dataset.csv \
        --augment 3   # captions per image (1–5 recommended)

Then split for training:
    uv run python scripts/prepare_vfx_dataset.py \
        --split --input data/processed/vfx_dataset.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

# ─── Category captions ────────────────────────────────────────────────────────
# Add/edit templates to match your actual content.
CATEGORY_CAPTIONS = {

    "assets": [
        "VFX asset render, isolated element with alpha channel",
        "visual effects element, compositing asset, black background",
        "FX asset: particle or explosion effect with transparency",
        "VFX stock footage element, pre-keyed asset for compositing",
        "rendered VFX element, dark background, alpha channel included",
        "motion graphics VFX asset, isolated effect on black",
        "practical or CGI VFX element ready for compositing",
    ],

    "chroma": [
        "green screen chroma key plate, talent on green background",
        "chroma key footage, even green lighting, subject isolated",
        "green screen shot, VFX plate for keying and compositing",
        "chroma key plate showing person against green background",
        "green screen interview or performance shot, clean key setup",
        "VFX chroma plate, green background, studio lighting setup",
        "keying plate with green screen, ready for rotoscope or key",
    ],

    "depth": [
        "linear Z-depth pass, grayscale render, near white far black",
        "depth AOV, per-pixel distance from camera encoded in grey",
        "depth of field pass, depth render for blur in compositing",
        "Z-depth render pass, black to white gradient from near to far",
        "depth map render, grayscale depth information, CG render pass",
        "3D render depth pass, camera depth buffer, compositing AOV",
        "depth channel render, grayscale depth map, white foreground",
    ],

    "normal": [
        "surface normal map, RGB encoded normals, blue dominant pass",
        "normal map render, XYZ surface directions encoded as color",
        "normal AOV render pass, colored normal vectors, CGI surface",
        "tangent space normal map, blue-purple tones, surface detail",
        "3D render normal pass, RGB normal direction, VFX compositing",
        "world or tangent space normals render, colorful surface map",
        "normal render pass showing surface orientation as RGB color",
    ],

    # Fallback for any unknown folder name
    "_default": [
        "VFX image, visual effects production asset or render pass",
        "compositing element, VFX production, render or plate",
        "VFX production image, visual effects pipeline asset",
    ],
}

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def get_captions(folder_name: str, n: int) -> list[str]:
    """Pick n unique captions for a folder, falling back to _default."""
    pool = CATEGORY_CAPTIONS.get(folder_name.lower(),
                                  CATEGORY_CAPTIONS["_default"])
    if n >= len(pool):
        return pool
    return random.sample(pool, n)


def build_csv(root: Path, output: Path, augment: int = 2):
    rows = []
    folders = [f for f in sorted(root.iterdir()) if f.is_dir()]

    if not folders:
        print(f"❌ No subfolders found in {root}")
        return

    for folder in folders:
        images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED]
        if not images:
            print(f"⚠️  No images in {folder.name}/ — skipping")
            continue

        captions_pool = CATEGORY_CAPTIONS.get(folder.name.lower(),
                                               CATEGORY_CAPTIONS["_default"])
        print(f"  📂 {folder.name:<12} → {len(images)} images, "
              f"{len(captions_pool)} caption templates")

        for img_path in images:
            # Pick `augment` distinct captions per image
            chosen = random.sample(captions_pool, min(augment, len(captions_pool)))
            for caption in chosen:
                rows.append({
                    "image_path": str(img_path),
                    "description": caption
                })

    if not rows:
        print("❌ No rows generated — check your folder paths and image files.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output, index=False)

    print(f"\n✅ Dataset CSV saved: {output}")
    print(f"   Total pairs : {len(df)}")
    print(f"   Images      : {df['image_path'].nunique()}")
    print(f"   Categories  : {df['image_path'].apply(lambda p: Path(p).parent.name).nunique()}")
    print(f"\nNext steps:")
    print(f"  uv run python scripts/validate_dataset.py --csv {output}")
    print(f"  uv run python scripts/prepare_vfx_dataset.py --split --input {output}")


def main():
    parser = argparse.ArgumentParser(description="Auto-build VFX dataset CSV from folders")
    parser.add_argument("--root",    type=str, default="data/raw/vfx_assets/",
                        help="Root folder containing category subfolders")
    parser.add_argument("--output",  type=str, default="data/processed/vfx_dataset.csv")
    parser.add_argument("--augment", type=int, default=2,
                        help="Number of captions per image (default: 2, max: 7)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ Folder not found: {root}")
        return

    print(f"\n🔍 Scanning {root}/ ...")
    build_csv(root, Path(args.output), augment=args.augment)


if __name__ == "__main__":
    main()
