"""
scripts/download_coco_val.py
Downloads COCO 2017 validation set (~1GB, 5k images) and generates the metadata CSV.
Usage:  uv run python scripts/download_coco_val.py
"""
import os
import json
import zipfile
import urllib.request
from pathlib import Path

COCO_VAL_URL  = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DATA_DIR      = Path("data/raw/coco")
CSV_OUT       = Path("data/processed/coco_val.csv")

def download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / Path(url).name
    if zip_path.exists():
        print(f"✅ Already downloaded: {zip_path}")
    else:
        print(f"⬇️  Downloading {url}  →  {zip_path}")
        print("    (may take a few minutes on first run...)")
        urllib.request.urlretrieve(url, zip_path,
            reporthook=lambda b, bs, t: print(f"\r    {b*bs/1e6:.1f}/{t/1e6:.1f} MB", end="", flush=True))
        print()
    print(f"📦 Extracting {zip_path}  →  {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print("   Done.\n")

def build_csv():
    ann_file = DATA_DIR / "annotations" / "captions_val2017.json"
    img_dir  = DATA_DIR / "val2017"

    print(f"📝 Building metadata CSV from {ann_file}")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id → filename map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Take first caption per image (5 exist per image, we use index 0)
    seen = set()
    rows = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        fname = id_to_file[img_id]
        path  = str(img_dir / fname)
        rows.append(f'"{path}","{ann["caption"].strip()}"\n')

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8") as f:
        f.write("image_path,description\n")
        f.writelines(rows)

    print(f"✅ CSV saved: {CSV_OUT}  ({len(rows)} pairs)\n")

if __name__ == "__main__":
    download(COCO_VAL_URL,  DATA_DIR)
    download(COCO_ANN_URL,  DATA_DIR)
    build_csv()
    print("🎉 Done! Run next:")
    print("   uv run python scripts/index_images.py --metadata data/processed/coco_val.csv")
