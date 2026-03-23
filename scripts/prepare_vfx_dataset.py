"""
scripts/prepare_vfx_dataset.py

Downloads open-source VFX footage (Sintel / Tears of Steel frames via official Blender CDN)
and builds a labeled CSV with VFX-specific captions for CLIP LoRA fine-tuning.

Usage:
  # 1. Download Sintel frames (~300 MB for a representative subset)
  uv run python scripts/prepare_vfx_dataset.py --source sintel --output data/raw/sintel/

  # 2. Build the VFX caption CSV (pass --build_csv after download)
  uv run python scripts/prepare_vfx_dataset.py --source sintel --output data/raw/sintel/ --build_csv

  # 3. Split into train/val
  uv run python scripts/prepare_vfx_dataset.py --split --input data/processed/vfx_dataset.csv
"""
import argparse
import random
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─── Sintel shots metadata ────────────────────────────────────────────────────
# Each entry describes a shot from the Blender Sintel open movie.
# Source: https://durian.blender.org/
# We use the official mirror for individual frame downloads.
SINTEL_BASE = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete"

# Shot metadata: (sequence_name, start_frame, end_frame, description_template)
SINTEL_SHOTS = [
    ("cave_2",       1,  50, "CGI character in dark cave, cinematic lighting, subsurface scattering skin shader"),
    ("market_2",     1,  50, "fantasy market scene, outdoor CGI environment, warm ambient occlusion"),
    ("mountain_1",   1,  50, "aerial mountain landscape, volumetric fog, depth of field render"),
    ("temple_2",     1,  50, "ancient temple interior, dramatic rim lighting, specular stone surfaces"),
    ("bamboo_1",     1,  50, "bamboo forest environment, translucent leaf shaders, soft shadows"),
    ("alley_1",      1,  30, "narrow alley VFX composite, motion blur, wet surface reflections"),
    ("cave_4",       1,  30, "cave with dragon, fire VFX, volumetric smoke particles"),
    ("market_5",     1,  30, "crowded market crowd simulation, depth of field, color grading"),
    ("temple_3",     1,  30, "temple destruction, particle simulation, debris dust FX"),
    ("ambushfight",  1,  30, "action fight sequence, motion blur, cinematic depth of field"),
]

# Fine-grained VFX caption templates applied per-frame with variation
CAPTION_TEMPLATES = [
    "{base_desc}, frame {frame:04d}",
    "{base_desc}, single CG render frame",
    "CGI animation frame: {base_desc}",
    "VFX production render, {base_desc}",
    "{base_desc}, rendered with ray tracing",
    "3D animation frame showing {base_desc}",
    "Blender Sintel open movie: {base_desc}",
    "Cinematic CGI: {base_desc}",
]

# ─── Download ─────────────────────────────────────────────────────────────────
def download_sintel_frames(output_dir: Path, max_per_shot: int = 30):
    """Download a curated subset of Sintel frames from the official dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for (shot, start, end, _) in SINTEL_SHOTS:
        shot_dir = output_dir / shot
        shot_dir.mkdir(exist_ok=True)
        frames = range(start, min(end + 1, start + max_per_shot))
        print(f"\n📂 Shot: {shot} ({len(list(frames))} frames)")

        for frame in tqdm(frames, desc=f"  {shot}"):
            fname   = f"frame_{frame:04d}.png"
            out_path = shot_dir / fname
            if out_path.exists():
                downloaded.append(str(out_path))
                continue

            url = f"{SINTEL_BASE}/training/final/{shot}/{fname}"
            try:
                urllib.request.urlretrieve(url, out_path)
                downloaded.append(str(out_path))
            except Exception as e:
                print(f"\n  ⚠️  Could not download {url}: {e}")

    print(f"\n✅ Downloaded {len(downloaded)} frames to {output_dir}")
    return downloaded


# ─── Caption generation ───────────────────────────────────────────────────────
def generate_captions(output_dir: Path):
    """Build image-caption pairs from downloaded Sintel frames."""
    rows = []
    shot_meta = {s[0]: s[3] for s in SINTEL_SHOTS}  # name → description

    for shot_dir in sorted(output_dir.iterdir()):
        if not shot_dir.is_dir():
            continue
        shot_name = shot_dir.name
        base_desc = shot_meta.get(shot_name, f"CGI animation, {shot_name} sequence")

        for img_path in sorted(shot_dir.glob("*.png")):
            try:
                # Quick validation
                Image.open(img_path).verify()
            except Exception:
                continue

            frame_num = int(img_path.stem.split("_")[-1])
            template  = random.choice(CAPTION_TEMPLATES)
            caption   = template.format(base_desc=base_desc, frame=frame_num)

            rows.append({"image_path": str(img_path), "description": caption})

    return rows


# ─── Train / Val split ────────────────────────────────────────────────────────
def split_dataset(input_csv: Path, val_ratio: float = 0.2):
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_ratio))
    train_df  = df[:split_idx]
    val_df    = df[split_idx:]

    train_path = input_csv.parent / "vfx_train.csv"
    val_path   = input_csv.parent / "vfx_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"✅ Train: {len(train_df)} pairs  →  {train_path}")
    print(f"✅ Val:   {len(val_df)} pairs    →  {val_path}")
    return train_path, val_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare VFX dataset for CLIP LoRA fine-tuning")
    parser.add_argument("--source",    type=str, choices=["sintel"], default="sintel")
    parser.add_argument("--output",    type=str, default="data/raw/sintel/")
    parser.add_argument("--build_csv", action="store_true", help="Generate caption CSV after download")
    parser.add_argument("--split",     action="store_true", help="Split existing CSV into train/val")
    parser.add_argument("--input",     type=str, default="data/processed/vfx_dataset.csv")
    parser.add_argument("--max_per_shot", type=int, default=30, help="Max frames per shot (default 30)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    if args.split:
        split_dataset(Path(args.input), args.val_ratio)
        return

    output_dir = Path(args.output)

    print(f"⬇️  Downloading Sintel frames to {output_dir}/ ...")
    download_sintel_frames(output_dir, max_per_shot=args.max_per_shot)

    if args.build_csv:
        print("\n📝 Generating captions...")
        rows = generate_captions(output_dir)

        csv_out = Path("data/processed/vfx_dataset.csv")
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"✅ Dataset CSV: {csv_out}  ({len(rows)} pairs)")

        print("\n🎉 Done! Next steps:")
        print(f"   uv run python scripts/prepare_vfx_dataset.py --split --input {csv_out}")
        print("   uv run python main.py --mode train --metadata data/processed/vfx_train.csv --epochs 10")


if __name__ == "__main__":
    main()
