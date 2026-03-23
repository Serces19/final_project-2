"""
scripts/validate_dataset.py
Validates a VFX dataset CSV before training.
Checks: file existence, image readability, description length, class balance.

Usage:
    uv run python scripts/validate_dataset.py --csv data/processed/vfx_dataset.csv
"""
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm


def validate(csv_path: str):
    print(f"\n📋 Validating: {csv_path}\n{'─'*50}")
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # ── Basic checks ──────────────────────────────────────
    assert "image_path"  in df.columns, "❌ Missing column: 'image_path'"
    assert "description" in df.columns, "❌ Missing column: 'description'"
    print(f"✅ Columns OK")
    print(f"   Total rows : {len(df)}")

    # ── Duplicate check ───────────────────────────────────
    dupes = df["image_path"].duplicated().sum()
    print(f"   Duplicates : {dupes} {'⚠️ ' if dupes else '✅'}")

    # ── Image validation ──────────────────────────────────
    print(f"\n🖼️  Checking images...")
    missing, corrupt, ok = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Scanning"):
        p = Path(row["image_path"])
        if not p.exists():
            missing.append(str(p))
        else:
            try:
                Image.open(p).verify()
                ok.append(str(p))
            except Exception:
                corrupt.append(str(p))

    print(f"   ✅ Valid   : {len(ok)}")
    if missing:
        print(f"   ❌ Missing : {len(missing)}")
        for m in missing[:5]:
            print(f"      {m}")
        if len(missing) > 5:
            print(f"      ... and {len(missing)-5} more")
    if corrupt:
        print(f"   ⚠️  Corrupt : {len(corrupt)}")

    # ── Description stats ─────────────────────────────────
    df["desc_len"] = df["description"].astype(str).str.split().str.len()
    print(f"\n📝 Description stats:")
    print(f"   Words avg  : {df['desc_len'].mean():.1f}")
    print(f"   Words min  : {df['desc_len'].min()}")
    print(f"   Words max  : {df['desc_len'].max()}")
    short = (df["desc_len"] < 5).sum()
    if short:
        print(f"   ⚠️  Too short (<5 words): {short} rows")

    # ── Category distribution (inferred from parent folder) ───
    df["category"] = df["image_path"].apply(lambda p: Path(p).parent.name)
    cats = Counter(df["category"])
    print(f"\n📊 Category distribution ({len(cats)} categories):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        print(f"   {cat:<20} {bar} {count}")

    # ── Final verdict ─────────────────────────────────────
    print(f"\n{'─'*50}")
    ready = len(ok)
    if ready == 0:
        print("❌ No valid images found. Check your paths.")
    elif ready < 100:
        print(f"⚠️  Only {ready} valid pairs. Minimum recommended: 100. Ideal: 300+")
    elif ready < 300:
        print(f"✅ {ready} valid pairs — workable for a first fine-tuning run.")
    else:
        print(f"✅ {ready} valid pairs — good dataset size for LoRA fine-tuning!")

    if len(cats) < 3:
        print("⚠️  Only {len(cats)} categories detected. More diversity helps generalization.")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VFX dataset CSV before training")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    args = parser.parse_args()
    validate(args.csv)
