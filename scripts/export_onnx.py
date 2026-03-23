"""
scripts/export_onnx.py
Exports the CLIP vision + text encoders to ONNX format for production deployment.

After export, use onnxruntime for fast CPU/GPU inference without PyTorch.

Usage:
    uv run python scripts/export_onnx.py
    uv run python scripts/export_onnx.py --checkpoint checkpoints/   # with LoRA
    uv run python scripts/export_onnx.py --quantize                   # + INT8

Then verify:
    uv run python scripts/export_onnx.py --verify
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
OUT_DIR    = Path("onnx_export")


# ─── Wrapper modules for ONNX-friendly export ────────────────────────────────
class VisionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model     = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        out   = self.vision_model(pixel_values=pixel_values)
        embed = self.visual_projection(out.pooler_output)
        return torch.nn.functional.normalize(embed, p=2, dim=-1)


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model     = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask):
        out   = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        embed = self.text_projection(out.pooler_output)
        return torch.nn.functional.normalize(embed, p=2, dim=-1)


# ─── Load model ───────────────────────────────────────────────────────────────
def load_clip(checkpoint=None):
    base = CLIPModel.from_pretrained(MODEL_NAME)
    if checkpoint and Path(checkpoint).exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        print(f"✅ LoRA merged from: {checkpoint}")
    else:
        print("✅ Using base CLIP (no LoRA)")
    base.eval()
    return base


# ─── Export ──────────────────────────────────────────────────────────────────
def export_vision(model, out_dir: Path):
    dummy_img = torch.zeros(1, 3, 224, 224)
    encoder   = VisionEncoder(model).eval()   # ensure eval mode on wrapper

    path = out_dir / "vision_encoder.onnx"
    torch.onnx.export(
        encoder, (dummy_img,),
        str(path),
        export_params = True,
        opset_version = 18,                    # match onnxscript native opset
        input_names   = ["pixel_values"],
        output_names  = ["image_embeds"],
        dynamic_axes  = {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
    )
    print(f"💾 Vision encoder → {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_text(model, out_dir: Path):
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    dummy     = processor(text=["a test caption"], return_tensors="pt", padding=True, truncation=True)
    encoder   = TextEncoder(model).eval()      # ensure eval mode on wrapper

    path = out_dir / "text_encoder.onnx"
    torch.onnx.export(
        encoder,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(path),
        export_params = True,
        opset_version = 18,                    # match onnxscript native opset
        input_names   = ["input_ids", "attention_mask"],
        output_names  = ["text_embeds"],
        dynamic_axes  = {
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "text_embeds":    {0: "batch"},
        },
    )
    print(f"💾 Text encoder  → {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# ─── INT8 Quantization ───────────────────────────────────────────────────────
def quantize_onnx(onnx_path: Path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("⚠️  Install: uv pip install onnxruntime")
        return None

    out_path = onnx_path.with_stem(onnx_path.stem + "_int8")
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    orig_mb  = onnx_path.stat().st_size / 1e6
    quant_mb = out_path.stat().st_size  / 1e6
    print(f"⚡ INT8 quantized → {out_path}  ({orig_mb:.1f} MB → {quant_mb:.1f} MB, "
          f"{(1 - quant_mb/orig_mb)*100:.0f}% smaller)")
    return out_path


# ─── Verify ──────────────────────────────────────────────────────────────────
def verify(out_dir: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  Install: uv pip install onnxruntime")
        return

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Vision
    vision_path = out_dir / "vision_encoder.onnx"
    if vision_path.exists():
        sess  = ort.InferenceSession(str(vision_path))
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        out   = sess.run(None, {"pixel_values": dummy})
        print(f"✅ Vision ONNX OK — output shape: {out[0].shape}")

    # Text
    text_path = out_dir / "text_encoder.onnx"
    if text_path.exists():
        sess   = ort.InferenceSession(str(text_path))
        inputs = processor(text=["a depth pass render"], return_tensors="np", padding=True, truncation=True)
        out    = sess.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
        print(f"✅ Text  ONNX OK — output shape: {out[0].shape}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export CLIP to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint dir (e.g. checkpoints/)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 post-training quantization after export")
    parser.add_argument("--verify",   action="store_true",
                        help="Run inference test on exported ONNX models")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True)

    if args.verify:
        verify(out_dir)
        return

    print(f"\n📦 Exporting CLIP to ONNX → {out_dir}/")
    model = load_clip(args.checkpoint)

    v_path = export_vision(model, out_dir)
    t_path = export_text(model, out_dir)

    if args.quantize:
        print("\n⚡ Applying INT8 quantization...")
        quantize_onnx(v_path)
        quantize_onnx(t_path)

    print(f"\n✅ Export complete!")
    print(f"   Verify:  uv run python scripts/export_onnx.py --verify")
    print(f"   Use in FastAPI: see onnx_export/ for runtime examples")


if __name__ == "__main__":
    main()
