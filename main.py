import argparse
import torch
from pathlib import Path
from src.data.dataset import create_dataloaders
from src.models.clip_lora import get_clip_lora
from src.models.loss import ContrastiveLoss
from src.engine.train import train_model
from src.engine.evaluate import evaluate_model


def load_model(args, device):
    """Load base CLIP+LoRA or fine-tuned checkpoint depending on args."""
    model = get_clip_lora(r=args.lora_r, lora_alpha=args.lora_alpha)

    ckpt = Path(args.checkpoint) if args.checkpoint else Path("checkpoints")
    if not args.base_only and ckpt.exists() and (ckpt / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model.get_base_model(), str(ckpt))
        print(f"✅ Loaded fine-tuned checkpoint: {ckpt}")
    else:
        if args.base_only:
            print("🧠 Running with BASE CLIP (LoRA initialized, no checkpoint loaded)")
        else:
            print("⚠️  No checkpoint found — using base CLIP (run training first)")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="ScopeSearch: VFX Asset Semantic Retrieval via PEFT CLIP")
    parser.add_argument("--mode",       type=str, choices=["train", "evaluate"], required=True)
    parser.add_argument("--metadata",   type=str, required=True, help="Path to train/eval CSV")
    parser.add_argument("--val",        type=str, default=None,  help="Path to val CSV (for training)")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   type=str, default="checkpoints/")
    parser.add_argument("--lora_r",     type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint dir. Defaults to 'checkpoints/' if it exists.")
    parser.add_argument("--base_only",  action="store_true",
                        help="Evaluate base CLIP without loading checkpoint (for comparison baseline)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    print("⬇️  Loading model...")
    model = load_model(args, device)

    if args.mode == "train":
        print(f"📂 Train: {args.metadata}")
        train_loader = create_dataloaders(args.metadata, batch_size=args.batch_size, shuffle=True)

        val_loader = None
        if args.val:
            print(f"📂 Val:   {args.val}")
            val_loader = create_dataloaders(args.val, batch_size=args.batch_size, shuffle=False)

        criterion = ContrastiveLoss()
        print(f"\n🚀 Training — {args.epochs} epochs, lr={args.lr}")
        model = train_model(
            model, train_loader, val_loader,
            criterion=criterion,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        print(f"\n💾 LoRA weights saved: {save_dir}")
        print("🎉 Training complete!")
        print(f"\nNext steps:")
        print(f"  # Compare base vs fine-tuned:")
        print(f"  uv run python main.py --mode evaluate --metadata {args.val or args.metadata} --base_only")
        print(f"  uv run python main.py --mode evaluate --metadata {args.val or args.metadata}")

    elif args.mode == "evaluate":
        print(f"📊 Evaluating on: {args.metadata}")
        val_loader = create_dataloaders(args.metadata, batch_size=args.batch_size, shuffle=False)
        evaluate_model(model, val_loader, device=device)


if __name__ == "__main__":
    main()
