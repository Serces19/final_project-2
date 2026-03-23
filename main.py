import argparse
import torch
from pathlib import Path
from src.data.dataset import create_dataloaders
from src.models.clip_lora import get_clip_lora
from src.models.loss import ContrastiveLoss
from src.engine.train import train_model
from src.engine.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="ScopeSearch: VFX Asset Semantic Retrieval via PEFT CLIP")
    parser.add_argument("--mode",       type=str, choices=["train", "evaluate"], required=True)
    parser.add_argument("--metadata",   type=str, required=True, help="Path to train CSV")
    parser.add_argument("--val",        type=str, default=None,  help="Path to val CSV (optional)")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   type=str, default="checkpoints/", help="Where to save LoRA weights")
    parser.add_argument("--lora_r",     type=int, default=8,  help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    print("⬇️  Loading CLIP + LoRA model...")
    model = get_clip_lora(r=args.lora_r, lora_alpha=args.lora_alpha).to(device)

    if args.mode == "train":
        print(f"📂 Loading train dataset: {args.metadata}")
        train_loader = create_dataloaders(args.metadata, batch_size=args.batch_size, shuffle=True)

        val_loader = None
        if args.val:
            print(f"📂 Loading val dataset:   {args.val}")
            val_loader = create_dataloaders(args.val, batch_size=args.batch_size, shuffle=False)

        criterion = ContrastiveLoss()

        print(f"\n🚀 Starting training — {args.epochs} epochs, lr={args.lr}")
        model = train_model(
            model, train_loader, val_loader,
            criterion=criterion,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device
        )

        # Save LoRA weights
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        print(f"\n💾 LoRA weights saved to: {save_dir}")
        print("🎉 Training complete!")
        print(f"\nNext — re-index with the fine-tuned model and launch the UI:")
        print(f"  uv run python scripts/index_images.py --metadata {args.val or args.metadata}")
        print(f"  streamlit run app.py")

    elif args.mode == "evaluate":
        print(f"📊 Evaluating on: {args.metadata}")
        val_loader = create_dataloaders(args.metadata, batch_size=args.batch_size, shuffle=False)
        evaluate_model(model, val_loader, device=device)


if __name__ == "__main__":
    main()
