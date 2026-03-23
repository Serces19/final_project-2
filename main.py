import argparse
import torch
from pathlib import Path
from src.data.dataset import create_dataloaders
from src.models.clip_lora import get_clip_lora
from src.models.loss import ContrastiveLoss
from src.engine.train import train_model
from src.engine.evaluate import evaluate_model


def build_model(args, device):
    """
    Train mode  → always fresh LoRA weights (or --resume loads checkpoint)
    Evaluate    → auto-loads checkpoints/ if present (or --base_only skips it)
    """
    if args.mode == "train":
        if args.resume:
            from peft import PeftModel
            from transformers import CLIPModel
            base  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model = PeftModel.from_pretrained(base, args.resume)
            print(f"🔁 Resuming from: {args.resume}")
        else:
            model = get_clip_lora(r=args.lora_r, lora_alpha=args.lora_alpha,
                                  lora_dropout=args.lora_dropout)
            print("🆕 Fresh LoRA weights (training from scratch)")

    else:  # evaluate
        ckpt = Path(args.checkpoint) if args.checkpoint else Path("checkpoints")
        if not args.base_only and ckpt.exists() and (ckpt / "adapter_config.json").exists():
            from peft import PeftModel
            from transformers import CLIPModel
            base  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model = PeftModel.from_pretrained(base, str(ckpt))
            print(f"✅ Checkpoint loaded: {ckpt}")
        else:
            model = get_clip_lora(r=args.lora_r, lora_alpha=args.lora_alpha)
            print("🧠 Base CLIP (no checkpoint)")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="ScopeSearch — CLIP + LoRA for VFX Asset Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--metadata",   type=str, required=True, help="Train/eval CSV")
    parser.add_argument("--val",        type=str, default=None,  help="Val CSV (training)")

    # ── Training basics ───────────────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=30,   help="Number of epochs")
    parser.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Peak learning rate")

    # ── Regularization ────────────────────────────────────────────────────────
    parser.add_argument("--weight_decay",  type=float, default=1e-2,
                        help="AdamW weight decay (L2 regularization)")
    parser.add_argument("--grad_clip",     type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for contrastive loss (0.0–0.2)")

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    parser.add_argument("--scheduler",    type=str, default="cosine",
                        choices=["cosine", "linear", "none"],
                        help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Warmup epochs before peak LR")

    # ── LoRA config ───────────────────────────────────────────────────────────
    parser.add_argument("--lora_r",       type=int,   default=8,   help="LoRA rank")
    parser.add_argument("--lora_alpha",   type=int,   default=16,  help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # ── Checkpoint handling ───────────────────────────────────────────────────
    parser.add_argument("--save_dir",   type=str, default="checkpoints/",
                        help="Where to save LoRA weights after training")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Resume training from a checkpoint dir (e.g. checkpoints/)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate (defaults to checkpoints/)")
    parser.add_argument("--base_only",  action="store_true",
                        help="Evaluate base CLIP without checkpoint (baseline comparison)")

    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    model = build_model(args, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.mode == "train":
        train_loader = create_dataloaders(args.metadata,
                                          batch_size=args.batch_size, shuffle=True)
        val_loader   = None
        if args.val:
            val_loader = create_dataloaders(args.val,
                                            batch_size=args.batch_size, shuffle=False)

        criterion = ContrastiveLoss(label_smoothing=args.label_smoothing)

        print(f"\n🚀 Training {args.epochs} epochs  lr={args.lr}  "
              f"wd={args.weight_decay}  sched={args.scheduler}  "
              f"lora_r={args.lora_r}  dropout={args.lora_dropout}")

        model = train_model(
            model, train_loader, val_loader,
            criterion      = criterion,
            num_epochs     = args.epochs,
            learning_rate  = args.lr,
            weight_decay   = args.weight_decay,
            grad_clip      = args.grad_clip,
            scheduler_type = args.scheduler,
            warmup_epochs  = args.warmup_epochs,
            device         = device,
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        print(f"\n💾 LoRA weights saved: {save_dir}")
        print("🎉 Done! Next:")
        print(f"  uv run python main.py --mode evaluate --metadata {args.val or args.metadata}")
        print(f"  uv run python main.py --mode evaluate --metadata {args.val or args.metadata} --base_only")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        label = "Base CLIP" if args.base_only else "LoRA fine-tuned"
        print(f"📊 [{label}] Evaluating on: {args.metadata}")
        val_loader = create_dataloaders(args.metadata,
                                        batch_size=args.batch_size, shuffle=False)
        evaluate_model(model, val_loader, device=device)


if __name__ == "__main__":
    main()
