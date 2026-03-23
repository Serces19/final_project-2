import json
import math
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

LOGS_DIR = Path("logs")


# ─── Recall@K ────────────────────────────────────────────────────────────────
def _compute_recall_at_k(model, dataloader, device, k=1):
    import torch.nn.functional as F
    model.eval()
    all_img, all_txt = [], []
    with torch.no_grad():
        for batch in dataloader:
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                pixel_values   = batch["pixel_values"].to(device),
            )
            all_img.append(F.normalize(out.image_embeds, p=2, dim=-1))
            all_txt.append(F.normalize(out.text_embeds,  p=2, dim=-1))
    imgs = torch.cat(all_img)
    txts = torch.cat(all_txt)
    sim  = imgs @ txts.t()
    top_k   = torch.topk(sim, k=k, dim=1).indices
    targets = torch.arange(len(imgs), device=sim.device).view(-1, 1)
    recall  = (top_k == targets).sum().item() / len(imgs)
    model.train()
    return recall


# ─── Single epoch ─────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer,
                    scheduler, device, epoch, grad_clip):
    model.train()
    total_loss = 0.0
    bar = tqdm(dataloader, desc=f"Epoch {epoch:>4}")
    for batch in bar:
        optimizer.zero_grad()
        out = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            pixel_values   = batch["pixel_values"].to(device),
        )
        loss = criterion(out.image_embeds, out.text_embeds)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix({"loss": f"{loss.item():.4f}",
                         "lr":   f"{scheduler.get_last_lr()[0]:.2e}"})
    scheduler.step()
    return total_loss / len(dataloader)


# ─── Main training loop ───────────────────────────────────────────────────────
def train_model(model, train_dataloader, val_dataloader,
                criterion,
                num_epochs     = 30,
                learning_rate  = 1e-4,
                weight_decay   = 1e-2,
                grad_clip      = 1.0,
                scheduler_type = "cosine",
                warmup_epochs  = 2,
                device         = "cuda"):

    model     = model.to(device)
    criterion = criterion.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "No trainable parameters found. "
            "Did you accidentally load a checkpoint on top of get_clip_lora()? "
            "Use --resume to continue training from a checkpoint."
        )
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    # Build LR scheduler
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        return 1.0

    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    if scheduler_type == "cosine":
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=learning_rate * 0.01
        )
    elif scheduler_type == "linear":
        main_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs
        )
    else:
        main_sched = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
    )

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Recall@1':>9} | {'LR':>10}")
    print("─" * 46)

    history = []
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_dataloader, criterion,
                                   optimizer, scheduler, device, epoch, grad_clip)
        r1 = None
        lr_now = scheduler.get_last_lr()[0]

        if val_dataloader is not None:
            r1 = _compute_recall_at_k(model, val_dataloader, device, k=1)
            print(f"{epoch:>6} | {avg_loss:>10.4f} | {r1:>8.2%} | {lr_now:>10.2e}")
        else:
            print(f"{epoch:>6} | {avg_loss:>10.4f} | {'N/A':>9} | {lr_now:>10.2e}")

        history.append({"epoch": epoch, "train_loss": avg_loss,
                         "recall_at_1": r1, "lr": lr_now})

    _save_metrics(history)
    _print_summary(history)
    return model


# ─── Persistence & summary ───────────────────────────────────────────────────
def _save_metrics(history: list):
    LOGS_DIR.mkdir(exist_ok=True)
    out = LOGS_DIR / "training_metrics.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Metrics saved: {out}")
    print("   Plots: uv run python scripts/plot_metrics.py")


def _print_summary(history):
    import pandas as pd
    print("\n" + "═" * 46)
    print("  TRAINING SUMMARY")
    print("═" * 46)
    df = pd.DataFrame(history)
    # Only show every 5th epoch for readability if many epochs
    step = max(1, len(df) // 20)
    print(df.iloc[::step].to_string(index=False, float_format="{:.4f}".format))
    best_loss = df.loc[df["train_loss"].idxmin()]
    print(f"\n  Best loss  : {best_loss.train_loss:.4f} (epoch {int(best_loss.epoch)})")
    if df["recall_at_1"].notna().any():
        best_r1 = df.loc[df["recall_at_1"].idxmax()]
        print(f"  Best R@1   : {best_r1.recall_at_1:.2%} (epoch {int(best_r1.epoch)})")
    print("═" * 46 + "\n")
