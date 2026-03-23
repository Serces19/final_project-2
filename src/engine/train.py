import json
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm


LOGS_DIR = Path("logs")


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
    top_k = torch.topk(sim, k=k, dim=1).indices
    targets = torch.arange(len(imgs), device=sim.device).view(-1, 1)
    recall = (top_k == targets).sum().item() / len(imgs)
    model.train()
    return recall


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    bar = tqdm(dataloader, desc=f"Epoch {epoch:>3}")
    for batch in bar:
        optimizer.zero_grad()
        out = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            pixel_values   = batch["pixel_values"].to(device),
        )
        loss = criterion(out.image_embeds, out.text_embeds)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(dataloader)


def train_model(model, train_dataloader, val_dataloader,
                criterion, num_epochs=10, learning_rate=1e-4, device="cuda"):
    model     = model.to(device)
    criterion = criterion.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=learning_rate)

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Recall@1 (val)':>14}")
    print("─" * 38)

    history = []
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_dataloader, criterion,
                                   optimizer, device, epoch)
        r1 = None
        if val_dataloader is not None:
            r1 = _compute_recall_at_k(model, val_dataloader, device, k=1)
            print(f"{epoch:>6} | {avg_loss:>10.4f} | {r1:>13.2%}")
        else:
            print(f"{epoch:>6} | {avg_loss:>10.4f} | {'N/A':>14}")

        history.append({"epoch": epoch, "train_loss": avg_loss, "recall_at_1": r1})

    _save_metrics(history)
    _print_summary(history)
    return model


def _save_metrics(history: list):
    """Persist training history to logs/training_metrics.json."""
    LOGS_DIR.mkdir(exist_ok=True)
    out = LOGS_DIR / "training_metrics.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Metrics saved: {out}")
    print(f"   Generate plots: uv run python scripts/plot_metrics.py")


def _print_summary(history):
    import pandas as pd
    print("\n" + "═" * 40)
    print("  TRAINING SUMMARY")
    print("═" * 40)
    df = pd.DataFrame(history)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    best_loss = df.loc[df["train_loss"].idxmin()]
    print(f"\n  Best loss  : {best_loss.train_loss:.4f} (epoch {int(best_loss.epoch)})")
    if df["recall_at_1"].notna().any():
        best_r1 = df.loc[df["recall_at_1"].idxmax()]
        print(f"  Best R@1   : {best_r1.recall_at_1:.2%} (epoch {int(best_r1.epoch)})")
    print("═" * 40 + "\n")
