"""
scripts/plot_metrics.py
Generates training plots from logs/training_metrics.json.

Usage:
    uv run python scripts/plot_metrics.py
    uv run python scripts/plot_metrics.py --input logs/training_metrics.json --output logs/
"""
import argparse
import json
from pathlib import Path


def plot(input_path: Path, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    with open(input_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    losses     = [h["train_loss"] for h in history]
    recalls    = [h["recall_at_1"] for h in history]
    has_recall = any(r is not None for r in recalls)

    # ── Style ──────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor":   "#1e293b",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#cbd5e1",
        "xtick.color":      "#94a3b8",
        "ytick.color":      "#94a3b8",
        "text.color":       "#e2e8f0",
        "grid.color":       "#334155",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "lines.linewidth":  2.0,
        "font.family":      "sans-serif",
    })

    n_plots = 2 if has_recall else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 4.5),
                              facecolor="#0f172a")
    if n_plots == 1:
        axes = [axes]

    # ── Plot 1: Training Loss ──────────────────────────────
    ax = axes[0]
    ax.plot(epochs, losses, color="#6366f1", marker="o", markersize=3)
    ax.fill_between(epochs, losses, alpha=0.15, color="#6366f1")
    best_epoch = epochs[losses.index(min(losses))]
    ax.axvline(best_epoch, color="#f59e0b", linestyle=":", alpha=0.7,
               label=f"Best epoch ({best_epoch})")
    ax.set_title("Training Loss", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.grid(True)
    ax.legend(fontsize=9)

    # ── Plot 2: Recall@1 ───────────────────────────────────
    if has_recall:
        clean_r = [r if r is not None else float("nan") for r in recalls]
        ax2 = axes[1]
        ax2.plot(epochs, [r * 100 for r in clean_r], color="#06b6d4",
                 marker="s", markersize=3)
        ax2.fill_between(epochs, [r * 100 for r in clean_r], alpha=0.15, color="#06b6d4")
        best_r = max(r for r in clean_r if r == r)  # skip nan
        best_r_epoch = epochs[[i for i, r in enumerate(clean_r) if r == best_r][0]]
        ax2.axhline(best_r * 100, color="#f59e0b", linestyle=":", alpha=0.7,
                    label=f"Best R@1: {best_r:.1%}")
        ax2.set_title("Recall@1 on Validation Set", fontsize=13, fontweight="bold",
                      color="#e2e8f0")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Recall@1 (%)")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax2.grid(True)
        ax2.legend(fontsize=9)

    plt.suptitle("ScopeSearch — LoRA Fine-Tuning Metrics",
                 fontsize=14, fontweight="bold", color="#e2e8f0", y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "training_curves.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"✅ Plot saved: {out_png}")

    # ── Print final numbers ────────────────────────────────
    print(f"\n📊 Final metrics:")
    print(f"   Final loss  : {losses[-1]:.4f}")
    print(f"   Best loss   : {min(losses):.4f} (epoch {best_epoch})")
    if has_recall:
        print(f"   Best R@1    : {best_r:.2%} (epoch {best_r_epoch})")
        print(f"   Final R@1   : {clean_r[-1]:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--input",  type=str, default="logs/training_metrics.json")
    parser.add_argument("--output", type=str, default="logs/")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"❌ Metrics file not found: {inp}")
        print("   Run training first: uv run python main.py --mode train ...")
        return

    plot(inp, Path(args.output))


if __name__ == "__main__":
    main()
