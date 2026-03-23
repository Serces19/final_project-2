import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_model_comparison():
    # Data from evaluations
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    
    baseline = [0.1154, 0.4231, 0.5769, 0.2570]
    v2       = [0.0769, 0.5385, 0.7308, 0.2668]
    v3       = [0.1154, 0.6154, 0.8462, 0.3170]
    
    # Convert to percentages for R@K, keep MRR as decimal
    baseline_display = [b * 100 if i < 3 else b for i, b in enumerate(baseline)]
    v2_display       = [v * 100 if i < 3 else v for i, v in enumerate(v2)]
    v3_display       = [v * 100 if i < 3 else v for i, v in enumerate(v3)]

    # Aesthetics
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor('#1e1e1e')
    
    colors = ['#888888', '#3b82f6', '#10b981'] # Gray, Blue, Emerald Green
    labels = ['Baseline (Base CLIP)', 'V2 (Naive LoRA)', 'V3 (Optimized LoRA)']
    
    # --- Subplot 1: Recall@K (%) ---
    ax1.set_facecolor('#1e1e1e')
    x = np.arange(3)  # R@1, R@5, R@10
    width = 0.25      # width of the bars
    
    rects1 = ax1.bar(x - width, baseline_display[:3], width, label=labels[0], color=colors[0])
    rects2 = ax1.bar(x,         v2_display[:3],       width, label=labels[1], color=colors[1])
    rects3 = ax1.bar(x + width, v3_display[:3],       width, label=labels[2], color=colors[2])
    
    ax1.set_ylabel('Recall Score (%)', fontsize=12, color='#e2e8f0')
    ax1.set_title('Recall@K Progression (Higher is better)', fontsize=14, color='white', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['R@1', 'R@5', 'R@10'], fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1e1e1e', edgecolor='#333333')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add text labels on the bars
    def autolabel(rects, ax, is_pct=True):
        for rect in rects:
            height = rect.get_height()
            label = f"{height:.1f}%" if is_pct else f"{height:.3f}"
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='white')

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)

    # --- Subplot 2: MRR ---
    ax2.set_facecolor('#1e1e1e')
    x_mrr = np.arange(1)
    width_mrr = 0.2
    
    rm1 = ax2.bar(x_mrr - width_mrr, [baseline_display[3]], width_mrr, color=colors[0])
    rm2 = ax2.bar(x_mrr,             [v2_display[3]],       width_mrr, color=colors[1])
    rm3 = ax2.bar(x_mrr + width_mrr, [v3_display[3]],       width_mrr, color=colors[2])
    
    ax2.set_title('Mean Reciprocal Rank', fontsize=14, color='white', pad=20)
    ax2.set_xticks(x_mrr)
    ax2.set_xticklabels(['MRR'], fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 0.4)

    autolabel(rm1, ax2, is_pct=False)
    autolabel(rm2, ax2, is_pct=False)
    autolabel(rm3, ax2, is_pct=False)

    # Save
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "final_metrics_comparison.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"📊 Plot saved to {out_path}")

if __name__ == "__main__":
    plot_model_comparison()
