import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss(losses: list[float], save_path: str = "loss_curve.png"):
    """
    Plot training loss over epochs.

    Args:
        losses:    List of loss values, one per epoch.
        save_path: Where to save the output image.

    Example:
        losses = [0.91, 0.82, 0.74, 0.61, 0.52, 0.45, 0.38, 0.32, 0.28]
        plot_loss(losses)
    """
    epochs = list(range(1, len(losses) + 1))

    start  = losses[0]
    final  = losses[-1]
    best   = min(losses)
    best_e = losses.index(best) + 1
    reduction = ((start - final) / start) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Shaded fill under curve
    ax.fill_between(epochs, losses, alpha=0.15, color="#4a90d9")

    # Main loss line
    ax.plot(epochs, losses, color="#4a90d9", linewidth=2.2, label="Training loss")

    # Mark best point
    ax.scatter([best_e], [best], color="#f5a623", zorder=5, s=60, label=f"Best: {best:.4f} (epoch {best_e})")

    # Axes styling
    ax.set_xlabel("Epoch", color="#aaaaaa", fontsize=12)
    ax.set_ylabel("Loss",  color="#aaaaaa", fontsize=12)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", color="#222222", linewidth=0.8)
    ax.grid(axis="x", color="#1a1a1a", linewidth=0.5)

    # Summary stats in top-right corner
    summary = (
        f"Epochs:    {len(losses)}\n"
        f"Start:     {start:.4f}\n"
        f"Final:     {final:.4f}\n"
        f"Reduction: {reduction:.1f}%"
    )
    ax.text(
        0.98, 0.97, summary,
        transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        color="#aaaaaa",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#333333")
    )

    ax.legend(framealpha=0.2, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#cccccc")
    plt.title("Training loss", color="#dddddd", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    # --- Example: replace with your actual loss list ---
    example_losses = [
        0.91, 0.82, 0.74, 0.67, 0.61, 0.57, 0.52, 0.48, 0.45, 0.43,
        0.40, 0.38, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28,
    ]
    plot_loss(example_losses)
