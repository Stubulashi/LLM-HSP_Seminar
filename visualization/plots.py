"""Visualisation (pipeline.md 11.3 / scaf.md section 13 L1255-1292; artifact mapping from ruling C12).

- Confidence Curve: mean confidence per step for each model (scaf.md L1259-1281)
- Emergence Distribution: emergence histogram for each model (L1284-1291)
- trajectory.png: adjacent-step interpretation distance curve (the Stability semantics; ruling C12)
"""

from __future__ import annotations

from pathlib import Path


def plot_confidence(df, out_path: str | Path) -> Path:
    """df must have model / sentence_position / confidence columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    conf_numeric = pd.to_numeric(df.get("confidence"), errors="coerce")
    if df.empty or conf_numeric.dropna().empty:
        # all confidences missing/non-numeric: emit a "no data" placeholder instead of `no numeric data to plot`
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no confidence data (missing / non-numeric)",
                ha="center", va="center")
        ax.set_title("Mean confidence per step by model")
        ax.set_xlabel("Sentence position")
        ax.set_ylabel("Mean confidence")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    pivot = df.pivot_table(index="sentence_position", columns="model",
                           values="confidence", aggfunc="mean")
    pivot.plot.line(marker="o")
    plt.title("Mean confidence per step by model")
    plt.xlabel("Sentence position")
    plt.ylabel("Mean confidence")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_emergence(df, out_path: str | Path) -> Path:
    """df must have model / emergence_point columns; runs without emergence are NaN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or df["emergence_point"].dropna().empty:
        # no emergence data at all: emit an empty-figure placeholder
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no emergence data", ha="center", va="center")
        ax.set_title("Mean emergence point by model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Emergence point (step)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    pivot = df.pivot_table(index="model", values="emergence_point",
                           aggfunc="mean", dropna=True)
    pivot.plot.bar()
    plt.title("Mean emergence point by model")
    plt.xlabel("Model")
    plt.ylabel("Emergence point (step)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_trajectory(stability_by_model: dict[str, list[tuple[int, float]]],
                    out_path: str | Path) -> Path:
    """Adjacent-step distance curve (ruling C12: the Stability semantics → trajectory.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for model, points in stability_by_model.items():
        if points:
            steps, dists = zip(*points)
            plt.plot(steps, dists, marker="o", label=model)
    plt.title("Interpretation revision trajectory (cosine distance)")
    plt.xlabel("Step")
    plt.ylabel("Distance from previous step")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
