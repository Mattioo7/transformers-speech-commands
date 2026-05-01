from pathlib import Path
from typing import Any
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Experiment


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def model_name_for_path(model_type: str) -> str:
    aliases = {
        "transformer": "trfm",
    }
    return aliases.get(model_type, model_type)


def run_name(run: dict[str, Any]) -> str:
    model = run["model"]
    data = run["data"]
    fit = run["fit"]
    return slugify(
        f"{model_name_for_path(model['model_type'])}_train{data['train_fraction']}_test{data['test_fraction']}_lr{fit['learning_rate']}_seed{data['seed']}"
    )


def output_paths(experiment: Experiment, run: dict[str, Any]) -> dict[str, Path]:
    root = Path(experiment.data_fixed.output_dir) / experiment.name / run_name(run)
    paths = {
        "root": root,
        "figures": root / "figures",
        "metrics": root / "metrics",
        "configs": root / "configs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def save_history_plot(history: list[dict[str, float]], path: Path) -> None:
    data = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(data["epoch"], data["train_accuracy"], label="train")
    axes[1].plot(data["epoch"], data["test_accuracy"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(matrix: np.ndarray, labels: tuple[str, ...], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
