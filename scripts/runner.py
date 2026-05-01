from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Experiment, LABEL_ORDER, expand_experiment_grid
from .data import make_dataloaders
from .models import build_model
from .outputs import output_paths, run_name, save_confusion_matrix_plot, save_history_plot, save_json
from .progress import stage
from .training import confusion_matrix, fit_model, metrics_summary


def format_value(value: Any) -> str:
    return str(value)


def describe_run(run: dict[str, Any], index: int, total: int) -> str:
    lines = [f"\nConfiguration run {index}/{total}:"]
    groups = (
        ("DATA", run["data"]),
        ("MODEL", run["model"]),
        ("FIT", run["fit"]),
    )

    for title, params in groups:
        lines.append(f"{title} (variable):")
        for key, value in params.items():
            lines.append(f"  - {key}: {format_value(value)}")

    return "\n".join(lines)


def split_summary(manifest: pd.DataFrame) -> str:
    train_count = int((manifest["split"] == "train").sum())
    test_count = int((manifest["split"] == "test").sum())
    return f"  -> TRAIN split | samples: {train_count}\n  -> TEST split | samples: {test_count}"


def run_single_config(experiment: Experiment, run: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    name = run_name(run)
    stage(describe_run(run, index, total), enabled=experiment.fit_fixed.verbose)
    torch.manual_seed(run["data"]["seed"])
    np.random.seed(run["data"]["seed"])

    paths = output_paths(experiment, run)
    save_json(paths["configs"] / "config.json", {"experiment": experiment.name, **run})

    stage("\nBuilding dataset", enabled=experiment.fit_fixed.verbose)
    train_loader, test_loader, manifest = make_dataloaders(experiment, run["data"], run["fit"])
    stage(split_summary(manifest), enabled=experiment.fit_fixed.verbose)
    manifest.to_csv(paths["metrics"] / "manifest.csv", index=False)

    model = build_model(run["model"], experiment.feature_fixed, len(LABEL_ORDER))
    stage("\nTraining model", enabled=experiment.fit_fixed.verbose)
    history, final_metrics = fit_model(model, train_loader, test_loader, run["fit"], experiment.fit_fixed)

    summary = metrics_summary(history, final_metrics)
    pd.DataFrame(history).to_csv(paths["metrics"] / "history.csv", index=False)
    pd.DataFrame([summary]).to_csv(paths["metrics"] / "summary.csv", index=False)

    matrix = confusion_matrix(final_metrics["targets"], final_metrics["predictions"], len(LABEL_ORDER))
    pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(paths["metrics"] / "confusion_matrix.csv")

    save_history_plot(history, paths["figures"] / "learning_curves.png")
    save_confusion_matrix_plot(matrix, LABEL_ORDER, paths["figures"] / "confusion_matrix.png")

    return {"run": name, **summary}


def run_experiment(experiment: Experiment) -> pd.DataFrame:
    stage(f"Starting experiment: {experiment.name}", enabled=experiment.fit_fixed.verbose)
    runs = expand_experiment_grid(experiment)
    results = [run_single_config(experiment, run, index, len(runs)) for index, run in enumerate(runs, start=1)]
    summary = pd.DataFrame(results)
    output_dir = Path(experiment.data_fixed.output_dir) / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    stage(f"\nExperiment finished | total runs = {len(runs)}", enabled=experiment.fit_fixed.verbose)
    return summary
