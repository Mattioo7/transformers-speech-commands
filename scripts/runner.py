from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Experiment, LABEL_ORDER, expand_experiment_grid
from .data import PreparedData, make_dataloaders, prepare_experiment_data
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
    train_distribution = (
        manifest.loc[manifest["split"] == "train", "label"]
        .value_counts()
        .sort_index()
    )
    test_distribution = (
        manifest.loc[manifest["split"] == "test", "label"]
        .value_counts()
        .sort_index()
    )

    lines = [
        f"  -> TRAIN split | samples: {train_count}",
        "     class distribution:",
    ]
    lines.extend(f"       - {label}: {count}" for label, count in train_distribution.items())
    lines.append(f"  -> TEST split | samples: {test_count}")
    lines.append("     class distribution:")
    lines.extend(f"       - {label}: {count}" for label, count in test_distribution.items())
    return "\n".join(lines)


def data_key(run: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(run["data"].items()))


def prepare_data_for_runs(experiment: Experiment, runs: list[dict[str, Any]]) -> dict[tuple[tuple[str, Any], ...], PreparedData]:
    prepared = {}
    data_configs = []

    for run in runs:
        key = data_key(run)
        if key not in prepared:
            prepared[key] = None
            data_configs.append(run["data"])

    stage("\nBuilding dataset", enabled=experiment.fit_fixed.verbose)
    for index, data_config in enumerate(data_configs, start=1):
        if len(data_configs) > 1:
            stage(f"DATA configuration {index}/{len(data_configs)}", enabled=experiment.fit_fixed.verbose)

        data_run = {"data": data_config}
        prepared_data = prepare_experiment_data(experiment, data_config)
        prepared[data_key(data_run)] = prepared_data
        stage(split_summary(prepared_data.manifest), enabled=experiment.fit_fixed.verbose)

    return prepared


def run_single_config(
    experiment: Experiment,
    run: dict[str, Any],
    prepared_data: PreparedData,
    index: int,
    total: int,
) -> dict[str, Any]:
    name = run_name(run)
    stage(describe_run(run, index, total), enabled=experiment.fit_fixed.verbose)
    torch.manual_seed(run["data"]["seed"])
    np.random.seed(run["data"]["seed"])

    paths = output_paths(experiment, run)
    save_json(paths["configs"] / "config.json", {"experiment": experiment.name, **run})

    train_loader, test_loader = make_dataloaders(prepared_data, experiment, run["data"], run["fit"])
    prepared_data.manifest.to_csv(paths["metrics"] / "manifest.csv", index=False)

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
    prepared_data_by_key = prepare_data_for_runs(experiment, runs)
    results = [
        run_single_config(experiment, run, prepared_data_by_key[data_key(run)], index, len(runs))
        for index, run in enumerate(runs, start=1)
    ]
    summary = pd.DataFrame(results)
    output_dir = Path(experiment.data_fixed.output_dir) / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    stage(f"\nExperiment finished | total runs = {len(runs)}", enabled=experiment.fit_fixed.verbose)
    return summary
