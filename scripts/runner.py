from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Experiment, LABEL_ORDER, expand_experiment_grid
from .data import make_dataloaders
from .models import build_model
from .outputs import output_paths, run_name, save_confusion_matrix_plot, save_history_plot, save_json
from .training import confusion_matrix, fit_model, metrics_summary


def run_single_config(experiment: Experiment, run: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(run["data"]["seed"])
    np.random.seed(run["data"]["seed"])

    paths = output_paths(experiment, run)
    save_json(paths["configs"] / "config.json", {"experiment": experiment.name, **run})

    train_loader, test_loader, manifest = make_dataloaders(experiment, run["data"], run["fit"])
    manifest.to_csv(paths["metrics"] / "manifest.csv", index=False)

    model = build_model(run["model"], experiment.feature_fixed, len(LABEL_ORDER))
    history, final_metrics = fit_model(model, train_loader, test_loader, run["fit"], experiment.fit_fixed)

    summary = metrics_summary(history, final_metrics)
    pd.DataFrame(history).to_csv(paths["metrics"] / "history.csv", index=False)
    pd.DataFrame([summary]).to_csv(paths["metrics"] / "summary.csv", index=False)

    matrix = confusion_matrix(final_metrics["targets"], final_metrics["predictions"], len(LABEL_ORDER))
    pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(paths["metrics"] / "confusion_matrix.csv")

    save_history_plot(history, paths["figures"] / "learning_curves.png")
    save_confusion_matrix_plot(matrix, LABEL_ORDER, paths["figures"] / "confusion_matrix.png")

    return {"run": run_name(run), **summary}


def run_experiment(experiment: Experiment) -> pd.DataFrame:
    runs = expand_experiment_grid(experiment)
    iterator = tqdm(runs, desc=experiment.name) if experiment.fit_fixed.use_tqdm else runs
    results = [run_single_config(experiment, run) for run in iterator]
    summary = pd.DataFrame(results)
    output_dir = Path(experiment.data_fixed.output_dir) / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    return summary
