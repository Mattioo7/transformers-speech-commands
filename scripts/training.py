from typing import Any
import subprocess
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Device, FitFixedParams
from .progress import progress_bar, stage


def pytorch_supported_cuda_arches() -> list[int]:
    return [
        int(arch.removeprefix("sm_"))
        for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_")
    ]


def nvidia_smi_device_arch() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_device = result.stdout.splitlines()[0].strip()
    major, minor = first_device.split(".", 1)
    return int(major) * 10 + int(minor)


def cuda_supports_current_device() -> bool:
    device_arch = nvidia_smi_device_arch()
    supported_arches = pytorch_supported_cuda_arches()

    if device_arch is not None:
        return device_arch in supported_arches and torch.cuda.is_available()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if not torch.cuda.is_available():
            return False

        try:
            major, minor = torch.cuda.get_device_capability()
            current_arch = major * 10 + minor
        except RuntimeError:
            return False

    return any(current_arch == arch for arch in supported_arches)


def resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if cuda_supports_current_device() else "cpu")

    if device == "cuda" and not cuda_supports_current_device():
        raise RuntimeError(
            "CUDA was requested, but the installed PyTorch build does not support this GPU. "
            "Use FitFixedParams(device='cpu') or install a PyTorch build compatible with your GPU."
        )

    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            predicted = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": predictions,
        "targets": targets,
    }


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    fit_params: dict[str, Any],
    fixed_params: FitFixedParams,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    device = resolve_device(fixed_params.device)
    stage(f"Using device: {device}", enabled=fixed_params.verbose)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fit_params["learning_rate"],
        weight_decay=fit_params["weight_decay"],
    )

    history = []
    best_state = clone_state_dict(model)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    start_time = time.perf_counter()
    epochs = progress_bar(
        range(1, fit_params["epochs"] + 1),
        enabled=fixed_params.use_tqdm,
        backend=fixed_params.progress_backend,
        description="Training",
        leave=True,
    )

    for epoch in epochs:
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        improved = validation_metrics["loss"] < best_validation_loss - fixed_params.early_stopping_min_delta
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if fixed_params.use_tqdm:
            epochs.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                val_loss=f"{validation_metrics['loss']:.4f}",
                val_acc=f"{validation_metrics['accuracy']:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6g}",
                refresh=False,
            )

        if (
            fixed_params.early_stopping
            and epochs_without_improvement >= fixed_params.early_stopping_patience
        ):
            stopped_early = True
            stage(
                f"Early stopping at epoch {epoch}; best validation loss at epoch {best_epoch}.",
                enabled=fixed_params.verbose,
            )
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_validation_loss"] = best_validation_loss
    final_metrics["epochs_trained"] = len(history)
    final_metrics["stopped_early"] = stopped_early
    elapsed = time.perf_counter() - start_time
    stage(f"Training finished in {elapsed:.2f} seconds\n\n", enabled=fixed_params.verbose)
    return history, final_metrics


def confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def metrics_summary(history: list[dict[str, float]], final_metrics: dict[str, Any]) -> dict[str, Any]:
    best = min(history, key=lambda row: row["validation_loss"])
    return {
        "best_epoch": final_metrics["best_epoch"],
        "epochs_trained": final_metrics["epochs_trained"],
        "stopped_early": final_metrics["stopped_early"],
        "train_loss": best["train_loss"],
        "train_accuracy": best["train_accuracy"],
        "validation_loss": best["validation_loss"],
        "validation_accuracy": best["validation_accuracy"],
        "test_loss": final_metrics["loss"],
        "test_accuracy": final_metrics["accuracy"],
    }
