from dataclasses import dataclass, field, fields
from itertools import product
from typing import Any, Literal


TARGET_LABEL_ORDER = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
UNKNOWN_LABEL = "unknown"
SILENCE_LABEL = "silence"
LABEL_ORDER = TARGET_LABEL_ORDER + (UNKNOWN_LABEL, SILENCE_LABEL)
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

ModelType = Literal["lstm", "transformer"]
SamplingStrategy = Literal["natural", "class_balanced"]
Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class DataFixedParams:
    data_dir: str = "data"
    train_archive: str = "train.7z"
    cache_dir: str = ".cache/baseline_audio"
    output_dir: str = "reports/02_baseline_models"
    target_labels: tuple[str, ...] = TARGET_LABEL_ORDER
    unknown_label: str = UNKNOWN_LABEL
    silence_label: str = SILENCE_LABEL
    sample_rate: int = 16_000
    clip_seconds: float = 1.0
    include_silence: bool = True


@dataclass(frozen=True)
class DataGridParams:
    train_fraction: float | list[float] = 1.0
    test_fraction: float | list[float] = 1.0
    unknown_fraction: float | list[float] = 1.0
    silence_examples_per_split: int | list[int] = 2_000
    sampling_strategy: SamplingStrategy | list[SamplingStrategy] = "natural"
    seed: int | list[int] = 42


@dataclass(frozen=True)
class FeatureFixedParams:
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    normalize: bool = True


@dataclass(frozen=True)
class ModelGridParams:
    model_type: ModelType | list[ModelType] = field(default_factory=lambda: ["lstm", "transformer"])
    dropout: float | list[float] = 0.2

    lstm_hidden_size: int | list[int] = 128
    lstm_layers: int | list[int] = 2
    lstm_bidirectional: bool | list[bool] = True

    transformer_d_model: int | list[int] = 128
    transformer_heads: int | list[int] = 4
    transformer_layers: int | list[int] = 2
    transformer_ff_dim: int | list[int] = 256


@dataclass(frozen=True)
class FitFixedParams:
    device: Device = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    log_every: int = 20


@dataclass(frozen=True)
class FitGridParams:
    epochs: int | list[int] = 5
    batch_size: int | list[int] = 64
    learning_rate: float | list[float] = 1e-3
    weight_decay: float | list[float] = 1e-4


@dataclass(frozen=True)
class Experiment:
    name: str
    data_fixed: DataFixedParams = field(default_factory=DataFixedParams)
    data_grid: DataGridParams = field(default_factory=DataGridParams)
    feature_fixed: FeatureFixedParams = field(default_factory=FeatureFixedParams)
    model_grid: ModelGridParams = field(default_factory=ModelGridParams)
    fit_fixed: FitFixedParams = field(default_factory=FitFixedParams)
    fit_grid: FitGridParams = field(default_factory=FitGridParams)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def expand_grid(dataclass_instance: Any) -> list[dict[str, Any]]:
    keys = [field.name for field in fields(dataclass_instance)]
    values = [as_list(getattr(dataclass_instance, key)) for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def expand_experiment_grid(experiment: Experiment) -> list[dict[str, Any]]:
    runs = []
    for data_params, model_params, fit_params in product(
        expand_grid(experiment.data_grid),
        expand_grid(experiment.model_grid),
        expand_grid(experiment.fit_grid),
    ):
        runs.append(
            {
                "experiment": experiment.name,
                "data": data_params,
                "model": model_params,
                "fit": fit_params,
            }
        )
    return runs


def experiment_grid_dataframe(experiment: Experiment):
    import pandas as pd

    rows = []
    for run in expand_experiment_grid(experiment):
        rows.append(
            {
                "experiment": run["experiment"],
                **{f"data.{key}": value for key, value in run["data"].items()},
                **{f"model.{key}": value for key, value in run["model"].items()},
                **{f"fit.{key}": value for key, value in run["fit"].items()},
            }
        )
    return pd.DataFrame(rows)
