from pathlib import Path
from dataclasses import dataclass
import math

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchaudio

from .archive import archive_files, archive_lines, extract_archive_files
from .config import DataFixedParams, FeatureFixedParams, LABEL_TO_ID, UNKNOWN_LABEL


@dataclass(frozen=True)
class PreparedDataFiles:
    manifest: pd.DataFrame
    train_manifest: pd.DataFrame
    validation_manifest: pd.DataFrame
    test_manifest: pd.DataFrame
    local_paths: dict[str, Path]


@dataclass(frozen=True)
class PreparedData:
    manifest: pd.DataFrame
    train_manifest: pd.DataFrame
    validation_manifest: pd.DataFrame
    test_manifest: pd.DataFrame
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset


def train_audio_label(path: Path) -> str | None:
    parts = path.parts
    if len(parts) != 4 or parts[:2] != ("train", "audio") or path.suffix != ".wav":
        return None
    return parts[2]


def speaker_id(path: Path) -> str:
    return path.name.split("_nohash_", 1)[0]


def build_command_manifest(data_params: DataFixedParams) -> pd.DataFrame:
    archive = Path(data_params.data_dir) / data_params.train_archive
    validation_paths = archive_lines(archive, "train/validation_list.txt")
    testing_paths = archive_lines(archive, "train/testing_list.txt")

    rows = []
    for archive_path in archive_files(archive):
        label = train_audio_label(archive_path)
        if label is None or label == "_background_noise_":
            continue

        relative_path = Path(*archive_path.parts[2:])
        split = "test" if relative_path in testing_paths else "validation" if relative_path in validation_paths else "train"
        mapped_label = label if label in data_params.target_labels else data_params.unknown_label

        rows.append(
            {
                "archive_path": str(archive_path),
                "relative_path": str(relative_path),
                "original_label": label,
                "label": mapped_label,
                "split": split,
                "speaker_id": speaker_id(archive_path),
                "segment_index": 0,
            }
        )

    return pd.DataFrame(rows)


def background_noise_paths(data_params: DataFixedParams) -> list[Path]:
    archive = Path(data_params.data_dir) / data_params.train_archive
    return [
        path
        for path in archive_files(archive)
        if len(path.parts) == 4
        and path.parts[:3] == ("train", "audio", "_background_noise_")
        and path.suffix == ".wav"
    ]


def add_silence_examples(
    manifest: pd.DataFrame,
    data_params: DataFixedParams,
    examples_per_split: int,
) -> pd.DataFrame:
    if not data_params.include_silence or examples_per_split <= 0:
        return manifest

    noise_paths = background_noise_paths(data_params)
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(examples_per_split):
            path = noise_paths[index % len(noise_paths)]
            rows.append(
                {
                    "archive_path": str(path),
                    "relative_path": str(Path(*path.parts[2:])),
                    "original_label": "_background_noise_",
                    "label": data_params.silence_label,
                    "split": split,
                    "speaker_id": f"background_{index % len(noise_paths)}",
                    "segment_index": index,
                }
            )

    return pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)


def sample_split(
    manifest: pd.DataFrame,
    split: str,
    fraction: float,
    unknown_fraction: float,
    seed: int,
) -> pd.DataFrame:
    split_data = manifest[manifest["split"] == split].copy()
    sampled = []

    for label, group in split_data.groupby("label", sort=False):
        label_fraction = unknown_fraction if label == UNKNOWN_LABEL else fraction
        if label_fraction >= 1.0:
            sampled.append(group)
            continue

        count = max(1, math.ceil(len(group) * label_fraction))
        sampled.append(group.sample(n=count, random_state=seed))

    return pd.concat(sampled, ignore_index=True)


def build_experiment_manifest(
    data_params: DataFixedParams,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    unknown_fraction: float,
    silence_examples_per_split: int,
    seed: int,
) -> pd.DataFrame:
    manifest = build_command_manifest(data_params)
    manifest = add_silence_examples(manifest, data_params, silence_examples_per_split)

    train_manifest = sample_split(manifest, "train", train_fraction, unknown_fraction, seed)
    validation_manifest = sample_split(manifest, "validation", validation_fraction, unknown_fraction, seed)
    test_manifest = sample_split(manifest, "test", test_fraction, unknown_fraction, seed)
    return pd.concat([train_manifest, validation_manifest, test_manifest], ignore_index=True)


def pad_or_trim(waveform: torch.Tensor, target_length: int, segment_index: int = 0) -> torch.Tensor:
    if waveform.numel() < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - waveform.numel()))

    if waveform.numel() == target_length:
        return waveform

    max_start = waveform.numel() - target_length
    start = 0 if segment_index == 0 else (segment_index * target_length) % (max_start + 1)
    return waveform[start : start + target_length]


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        local_paths: dict[str, Path],
        data_params: DataFixedParams,
        feature_params: FeatureFixedParams,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.local_paths = local_paths
        self.data_params = data_params
        self.feature_params = feature_params
        self.target_length = int(data_params.sample_rate * data_params.clip_seconds)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=data_params.sample_rate,
            n_fft=feature_params.n_fft,
            hop_length=feature_params.hop_length,
            n_mels=feature_params.n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        waveform, sample_rate = torchaudio.load(self.local_paths[row.archive_path])
        waveform = waveform.mean(dim=0)

        if sample_rate != self.data_params.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.data_params.sample_rate)

        waveform = pad_or_trim(waveform, self.target_length, int(row.segment_index))
        features = self.to_db(self.mel(waveform)).transpose(0, 1)

        if self.feature_params.normalize:
            features = (features - features.mean()) / features.std().clamp_min(1e-6)

        return features, torch.tensor(LABEL_TO_ID[row.label], dtype=torch.long)


def prepare_experiment_data_files(experiment, data_grid: dict) -> PreparedDataFiles:
    data_params = experiment.data_fixed
    manifest = build_experiment_manifest(
        data_params=data_params,
        train_fraction=data_grid["train_fraction"],
        validation_fraction=data_grid["validation_fraction"],
        test_fraction=data_grid["test_fraction"],
        unknown_fraction=data_grid["unknown_fraction"],
        silence_examples_per_split=data_grid["silence_examples_per_split"],
        seed=data_grid["seed"],
    )

    archive = Path(data_params.data_dir) / data_params.train_archive
    cache_dir = Path(data_params.cache_dir) / experiment.name
    local_paths = extract_archive_files(archive, [Path(path) for path in manifest["archive_path"]], cache_dir)

    train_manifest = manifest[manifest["split"] == "train"]
    validation_manifest = manifest[manifest["split"] == "validation"]
    test_manifest = manifest[manifest["split"] == "test"]

    return PreparedDataFiles(
        manifest=manifest,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        local_paths=local_paths,
    )


def build_datasets_from_prepared_files(experiment, prepared_files: PreparedDataFiles) -> PreparedData:
    data_params = experiment.data_fixed
    train_dataset = SpeechCommandsDataset(
        prepared_files.train_manifest,
        prepared_files.local_paths,
        data_params,
        experiment.feature_fixed,
    )
    validation_dataset = SpeechCommandsDataset(
        prepared_files.validation_manifest,
        prepared_files.local_paths,
        data_params,
        experiment.feature_fixed,
    )
    test_dataset = SpeechCommandsDataset(
        prepared_files.test_manifest,
        prepared_files.local_paths,
        data_params,
        experiment.feature_fixed,
    )

    return PreparedData(
        manifest=prepared_files.manifest,
        train_manifest=prepared_files.train_manifest,
        validation_manifest=prepared_files.validation_manifest,
        test_manifest=prepared_files.test_manifest,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )


def prepare_experiment_data(experiment, data_grid: dict) -> PreparedData:
    prepared_files = prepare_experiment_data_files(experiment, data_grid)
    return build_datasets_from_prepared_files(experiment, prepared_files)


def make_dataloaders(
    prepared_data: PreparedData,
    experiment,
    data_grid: dict,
    fit_grid: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sampler = None
    shuffle = True
    if data_grid["sampling_strategy"] == "class_balanced":
        label_counts = prepared_data.train_manifest["label"].value_counts().to_dict()
        weights = [1.0 / label_counts[label] for label in prepared_data.train_manifest["label"]]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        prepared_data.train_dataset,
        batch_size=fit_grid["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=experiment.fit_fixed.num_workers,
        pin_memory=experiment.fit_fixed.pin_memory,
    )
    validation_loader = DataLoader(
        prepared_data.validation_dataset,
        batch_size=fit_grid["batch_size"],
        shuffle=False,
        num_workers=experiment.fit_fixed.num_workers,
        pin_memory=experiment.fit_fixed.pin_memory,
    )
    test_loader = DataLoader(
        prepared_data.test_dataset,
        batch_size=fit_grid["batch_size"],
        shuffle=False,
        num_workers=experiment.fit_fixed.num_workers,
        pin_memory=experiment.fit_fixed.pin_memory,
    )

    return train_loader, validation_loader, test_loader
