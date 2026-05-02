import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar


MAX_COMMAND_LENGTH = 24_000
T = TypeVar("T")


def _archive_command(preferred: str | None = None) -> str:
    if preferred:
        command = shutil.which(preferred)
        if command is not None:
            return preferred

    for command in ("7z", "tar"):
        if shutil.which(command) is not None:
            return command

    raise FileNotFoundError("Neither '7z' nor 'tar' is available in PATH.")


def archive_files(archive: Path) -> list[Path]:
    command = _archive_command()
    if command == "7z":
        result = subprocess.run(
            ["7z", "l", "-slt", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        )

        files = []
        entry = {}

        def add_file() -> None:
            if entry.get("Path") and "Size" in entry and not entry.get("Attributes", "").startswith("D"):
                files.append(Path(entry["Path"]))

        for line in result.stdout.splitlines():
            if not line:
                add_file()
                entry = {}
                continue

            if " = " in line:
                key, value = line.split(" = ", 1)
                entry[key] = value

        add_file()
        return files

    result = subprocess.run(
        ["tar", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line and not line.endswith("/")]


def archive_lines(archive: Path, file_name: str) -> set[Path]:
    command = _archive_command()
    if command == "7z":
        args = ["7z", "e", "-so", str(archive), file_name]
    else:
        args = ["tar", "-xOf", str(archive), file_name]

    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return {Path(line) for line in result.stdout.splitlines() if line}


def _argument_batches(prefix: list[str], arguments: list[str], max_length: int = MAX_COMMAND_LENGTH) -> list[list[str]]:
    batches = []
    batch = []
    length = sum(len(arg) + 3 for arg in prefix)

    for argument in arguments:
        argument_length = len(argument) + 3
        if batch and length + argument_length > max_length:
            batches.append(batch)
            batch = []
            length = sum(len(arg) + 3 for arg in prefix)

        batch.append(argument)
        length += argument_length

    if batch:
        batches.append(batch)

    return batches


def extract_archive_files(
    archive: Path,
    paths: list[Path],
    output_dir: Path,
    *,
    progress: Callable[[Iterable[T]], Iterable[T]] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_paths = list(dict.fromkeys(paths))
    command = _archive_command(preferred="7z")
    progress = progress or (lambda iterable: iterable)

    if command == "7z":
        base_args = ["7z", "x", "-y", str(archive), f"-o{output_dir}"]
        if len(unique_paths) > 5_000:
            for batch in progress([["train/audio"]]):
                subprocess.run(base_args + batch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            batches = _argument_batches(base_args, [str(path) for path in unique_paths])
            for batch in progress(batches):
                subprocess.run(base_args + batch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        base_args = ["tar", "-xf", str(archive), "-C", str(output_dir)]
        if len(unique_paths) > 5_000:
            archive_paths = ["train/audio"]
        else:
            archive_paths = [str(path).replace("\\", "/") for path in unique_paths]

        batches = _argument_batches(base_args, archive_paths)
        for batch in progress(batches):
            subprocess.run(base_args + batch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {str(path): output_dir / path for path in unique_paths}
