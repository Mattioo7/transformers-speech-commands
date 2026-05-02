import shutil
import subprocess
from pathlib import Path


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


def extract_archive_files(archive: Path, paths: list[Path], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_paths = list(dict.fromkeys(paths))
    command = _archive_command(preferred="7z")

    if command == "7z":
        if len(unique_paths) > 5_000:
            args = ["7z", "x", "-y", str(archive), f"-o{output_dir}", "train/audio"]
        else:
            args = ["7z", "x", "-y", str(archive), f"-o{output_dir}"] + [str(path) for path in unique_paths]
    else:
        if len(unique_paths) > 5_000:
            archive_paths = ["train/audio"]
        else:
            archive_paths = [str(path).replace("\\", "/") for path in unique_paths]

        args = ["tar", "-xf", str(archive), "-C", str(output_dir)] + archive_paths

    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {str(path): output_dir / path for path in unique_paths}
