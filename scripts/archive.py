from pathlib import Path
import subprocess


def archive_files(archive: Path) -> list[Path]:
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


def archive_lines(archive: Path, file_name: str) -> set[Path]:
    result = subprocess.run(
        ["7z", "e", "-so", str(archive), file_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def extract_archive_files(archive: Path, paths: list[Path], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_paths = list(dict.fromkeys(paths))

    if len(unique_paths) > 5_000:
        command = ["7z", "x", "-y", str(archive), f"-o{output_dir}", "train/audio"]
    else:
        command = ["7z", "x", "-y", str(archive), f"-o{output_dir}"] + [str(path) for path in unique_paths]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {str(path): output_dir / path for path in unique_paths}
