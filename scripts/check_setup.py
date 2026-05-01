from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def status(ok: bool, message: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {message}")


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    failed = False

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    status(sys.version_info >= (3, 12), "Python >= 3.12")
    failed |= sys.version_info < (3, 12)

    seven_zip = shutil.which("7z")
    status(seven_zip is not None, "7z is available in PATH")
    failed |= seven_zip is None

    archive = project_root / "data" / "train.7z"
    status(archive.exists(), "data/train.7z exists")
    failed |= not archive.exists()

    nvidia_smi = command_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
    )
    status(nvidia_smi is not None, "nvidia-smi works")
    if nvidia_smi:
        print(f"GPU: {nvidia_smi}")

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        status(False, f"PyTorch import failed: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    status(cuda_available, "torch.cuda.is_available()")
    failed |= not cuda_available

    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        try:
            x = torch.rand(256, 256, device="cuda")
            y = x @ x
        except RuntimeError as exc:
            status(False, f"simple CUDA tensor operation failed: {exc}")
            failed = True
        else:
            status(y.is_cuda, "simple CUDA tensor operation")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
