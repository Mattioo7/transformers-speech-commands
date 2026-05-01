# Uruchomienie projektu na Windows 11 z RTX 4070

Instrukcja zaklada czysty Windows 11, PowerShell oraz karte NVIDIA RTX 4070. Projekt uzywa `uv`, Jupyter Lab, PyTorch, torchaudio oraz programu `7z` do czytania archiwum `data/train.7z`.

## 1. Sterownik NVIDIA

1. Zainstaluj aktualny sterownik NVIDIA Game Ready albo Studio Driver dla RTX 4070.
2. Otworz nowy PowerShell i sprawdz:

```powershell
nvidia-smi
```

Jesli polecenie pokazuje RTX 4070, wersje sterownika i tabele procesow, sterownik jest widoczny. Nie trzeba instalowac pelnego CUDA Toolkit do zwyklego uzycia PyTorch z gotowych wheel-i CUDA.

## 2. Narzedzia systemowe

W PowerShell uruchomionym jako administrator:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id 7zip.7zip
winget install -e --id Astral-sh.UV
```

Zamknij PowerShell i otworz go ponownie, zeby odswiezyc `PATH`. Sprawdz:

```powershell
python --version
uv --version
7z
```

## 3. Skopiowanie projektu i danych

Skopiuj caly katalog projektu na Windows, np. do:

```text
C:\Users\<uzytkownik>\Projects\P2-transformers
```

Plik z danymi musi byc dokladnie tutaj:

```text
C:\Users\<uzytkownik>\Projects\P2-transformers\data\train.7z
```

Ten plik nie jest sledzony przez Git, bo katalog `data/` jest ignorowany. Trzeba go przeniesc osobno z obecnego laptopa.

## 4. Instalacja zaleznosci

W PowerShell przejdz do katalogu projektu:

```powershell
cd "C:\Users\<uzytkownik>\Projects\P2-transformers"
uv sync --python 3.12
```

Nastepnie wymus instalacje wariantu PyTorch z CUDA dla Windows. Dla RTX 4070 wybierz aktualny wheel CUDA z oficjalnego indeksu PyTorch, obecnie dobrym wyborem jest CUDA 13.0:

```powershell
uv pip install --reinstall torch==2.11.0+cu130 torchaudio==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Jesli oficjalny selektor PyTorch pokazuje nowsza stabilna komende dla Windows + Pip + CUDA, uzyj jej zamiast powyzszej, zachowujac zgodne wersje `torch` i `torchaudio`.

## 5. Weryfikacja srodowiska

Uruchom:

```powershell
uv run python scripts/check_setup.py
```

Oczekiwane najwazniejsze linie:

```text
[OK] Python >= 3.12
[OK] 7z is available in PATH
[OK] data/train.7z exists
[OK] nvidia-smi works
[OK] torch.cuda.is_available()
CUDA device: NVIDIA GeForce RTX 4070 ...
```

Jesli `torch.cuda.is_available()` ma `FAIL`, najczestsza przyczyna to CPU-only PyTorch. Powtorz instalacje PyTorch z indeksem CUDA z kroku 4.

## 6. Uruchomienie notebookow

Start Jupyter Lab:

```powershell
uv run jupyter lab
```

Otworz notebooki w tej kolejnosci:

1. `01_dataset_analysis.ipynb`
2. `02_baseline_models.ipynb`
3. `03_hyperparameter_experiments.ipynb`

W konfiguracjach treningu ustawienie `FitFixedParams(device="auto")` wybierze CUDA, jesli PyTorch widzi GPU. W `02_baseline_models.ipynb` zostalo to ustawione na `auto`.

## 7. Typowe problemy

`7z` nie jest rozpoznawane:

```powershell
$env:Path += ";C:\Program Files\7-Zip"
```

Potem uruchom PowerShell ponownie. Docelowo warto dodac `C:\Program Files\7-Zip` do zmiennych srodowiskowych systemu.

`nvidia-smi` nie dziala:

Zainstaluj lub przeinstaluj sterownik NVIDIA, a potem zrestartuj komputer.

CUDA dziala, ale trening jest wolny:

Sprawdz w notebooku, czy nie ma lokalnego `FitFixedParams(device="cpu")`. Dla RTX 4070 uzywaj `device="auto"` albo `device="cuda"`.

Brakuje pamieci GPU:

Zmniejsz `batch_size` w `FitGridParams`, np. z `64` do `32` albo `16`.

## Zrodla

- PyTorch Start Locally: https://docs.pytorch.org/get-started/locally/
- Oficjalne wheel-e PyTorch CUDA: https://pytorch.org/get-started/previous-versions/
