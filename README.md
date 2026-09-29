# AutoNeuroRepair

EEG/BCI self-healing research project. Current stage: Pipeline, ANR-T001 v1.2.
T001 loads only BCIC IV 2a `A01T.gdf` and records provenance; it does not
implement Detection, Diagnosis, or Repair.

## Environment and tests

Python 3.11 is required. Use a working Python 3.11 virtual environment; the
existing `.venv` has not been deleted or rebuilt by this task.

```powershell
Set-Location C:\AutoNeuroRepair
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest -q -m "not integration"
```

Gate A uses artificial files and a Mock Reader. It requires no real GDF and
does not import MNE on its injected-reader path. The ordinary editable install
above also installs the pinned MNE dependency for the actual loader.

Gate B is separate and requires the real file. A trusted acquisition record
is optional; without it the source identity is not established and the
expected `processing_status` is `UNKNOWN`:

```powershell
$env:ANR_A01T_PATH = 'C:\data\A01T.gdf'
# Optional, all three or none:
$env:ANR_A01T_REFERENCE_SHA256 = '<64 lowercase hex characters from a trusted acquisition record>'
$env:ANR_A01T_REFERENCE_URL = '<HTTPS official acquisition URL>'
$env:ANR_A01T_REFERENCE_NOTE = '<how and when official acquisition was verified>'
& .\.venv\Scripts\python.exe -m pytest -q -rP -m integration
```

Replace every placeholder before running Gate B. Do not label a hash freshly
computed from an unverified file as an independently verified reference.
The reference can be a previously recorded hash from a verified official
download; ANR does not claim that the publisher has published a checksum.
A missing file causes a skip with `Gate B BLOCKED`. A skip is not a Gate B PASS.
Gate B checks that the status follows consistently from the recorded evidence;
it does not require `RAW`. The A01T assessment is printed separately.

## Current verification status

T001 code is committed on `main`. Gate A was run at commit `f775173`
(Python 3.11.5, pytest 8.4.2): 52 passed, 0 failed, 0 skipped. Gate B is
BLOCKED because no actual `A01T.gdf` is available, so T001 is not PASS. See
[implementation notes](docs/anr-t001.md) for decisions, scope, and limitations.
