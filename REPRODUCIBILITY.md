# Reproducibility guide

## Python and dependencies
Notebook metadata and the active project environment use Python 3.13, but not every early notebook was verified against a formal version range. Install direct packages from `requirements-minimal.txt`; use `requirements-reproducible.txt` to match versions captured from the active `.venv`.

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-minimal.txt
python scripts/build_notebook26.py
```

### Unix-like shells
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimal.txt
python scripts/build_notebook26.py
```

## Data layout
Place `Train.csv`, `Test.csv`, and `SampleSubmission.csv` in `data/raw/`. Data are not fetched automatically and may have competition licensing restrictions.

## Reading/execution order
For the final evidence chain use 11 → 17 → 19 → 20 → 21 → 23 → 24 → 25 → 26. Builders exist for later notebooks as `scripts/build_notebookNN.py`.

## Expected outputs
Notebook 26 writes reports/figures under `artifacts/notebook26/` and root documentation. Notebook 25 creates large OOF/cache artefacts; a second run validates the frozen policy key before reuse.

## SHA256 verification
Expected canonical: `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`.
Expected current-best: `84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488`.
Use `Get-FileHash ... -Algorithm SHA256` on Windows or `sha256sum ...` on Unix.

## Canonical/current-best reproduction
Notebook 17 identifies the exact canonical artifact; Notebook 21 creates the refinement; Notebook 23 verifies numerical policy reproduction. Do not overwrite either final CSV.

## Determinism
Advanced scripts use fixed seeds and explicit date ordering. LightGBM, threads, OS, compiler and package versions can still cause small numeric differences. Repository-external leaderboard scoring cannot be reproduced locally.

## Hardware
Report notebooks are lightweight. Model/OOF notebooks need several GB of RAM and multi-minute CPU runs; exact peak memory/latency is not available. Notebook 25 Parquet/cache files reduce repeat cost.

## Troubleshooting
- Run from repository root so relative paths resolve.
- Select a kernel backed by the environment containing LightGBM.
- If a cache policy key fails, allow Notebook 25 to refit rather than editing the manifest.
- Treat missing competition data or SHA mismatch as fail-fast conditions.
