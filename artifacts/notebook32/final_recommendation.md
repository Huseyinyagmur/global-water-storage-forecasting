# Notebook 32 final recommendation

**Status: ready for controlled global download**

- Strategy: 30-degree tiles intersecting competition coordinates; no blind global bbox.
- Used tiles: 50; requests: 850 total, 850 pending.
- Source months: 2002-04 to 2018-11.
- Estimated download: 8.91 GB; uncompressed: 20.25 GB.
- Free disk: 51.68 GB; minimum required: 34.24 GB; gate passed: True.
- Estimated CDS duration: 10-24 depending on CDS queue.
- Exact command after explicit approval: `.venv\Scripts\python.exe scripts\download_era5_land_global.py --execute --confirm I_APPROVE_GLOBAL_ERA5_DOWNLOAD`.
- No global download, validation, prediction or submission was performed in this preparation run.

After download completes, run `python scripts/prepare_era5_land_global_features.py`, then rerun Notebook 32 for empirical validation. Existing canonical/current-best submissions remain immutable.
