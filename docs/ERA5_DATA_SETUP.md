# ERA5-Land data setup

Notebook 28 uses the Copernicus CDS `reanalysis-era5-land-monthly-means` dataset. Raw files are external, large, provider-governed data and are not committed.

## Account and API key

1. Create/sign in to a [Climate Data Store account](https://cds.climate.copernicus.eu/).
2. Open the [ERA5-Land monthly dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) and accept its terms.
3. Follow the [official CDS API setup](https://cds.climate.copernicus.eu/how-to-api). Store the personal token only in the user-level `.cdsapirc`; never in this repository.
4. Install `cdsapi>=0.7.7`, `xarray`, `netCDF4`, and `pyarrow`.

## Directory structure

Raw NetCDF: `data/external/era5_land/raw/`  
Processed Parquet: `data/external/era5_land/processed/`  
Audit manifests: `artifacts/notebook28/`

Add these patterns to a local exclude/global gitignore if not already ignored:

```gitignore
data/external/era5_land/raw/
data/external/era5_land/processed/
.cdsapirc
```

## Variables and range

The first pilot downloads only total precipitation, total evaporation, surface runoff, sub-surface runoff, snowfall, snowmelt, and layer-4 volumetric soil water. Strict `t-1` coverage spans April 2002 through November 2018, derived automatically from Train/Test prediction origins.

## Run

```powershell
python scripts/download_era5_land.py
python scripts/download_era5_land.py --pilot --execute
python scripts/prepare_era5_land_features.py --pilot --method nearest
python scripts/prepare_era5_land_features.py --pilot --method linear
python scripts/download_era5_land.py --execute
python scripts/prepare_era5_land_features.py --method nearest
```

The first command is a dry run. The downloader never prints the token, skips cached files, and writes SHA256 values to the manifest. Run the full request only after pilot alignment and unit checks pass.

## Cutoff and units

Final selection uses source month `t-1`. ECMWF documents ERA5-Land-T at roughly five days behind real time and consolidated data at roughly 2–3 months. Current consolidated downloads do not prove their historical ERA5-Land-T vintage, so preserve release/version evidence and fail closed when uncertain.

For monthly means of daily means, hydrological accumulations are metres of water equivalent per day. Convert to millimetres per calendar month as `value * 1000 * days_in_month`. Verify ECMWF's evaporation sign convention before constructing water-loss features.

## Validation

First reproduce Notebook 11, then compare E0/E1/E2/E3 using strict temporal, irregular hindcast, spatial block and space-time folds. Never tune lag or acceptance thresholds after seeing validation results. No submission is made unless the frozen gates pass.

## Troubleshooting

- `blocked: CDS credentials/terms unavailable`: configure `.cdsapirc` and accept dataset terms.
- `cdsapi not installed`: install the packages above.
- NetCDF engine error: install `netCDF4`.
- Parquet error: install `pyarrow`.
- Coastal missingness: report it; do not backfill from future months.
- Existing cached file: verify its manifest checksum instead of downloading again.
