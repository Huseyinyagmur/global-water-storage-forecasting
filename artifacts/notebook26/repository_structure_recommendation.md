# Repository structure recommendation

Current folders `notebooks/`, `scripts/`, `artifacts/`, `submissions/`, `data/`, `models/`, `configs/`, `src/`, `reports/` and `experiments/` are retained to avoid breaking paths. Root README, reproduction guide, model card and requirements files now provide entry points.

Recommendations for a future non-competition branch: add `docs/` for rendered reports, `tests/` for hash/schema/fold tests, and a documented `cache/` convention (Notebook 25 currently keeps verified caches within its artefact folder). Do not move large artefacts or data without updating builders. Temporary `.tmp_*` directories and large generated files should be reviewed before release, not deleted automatically. Competition data licensing should determine whether `data/` can be distributed.
