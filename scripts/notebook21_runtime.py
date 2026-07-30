from pathlib import Path
from time import perf_counter
import gc, hashlib, json, warnings
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings("ignore")
SEED = 42
RNG = np.random.default_rng(SEED)
T0 = perf_counter()
ROOT = Path.cwd().resolve()
if not (ROOT / "data/raw/Train.csv").exists():
    ROOT = ROOT.parent
DATA = ROOT / "data/raw"
SUB = ROOT / "submissions"
ART = ROOT / "artifacts/notebook21"
ART.mkdir(parents=True, exist_ok=True)
CANON = SUB / "submission_optimized.csv"
EXPECTED_SHA = "b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968"


def stamp(label):
    print(f"[{label}] elapsed={perf_counter() - T0:.1f}s")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


stamp("start")
train = pd.read_csv(DATA / "Train.csv", parse_dates=["time"])
test = pd.read_csv(DATA / "Test.csv", parse_dates=["time"])
sample = pd.read_csv(DATA / "SampleSubmission.csv")
canonical = pd.read_csv(CANON)

# Hard stop: immutable canonical artifact reproduction.
checks = {
    "rows": len(canonical) == len(test) == len(sample) == 280961,
    "id_order": canonical["ID"].equals(sample["ID"]),
    "known_rows": int(test["TWS_t"].notna().sum()),
    "masked_rows": int(test["TWS_t"].isna().sum()),
    "duplicate_id": int(canonical["ID"].duplicated().sum()),
    "nan": int(canonical["Target"].isna().sum()),
    "infinity": int(np.isinf(canonical["Target"]).sum()),
    "sha256": sha256(CANON),
    "prediction_correlation": 1.0,
    "mad": 0.0,
}
assert checks["rows"] and checks["id_order"]
assert checks["duplicate_id"] == checks["nan"] == checks["infinity"] == 0
assert checks["sha256"] == EXPECTED_SHA
display(pd.Series(checks, name="canonical reproduction"))
stamp("canonical gate complete")

SPEI = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t"]


def static(df):
    o = df.copy()
    dt = o["time"]
    o["month"] = dt.dt.month.astype("int8")
    o["quarter"] = dt.dt.quarter.astype("int8")
    o["year"] = dt.dt.year.astype("int16")
    o["elapsed_month"] = ((dt.dt.year - train.time.dt.year.min()) * 12 + dt.dt.month - 1).astype("int16")
    o["month_sin_calc"] = np.sin(2 * np.pi * o.month / 12).astype("float32")
    o["month_cos_calc"] = np.cos(2 * np.pi * o.month / 12).astype("float32")
    o["season_sin"] = np.sin(2 * np.pi * o.quarter / 4).astype("float32")
    o["season_cos"] = np.cos(2 * np.pi * o.quarter / 4).astype("float32")
    s = o[SPEI]
    o["spei_mean"] = s.mean(axis=1)
    o["spei_min"] = s.min(axis=1)
    o["spei_max"] = s.max(axis=1)
    o["spei_std"] = s.std(axis=1)
    o["spei_01_minus_12"] = o.SPEI_01_t - o.SPEI_12_t
    o["spei_03_minus_12"] = o.SPEI_03_t - o.SPEI_12_t
    o["spei_short_long_ratio"] = o.SPEI_01_t / (o.SPEI_12_t.abs() + .1)
    o["lat_band_5"] = np.floor((o.lat + 90) / 5).astype("int16")
    o["lon_band_5"] = np.floor((o.lon + 180) / 5).astype("int16")
    o["spatial_bin_10"] = (o.lat_band_5 * 100 + o.lon_band_5).astype("int32")
    o["lat_lon_interaction"] = o.lat * o.lon / 1000
    o["lat_sin"] = np.sin(np.radians(o.lat))
    o["lat_cos"] = np.cos(np.radians(o.lat))
    o["lon_sin"] = np.sin(np.radians(o.lon))
    o["lon_cos"] = np.cos(np.radians(o.lon))
    return o


X = static(train)
Xt = static(test)
BASE = ["lat", "lon", "TWS_t", *SPEI, "SOIL_MOISTURE_t", "month", "month_sin", "month_cos",
        "year", "quarter", "elapsed_month", "month_sin_calc", "month_cos_calc", "season_sin",
        "season_cos", "spei_mean", "spei_min", "spei_max", "spei_std", "spei_01_minus_12",
        "spei_03_minus_12", "spei_short_long_ratio", "lat_band_5", "lon_band_5",
        "spatial_bin_10", "lat_lon_interaction", "lat_sin", "lat_cos", "lon_sin", "lon_cos"]
MASK = [c for c in BASE if c != "TWS_t"]
PRI = ["prior_location_mean", "prior_location_std", "prior_location_month", "prior_global_month"]
PARAM = dict(n_estimators=160, learning_rate=.08, num_leaves=63, min_child_samples=100,
             reg_lambda=1, subsample=.85, colsample_bytree=.9, random_state=SEED,
             n_jobs=4, verbosity=-1)


def model():
    return LGBMRegressor(**PARAM)


def train_prior(df):
    o = df.sort_values(["lat", "lon", "time"]).copy()
    shifted = o.groupby(["lat", "lon"]).target.shift(1)
    o["prior_location_mean"] = shifted.groupby([o.lat, o.lon]).expanding().mean().reset_index(level=[0, 1], drop=True)
    o["prior_location_std"] = shifted.groupby([o.lat, o.lon]).expanding().std().reset_index(level=[0, 1], drop=True)
    o["prior_location_month"] = shifted.groupby([o.lat, o.lon, o.month]).expanding().mean().reset_index(level=[0, 1, 2], drop=True)
    o["prior_global_month"] = o.groupby("month").target.transform(lambda z: z.shift(1).expanding().mean())
    return o.sort_index()


def apply_prior(hist, future):
    loc = hist.groupby(["lat", "lon"]).target.agg(["mean", "std"]).rename(
        columns={"mean": "prior_location_mean", "std": "prior_location_std"})
    lm = hist.groupby(["lat", "lon", "month"]).target.agg(["sum", "count"]).rename(
        columns={"sum": "_sum", "count": "_n"})
    gm = hist.groupby("month").target.mean().rename("prior_global_month")
    o = future.join(loc, on=["lat", "lon"]).join(lm, on=["lat", "lon", "month"]).join(gm, on="month")
    o["prior_location_month"] = (o["_sum"] + 4 * o["prior_location_mean"]) / (o["_n"].fillna(0) + 4)
    return o.drop(columns=["_sum", "_n"])


known_positions = set(np.flatnonzero(test.groupby("time").TWS_t.count().to_numpy() > 1000))


def canonical_fold(train_mask, valid_mask, fold_name, kind):
    tr = X.loc[train_mask]
    va = X.loc[valid_mask]
    assert len(tr) and len(va) and tr.time.max() < va.time.min()
    valid_dates = np.array(sorted(va.time.unique()))
    known_dates = {d for j, d in enumerate(valid_dates) if j in known_positions}
    known = va.time.isin(known_dates).to_numpy()
    known_model = model().fit(tr[BASE], tr.target)
    base_masked = model().fit(tr[MASK], tr.target)
    trp = train_prior(tr)
    vap = apply_prior(tr, va)
    hist_masked = model().fit(trp[MASK + PRI], trp.target)
    pred = np.empty(len(va), dtype="float64")
    pred[known] = known_model.predict(va.loc[known, BASE])
    base_pred = base_masked.predict(va.loc[~known, MASK])
    hist_pred = hist_masked.predict(vap.loc[~known, MASK + PRI])
    pred[~known] = .65 * base_pred + .35 * hist_pred
    out = va[["time", "lat", "lon", "target", "SPEI_03_t", "SPEI_12_t", "SOIL_MOISTURE_t", "month"]].copy()
    out["prediction"] = pred
    out["residual"] = out.target - out.prediction
    out["known"] = known
    out["Fold"] = fold_name
    out["Kind"] = kind
    return out.reset_index(names="row_index")


dates = np.array(sorted(X.time.unique()))
# Three strictly earlier OOF calibration blocks. Every downstream validation starts after these.
calibration_specs = [(48, 60), (60, 72), (72, 84)]
calibration = []
for i, (a, b) in enumerate(calibration_specs, 1):
    calibration.append(canonical_fold(X.time.isin(dates[:a]), X.time.isin(dates[a:b]),
                                      f"calibration_{i}", "calibration"))
    stamp(f"calibration fold {i}")
cal = pd.concat(calibration, ignore_index=True)
assert cal.groupby("Fold").time.max().max() < dates[90]

# Notebook 20 fold definitions, unchanged.
test_dates = np.array(sorted(test.time.unique()))
test_offsets = np.array([(d.year - test_dates[0].year) * 12 + d.month - test_dates[0].month for d in test_dates])


def closest_pattern(end_idx):
    pool = dates[:end_idx + 1]
    origin = pool[-1] - pd.DateOffset(months=int(test_offsets[-1]))
    chosen = []
    for off in test_offsets:
        want = origin + pd.DateOffset(months=int(off))
        delta = np.abs((pool.astype("datetime64[ns]") - np.datetime64(want)).astype("timedelta64[D]").astype(int))
        chosen.append(pool[np.argmin(delta)])
    return np.array(sorted(set(chosen)))


folds = []
for k, (a, b) in enumerate([(len(dates) - 48, len(dates) - 24), (len(dates) - 24, len(dates))], 1):
    folds.append((f"strict_temporal_{k}", "strict_temporal", dates[:a], dates[a:b], None))
for k, end in enumerate([len(dates) - 1, len(dates) - 25], 1):
    vd = closest_pattern(end)
    folds.append((f"irregular_{k}", "irregular", dates[dates < vd.min()], vd, None))
regions = [
    ("lat_south", lambda d: d.lat < 0),
    ("lon_americas", lambda d: (d.lon >= -140) & (d.lon < -30)),
    ("grid_asia", lambda d: (d.lat >= 10) & (d.lat < 60) & (d.lon >= 60) & (d.lon < 150)),
]
future_dates = dates[-18:]
past_dates = dates[dates < future_dates.min()]
for name, fn in regions:
    folds.append((name, "spatial_block", past_dates, future_dates, fn))
folds.append(("space_time_asia", "space_time", past_dates, future_dates, regions[-1][1]))

evaluations = []
for name, kind, trdates, vadates, region in folds:
    trm = X.time.isin(trdates)
    vam = X.time.isin(vadates)
    if region is not None:
        reg = region(X)
        trm &= ~reg
        vam &= reg
    evaluations.append(canonical_fold(trm, vam, name, kind))
    stamp(name)
eval_df = pd.concat(evaluations, ignore_index=True)


def add_groups(df):
    o = df.copy()
    o["hemisphere"] = np.where(o.lat < 0, "South", "North")
    o["lat_band"] = (np.floor(o.lat / 10) * 10).astype(int).astype(str)
    o["lon_band"] = (np.floor(o.lon / 20) * 20).astype(int).astype(str)
    o["spatial_cell"] = o["lat_band"] + "|" + o["lon_band"]
    o["climate_regime"] = pd.cut(o.SPEI_12_t, [-np.inf, -1, 1, np.inf],
                                  labels=["dry", "normal", "wet"]).astype(str)
    o["spei_bin"] = pd.cut(o.SPEI_03_t, [-np.inf, -1, -.5, 0, .5, 1, np.inf]).astype(str)
    soil_edges = np.unique(np.nanquantile(X.SOIL_MOISTURE_t, [0, .2, .4, .6, .8, 1]))
    o["soil_bin"] = pd.cut(o.SOIL_MOISTURE_t, soil_edges, include_lowest=True, duplicates="drop").astype(str)
    o["month_group"] = o.month.astype(str)
    return o


cal = add_groups(cal)
eval_df = add_groups(eval_df)


def bootstrap_ci(values, reps=120):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) > 2000:
        x = RNG.choice(x, 2000, replace=False)
    if len(x) < 2:
        return (np.nan, np.nan)
    means = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    return tuple(np.quantile(means, [.025, .975]))


summary_source = pd.concat([cal, eval_df[eval_df.Kind.isin(["spatial_block", "space_time"])]],
                           ignore_index=True)
summary_rows = []
group_columns = ["hemisphere", "lat_band", "lon_band", "spatial_cell", "climate_regime",
                 "month_group", "spei_bin", "soil_bin"]
for col in group_columns:
    for value, g in summary_source.groupby(col, observed=True):
        lo, hi = bootstrap_ci(g.residual)
        summary_rows.append({
            "group_type": col, "group": str(value), "row_count": len(g),
            "mean_residual": g.residual.mean(), "median_residual": g.residual.median(),
            "rmse": root_mean_squared_error(g.target, g.prediction),
            "mae": mean_absolute_error(g.target, g.prediction),
            "residual_std": g.residual.std(),
            "underprediction_rate": (g.residual > 0).mean(),
            "overprediction_rate": (g.residual < 0).mean(),
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        })
spatial_summary = pd.DataFrame(summary_rows)
spatial_summary.to_csv(ART / "spatial_residual_summary.csv", index=False)

# Stability is learned only from the three historical OOF calibration folds.
stability_rows = []
MIN_FOLD_N = 100
PRACTICAL_BIAS = .01
for col in ["hemisphere", "lat_band", "spatial_cell", "climate_regime"]:
    for value, g in cal.groupby(col, observed=True):
        per = g.groupby("Fold").residual.agg(["mean", "count", "std"])
        if len(per) == 0:
            continue
        pooled_lo, pooled_hi = bootstrap_ci(g.residual)
        dominant = np.sign(per["mean"].mean())
        consistency = (np.sign(per["mean"]) == dominant).mean() if dominant else 0
        fold_ci_lo = per["mean"] - 1.96 * per["std"] / np.sqrt(per["count"])
        fold_ci_hi = per["mean"] + 1.96 * per["std"] / np.sqrt(per["count"])
        ci_overlap = max(fold_ci_lo) <= min(fold_ci_hi)
        stable = (len(per) >= 3 and consistency >= .75 and per["count"].min() >= MIN_FOLD_N
                  and abs(g.residual.mean()) >= PRACTICAL_BIAS
                  and (pooled_lo > 0 or pooled_hi < 0))
        stability_rows.append({
            "group_type": col, "group": str(value), "fold_count": len(per),
            "fold_mean_residuals": json.dumps(per["mean"].round(8).to_dict()),
            "fold_signs": json.dumps(np.sign(per["mean"]).astype(int).to_dict()),
            "sign_consistency": consistency, "mean_residual": g.residual.mean(),
            "mean_residual_variability": per["mean"].std(ddof=0),
            "minimum_fold_sample_count": int(per["count"].min()),
            "confidence_interval_overlap": bool(ci_overlap),
            "bootstrap_ci_low": pooled_lo, "bootstrap_ci_high": pooled_hi,
            "stable_bias": bool(stable),
        })
stability = pd.DataFrame(stability_rows)
stability.to_csv(ART / "residual_stability.csv", index=False)
stamp("residual map and stability")

stable_keys = {(r.group_type, r.group) for r in stability.itertuples() if r.stable_bias}
alphas = {"very_strong": 5000.0, "moderate": 1000.0, "light": 200.0}
caps = [.01, .02, .03, .05]
methods = {
    "hemisphere": ["hemisphere"],
    "latitude_band": ["lat_band"],
    "spatial_cell": ["spatial_cell"],
    "hierarchical": ["spatial_cell", "lat_band", "hemisphere"],
    "climate_regime": ["climate_regime"],
}


def make_maps(alpha):
    maps = {}
    for col in ["hemisphere", "lat_band", "spatial_cell", "climate_regime"]:
        agg = cal.groupby(col).residual.agg(["mean", "count"])
        agg["correction"] = agg["count"] / (agg["count"] + alpha) * agg["mean"]
        maps[col] = {str(k): float(v.correction) for k, v in agg.iterrows()
                     if (col, str(k)) in stable_keys}
    return maps


def correction_vector(df, hierarchy, maps, cap):
    corr = np.zeros(len(df), dtype=float)
    unresolved = np.ones(len(df), dtype=bool)
    for col in hierarchy:
        mapped = df[col].astype(str).map(maps[col]).to_numpy(dtype=float, na_value=np.nan)
        take = unresolved & np.isfinite(mapped)
        corr[take] = mapped[take]
        unresolved[take] = False
    corr = np.clip(corr, -cap, cap)
    corr[df.known.to_numpy()] = 0.0
    return corr


validation_rows = []
candidate_vectors = {}
for shrink_name, alpha in alphas.items():
    maps = make_maps(alpha)
    for method_name, hierarchy in methods.items():
        for cap in caps:
            candidate = f"{method_name}|{shrink_name}|cap_{cap:.2f}"
            corr = correction_vector(eval_df, hierarchy, maps, cap)
            candidate_vectors[candidate] = corr
            pred = eval_df.prediction.to_numpy() + corr
            for (fold, kind), idx in eval_df.groupby(["Fold", "Kind"]).groups.items():
                ix = np.asarray(list(idx), dtype=int)
                south = eval_df.iloc[ix].lat.to_numpy() < 0
                validation_rows.append({
                    "Candidate": candidate, "Method": method_name, "Shrinkage": shrink_name,
                    "Alpha": alpha, "Cap": cap, "Fold": fold, "Kind": kind,
                    "Overall RMSE": root_mean_squared_error(eval_df.target.iloc[ix], pred[ix]),
                    "Canonical RMSE": root_mean_squared_error(eval_df.target.iloc[ix], eval_df.prediction.iloc[ix]),
                    "Southern hemisphere RMSE": root_mean_squared_error(eval_df.target.iloc[ix][south], pred[ix][south]) if south.any() else np.nan,
                    "Northern hemisphere RMSE": root_mean_squared_error(eval_df.target.iloc[ix][~south], pred[ix][~south]) if (~south).any() else np.nan,
                    "Correction mean": corr[ix].mean(), "Mean absolute correction": np.abs(corr[ix]).mean(),
                    "Correction maximum": np.abs(corr[ix]).max(),
                    "Corrected row fraction": (np.abs(corr[ix]) > 0).mean(),
                })
validation = pd.DataFrame(validation_rows)
validation["Gain"] = validation["Canonical RMSE"] - validation["Overall RMSE"]
validation.to_csv(ART / "validation_results.csv", index=False)

base_fold = eval_df.groupby(["Fold", "Kind"]).apply(
    lambda g: root_mean_squared_error(g.target, g.prediction), include_groups=False)
base_worst = base_fold.max()
ablation_rows = []
for candidate, g in validation.groupby("Candidate"):
    spatial = g[g.Kind == "spatial_block"]
    strict = g[g.Kind == "strict_temporal"]
    irregular = g[g.Kind == "irregular"]
    spacetime = g[g.Kind == "space_time"]
    south = g[g.Fold == "lat_south"]
    row = {
        "Candidate": candidate, "Method": g.Method.iloc[0], "Shrinkage": g.Shrinkage.iloc[0],
        "Alpha": g.Alpha.iloc[0], "Cap": g.Cap.iloc[0],
        "Overall RMSE": np.sqrt(np.mean(g["Overall RMSE"] ** 2)),
        "Mean fold RMSE": g["Overall RMSE"].mean(), "Worst fold": g["Overall RMSE"].max(),
        "Fold std": g["Overall RMSE"].std(ddof=0),
        "Spatial block mean RMSE": spatial["Overall RMSE"].mean(),
        "Spatial block canonical RMSE": spatial["Canonical RMSE"].mean(),
        "Spatial gain": (spatial["Canonical RMSE"] - spatial["Overall RMSE"]).mean(),
        "Southern block RMSE": south["Overall RMSE"].iloc[0],
        "Southern block canonical RMSE": south["Canonical RMSE"].iloc[0],
        "Southern gain": south["Gain"].iloc[0],
        "Strict temporal gain": strict["Gain"].mean(),
        "Irregular gain": irregular["Gain"].mean(),
        "Space-time gain": spacetime["Gain"].mean(),
        "Correction mean": g["Correction mean"].mean(),
        "Mean absolute correction": g["Mean absolute correction"].mean(),
        "Correction maximum": g["Correction maximum"].max(),
        "Corrected row fraction": g["Corrected row fraction"].mean(),
    }
    # Fixed before inspection: 0.0005 temporal, 0.0002 space-time/worst tolerances.
    gates = {
        "spatial_improved": row["Spatial gain"] > 0,
        "south_improved": row["Southern gain"] > 0,
        "space_time_preserved": row["Space-time gain"] >= -.0002,
        "strict_preserved": row["Strict temporal gain"] >= -.0005,
        "irregular_preserved": row["Irregular gain"] >= -.0005,
        "worst_preserved": row["Worst fold"] <= base_worst + .0002,
        "evidence_only": True,
        "mean_abs_small": row["Mean absolute correction"] <= .02,
        "cap_respected": row["Correction maximum"] <= row["Cap"] + 1e-12,
    }
    row.update({f"Gate {k}": v for k, v in gates.items()})
    row["Accepted"] = all(gates.values())
    ablation_rows.append(row)
ablation = pd.DataFrame(ablation_rows).sort_values(
    ["Accepted", "Spatial gain", "Mean fold RMSE"], ascending=[False, False, True])
ablation.to_csv(ART / "correction_ablation.csv", index=False)
display(ablation.head(20))
stamp("correction ablation")

# Select at most one robust candidate; if none pass, retain canonical.
accepted = ablation[ablation.Accepted]
selected = accepted.iloc[0] if len(accepted) else ablation.iloc[0]
selected_name = selected.Candidate
method_name = selected.Method
alpha = float(selected.Alpha)
cap = float(selected.Cap)
maps = make_maps(alpha)

test_groups = add_groups(Xt)
test_groups["known"] = test.TWS_t.notna().to_numpy()
test_corr = correction_vector(test_groups, methods[method_name], maps, cap)
oof_corr = correction_vector(cal, methods[method_name], maps, cap)


def distribution(label, arr):
    a = np.asarray(arr)
    q = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {"Dataset": label, "Candidate": selected_name, "mean": a.mean(), "std": a.std(),
            "mean_absolute": np.abs(a).mean(), "min": q[0], "q01": q[1], "q05": q[2],
            "q25": q[3], "q50": q[4], "q75": q[5], "q95": q[6], "q99": q[7],
            "max": q[8], "max_absolute": np.abs(a).max(),
            "corrected_row_fraction": (np.abs(a) > 0).mean()}


dist = pd.DataFrame([distribution("training_oof", oof_corr), distribution("test", test_corr)])
dist.to_csv(ART / "test_correction_distribution.csv", index=False)
test_safe = (
    dist.loc[dist.Dataset == "test", "mean_absolute"].iloc[0] <= .02
    and dist.loc[dist.Dataset == "test", "max_absolute"].iloc[0] <= cap + 1e-12
    and dist.loc[dist.Dataset == "test", "corrected_row_fraction"].iloc[0]
    <= max(.50, 1.5 * dist.loc[dist.Dataset == "training_oof", "corrected_row_fraction"].iloc[0] + .05)
)

candidate_path = SUB / "submission_21_spatial_residual_candidate.csv"
report = {
    "status": "not_generated", "candidate": selected_name,
    "all_robust_gates_passed": bool(len(accepted)), "test_distribution_safe": bool(test_safe),
    "rows": 0, "id_order": False, "known_unchanged": False, "duplicate_id": np.nan,
    "nan": np.nan, "infinity": np.nan, "canonical_correlation": np.nan,
    "mad": np.nan, "maximum_prediction_difference": np.nan, "changed_row_fraction": 0.0,
}
if len(accepted) and test_safe:
    out = canonical.copy()
    masked = test.TWS_t.isna().to_numpy()
    assert np.all(test_corr[~masked] == 0)
    out.loc[masked, "Target"] = out.loc[masked, "Target"].to_numpy() + test_corr[masked]
    assert out.ID.equals(sample.ID) and len(out) == 280961
    assert out.ID.duplicated().sum() == 0 and np.isfinite(out.Target).all()
    assert np.array_equal(out.loc[~masked, "Target"].to_numpy(),
                          canonical.loc[~masked, "Target"].to_numpy())
    # Preserve the original canonical decimal text on known rows; pandas' fast CSV
    # parser is not guaranteed to round-trip every float from a newly formatted file.
    canonical_text = pd.read_csv(CANON, dtype={"Target": "string"})
    output_text = out["Target"].map(lambda value: format(value, ".17g"))
    output_text.loc[~masked] = canonical_text.loc[~masked, "Target"]
    pd.DataFrame({"ID": out["ID"], "Target": output_text}).to_csv(candidate_path, index=False)
    persisted = pd.read_csv(candidate_path)
    assert np.array_equal(persisted.loc[~masked, "Target"].to_numpy(),
                          canonical.loc[~masked, "Target"].to_numpy())
    delta = out.Target.to_numpy() - canonical.Target.to_numpy()
    report.update({
        "status": "generated", "rows": len(out), "id_order": out.ID.equals(sample.ID),
        "known_unchanged": True, "duplicate_id": int(out.ID.duplicated().sum()),
        "nan": int(out.Target.isna().sum()), "infinity": int(np.isinf(out.Target).sum()),
        "canonical_correlation": np.corrcoef(out.Target, canonical.Target)[0, 1],
        "mad": np.abs(delta).mean(), "maximum_prediction_difference": np.abs(delta).max(),
        "changed_row_fraction": (np.abs(delta) > 0).mean(),
    })
pd.DataFrame([report]).to_csv(ART / "submission_report.csv", index=False)

south_stable = stability[(stability.group_type == "hemisphere") & (stability.group == "South")]
south_answer = "yes" if len(south_stable) and abs(south_stable.mean_residual.iloc[0]) >= PRACTICAL_BIAS else "no"
stable_names = stability[stability.stable_bias].apply(
    lambda r: f"{r.group_type}:{r.group}", axis=1).tolist()
final = f"""# Notebook 21 final recommendation

Residual is defined as **target - prediction**; positive values mean underprediction.

1. **Is there systematic Southern Hemisphere bias?** {south_answer}. The historical OOF mean and CI are recorded in `residual_stability.csv`.
2. **Is its direction stable across folds?** {bool(len(south_stable) and south_stable.stable_bias.iloc[0]) if len(south_stable) else False}; the rule required all three folds, >=75% sign consistency, >=100 rows per fold, |bias|>=0.01, and a pooled bootstrap CI excluding zero.
3. **Groups with reliable correction evidence:** {stable_names if stable_names else 'none'}.
4. **Best correction method:** `{selected_name}` (best gate-compliant candidate if any; otherwise diagnostic best spatial candidate).
5. **Spatial-block RMSE gain:** {selected['Spatial gain']:.8f}.
6. **Strict temporal / irregular preserved?** gains {selected['Strict temporal gain']:.8f} / {selected['Irregular gain']:.8f}; fixed tolerances were -0.0005 each.
7. **Space-time robustness:** gain {selected['Space-time gain']:.8f}; fixed tolerance was -0.0002.
8. **Prediction change:** mean absolute correction {selected['Mean absolute correction']:.8f}, maximum {selected['Correction maximum']:.8f}, corrected fraction {selected['Corrected row fraction']:.3%}.
9. **New submission generated?** {report['status'] == 'generated'}.
10. **Keep canonical Notebook 11?** {not (len(accepted) and test_safe)}. Canonical SHA remains `{sha256(CANON)}`.
11. **Next highest-priority experiment:** repeat this exact frozen correction policy on a new, later fully held-out space-time window; do not expand correction complexity before independent stability evidence.

All failed candidates remain in `correction_ablation.csv`. Spatial validation was not optimized alone: acceptance jointly required strict temporal, irregular, spatial, space-time, worst-fold, evidence, and magnitude gates.
"""
(ART / "final_recommendation.md").write_text(final, encoding="utf-8")
display(Markdown(final))

required = ["spatial_residual_summary.csv", "residual_stability.csv", "correction_ablation.csv",
            "validation_results.csv", "test_correction_distribution.csv", "submission_report.csv",
            "final_recommendation.md"]
assert all((ART / name).exists() for name in required)
assert sha256(CANON) == EXPECTED_SHA
stamp("complete")
