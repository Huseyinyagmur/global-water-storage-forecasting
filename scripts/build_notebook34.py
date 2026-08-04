from __future__ import annotations

import json
from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "notebook34"
NB = ROOT / "notebooks" / "34_competitor_solution_reverse_engineering.ipynb"
AS_OF = "2026-08-04"

ZINDI_INFO = "https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge"
ZINDI_DATA = "https://zindi.africa/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/data"
ZINDI_LB = "https://zindi.africa/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/leaderboard"


def df(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def generate_artifacts() -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)

    discussions = df([
        ["Public discussion corpus unavailable without a stable authenticated rendering", "N/A", AS_OF,
         "No individual thread could be verified; no synthetic thread was created.", "Unknown",
         "Organizer clarifications must be collected through an authenticated manual export.",
         "Retain the prediction-time rules already transcribed and audited in Notebook 27; do not infer new rules.",
         "Access limitation", "Official competition discussion page", ZINDI_INFO, "Low"],
    ], ["Title", "Author", "Date", "Main_message", "Organizer_reply", "Technical_inference",
        "Pipeline_impact", "Category", "Source", "URL", "Confidence"])

    ranks = [1, 5, 10, 20, 50, 100, 125]
    leaderboard = df([[r, 0.823609466 if r == 125 else pd.NA,
                       "User-supplied approximate rank/current score" if r == 125 else "Not publicly retrievable; deliberately blank",
                       AS_OF, ZINDI_LB] for r in ranks],
                     ["Rank", "Public_RMSE", "Evidence", "Access_date", "URL"])
    leaderboard["Gap_from_current"] = leaderboard["Public_RMSE"].apply(
        lambda x: 0.823609466 - float(x) if pd.notna(x) else pd.NA)
    leaderboard["Approx_score_density_per_rank"] = pd.NA

    inventory = df([
        [ZINDI_INFO, "Official challenge information", "Live challenge", "Not a solution", "Competition columns/rules", "Rules permit qualifying external data", "Official split/evaluation", "N/A", "Primary", "High", "Specification", "None", "High"],
        [ZINDI_DATA, "Official data description", "Live challenge", "Not a solution", "TWS/SPEI/soil moisture", "N/A", "N/A", "N/A", "Primary", "High", "Exact schema", "None", "High"],
        ["https://github.com/hcu-cml/MTGNN-TWS-Reconstruction-GRACE", "MTGNN TWS reconstruction", "2026", "Spatiotemporal graph neural network", "ERA5 forcing, hybrid adjacency", "ERA5", "Spatiotemporal reconstruction", "Paper reports correlations, not competition RMSE", "Research code", "Medium", "Global panel/spatial dependence", "Graph adjacency and teleconnections", "Medium"],
        ["https://arxiv.org/abs/2101.09361", "Bayesian CNN GRACE gap filling", "2021", "Bayesian CNN with attention/residual/dense blocks", "Multi-source climate fields", "Climate products", "Spatial holdout not established from abstract", "Paper metrics; not comparable", "Academic", "Medium", "TWS reconstruction", "Spatial CNN plus uncertainty", "High"],
        ["https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL112677", "Near-real-time global TWSA", "2025", "Bayesian CNN", "Precipitation, temperature, reanalysed TWSA", "ERA5-Land", "Grid-consistent evaluation", "Not comparable", "Peer-reviewed", "Medium", "External forcing", "Stationarization and spatial CNN", "High"],
        ["https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR041710", "GRACE-FCast", "2026", "LSTM for non-seasonal signal", "Lagged hydrometeorological variables", "Observation products", "Out-of-time forecast", "Not comparable", "Peer-reviewed", "Medium", "Sequence memory", "Trend/seasonal decomposition plus uncertainty", "High"],
        ["https://blog.drivendata.org/blog/water-supply-hindcast-winners", "Water Supply Forecast Rodeo winners", "2024-05-23", "CatBoost ensemble and MLP", "SWE, precipitation, streamflow, temperature, elevation", "Official hydrology observations", "10 held-out historical years", "Quantile loss; not comparable", "Official winner write-up", "High", "Pooled hydrologic sites", "Site identity, direct/monthly decomposition", "High"],
        ["https://www.sciencedirect.com/science/article/pii/S1569843226002797", "WSDI component framework", "2026", "Index/relational analysis", "TWS components", "Hydrologic components", "Historical analysis", "Not forecasting score", "Peer-reviewed", "Low", "TWS physical interpretation", "Groundwater decoupling/human impacts", "Medium"],
        ["https://www.tandfonline.com/doi/full/10.1080/02626667.2020.1790564", "Ethiopia TWS forecasting", "2020", "Regional linear forecast", "SST and sea-level pressure", "Climate indices", "Regional temporal forecast", "Not comparable", "Peer-reviewed", "Medium", "TWS prediction", "Teleconnection predictors", "High"],
        ["https://doi.org/10.1175/JHM-D-21-0046.1", "Seasonal TWS forecast skill", "2021", "Hydrologic forecast plus regression", "TWS persistence", "NHyFAS", "Seasonal skill by region", "Not comparable", "Peer-reviewed", "Medium", "Persistence", "Region/season-dependent memory", "High"],
    ], ["URL", "Repository_or_work", "Last_update", "Models", "Features", "External_data", "Validation",
        "Score", "Code_quality", "Reproducibility", "Similarity", "Technique_missing_here", "Reliability"])

    analogs = df([
        ["DrivenData Water Supply Forecast Rodeo", "Seasonal basin inflow quantiles", "1/2/3", "CatBoost ensemble; MLP", "SWE, precipitation, flow", "Issue-date/to-date features", "Pooled sites with site_id", "Weighted", "10 held-out years", "Hydrologically causal forcings plus pooled/local representation", "High"],
        ["Forecast Rodeo II", "Subseasonal temperature/precipitation", "Winners", "Ensembles", "Operational forecasts/climate state", "Real-time issue dates", "Regional", "Yes", "26 forecast sprints", "Operational hindcasts and slow climate state", "Medium"],
        ["GRACE-FCast", "Global TWS change", "Paper", "LSTM", "Lagged hydrometeorology", "Trend/seasonal/nonseasonal decomposition", "Global grid", "Implicit decomposition", "Out-of-time 2010–2024", "Separate predictable components", "High"],
        ["Near-real-time global TWSA", "TWSA reconstruction", "Paper", "Bayesian CNN", "ERA5-Land precipitation/temperature", "Stationarization", "Spatial convolution", "Uncertainty", "Time-based", "External forcing plus spatial field", "High"],
        ["Bayesian CNN GRACE gap", "Gap filling", "Paper", "BCNN attention/residual", "Multi-source climate", "Sequence of fields", "Convolution", "Uncertainty", "GRACE gap", "Spatial context and uncertainty", "High"],
        ["Ethiopia TWS forecast", "Seasonal TWS", "Paper", "Linear regression", "SST/SLP", "Lag correlation", "Four hydro-regions", "No", "Regional time split", "Teleconnections can add information cheaply", "High"],
        ["GRACE downscaling", "TWSC downscaling", "Paper", "Partial least squares", "WaterGAP/P/ET/runoff", "Monthly", "0.5-degree fields", "Multi-product", "Cross-product checks", "Water-budget variables and regularization", "Medium"],
        ["Amazon drought TWS", "Months-ahead TWS", "Paper", "Statistical/ML workflow", "CHIRPS, CERES, MODIS, GLDAS", "Lagged forcing", "Amazon region", "Multi-source", "Uncertainty analysis", "Precipitation distribution and land-state forcing", "Medium"],
    ], ["Competition_or_paper", "Problem", "Winning_rank", "Model", "External_data", "Temporal_method",
        "Spatial_method", "Ensemble", "Validation", "Success_factor", "Transferability"])

    local = {
        "Gradient boosting": "Extensively; canonical family remains public-best",
        "Sequence models": "Notebook 33 bounded pilot: S1 0.696528 vs E0 0.772760",
        "LSTM/GRU": "GRU won pilot; LSTM weaker at L6",
        "TCN": "Notebook 33 L6 pilot weaker",
        "External reanalysis data": "ERA5-Land three-region pilot gain 0.009427",
        "ERA5/ERA5-Land": "Notebook 31 positive pilot; global download deferred",
        "Residual correction": "Tried; transfer concerns recorded",
        "Multi-stage experts": "Known/masked dual experts used",
        "Lag and rolling windows": "Multiple notebooks; flattened full panel not isolated",
        "Ensemble": "Canonical blend and sequence pilot blend useful",
    }
    pattern_defs = [
        ("Gradient boosting",3,"Hydrology/tabular pooled panels","Yes","None","Medium","High"),
        ("Sequence models",3,"TWS and hydrologic memory","Yes","Existing CPU PyTorch","High","High"),
        ("LSTM/GRU",2,"Global/seasonal TWS","Yes","CUDA desirable","High","High"),
        ("TCN",1,"Generic earth time series","Yes","Existing CPU PyTorch","Low","Medium"),
        ("Transformer",1,"Groundwater/EO sequences","No","Large framework/compute","Uncertain","Low"),
        ("Graph neural networks",1,"Spatial TWS reconstruction","No","PyTorch/graph package","Medium-long term","Medium"),
        ("ConvLSTM",2,"Gridded TWS fields","No","GPU plus gridded inputs","Medium","Medium"),
        ("Spatial lag features",2,"Geospatial hydrology","Partly","None","Medium","Medium"),
        ("External reanalysis data",4,"TWS reconstruction/forecast","Yes","Data download","High","High"),
        ("ERA5/ERA5-Land",3,"Global TWS","Yes","9GB deferred","High","High"),
        ("GLDAS",3,"TWS components/forecast","No","Large external data","Medium","High"),
        ("SMAP",1,"Soil moisture forecasting","No","External data","Low-medium","Medium"),
        ("IMERG/CHIRPS precipitation",2,"Drought/TWS","No","Small-to-medium data","Medium","High"),
        ("NDVI/MODIS",2,"Drought/vegetation coupling","No","External raster","Low-medium","Medium"),
        ("Climatology",3,"Seasonal hydrology","Yes","None","Medium","High"),
        ("Residual correction",2,"Bias/local regime correction","Yes","None","Low-medium","Medium"),
        ("Multi-stage experts",2,"Regime/missingness handling","Yes","None","Medium","High"),
        ("Location embeddings",2,"Pooled-site forecasting","No","Neural framework","Medium","Medium"),
        ("Lag and rolling windows",5,"Hydrologic memory","Yes","None","High","High"),
        ("Physics-informed loss",1,"Water balance","No","Custom training","Uncertain","Low"),
        ("Ensemble",4,"Competition hydrology","Yes","None","Medium","High"),
        ("Calibration",2,"Probabilistic/bias correction","Yes","None","Low","Medium"),
        ("Pseudo labeling",0,"No reliable direct evidence","No","None","Negative/risky","Low"),
    ]
    patterns = df([[p,n,kind,p in local,local.get(p,"Not isolated in repository"),
                    "Yes" if pkg in ("None","Existing CPU PyTorch") else "No",pkg,effect,conf]
                   for p,n,kind,_,pkg,effect,conf in pattern_defs],
                  ["Pattern", "Reliable_source_count", "Effective_problem_type", "Tried_here", "Local_result",
                   "Low_bandwidth_feasible", "Package_need", "Expected_effect", "Confidence"])

    hypotheses_text = [
        ("Top teams exploit longer target history with leakage-safe windows", "GRU-L12 pilot and hydrologic persistence literature", "Pilot is small and selected on same folds", "High", "Canonical is mostly tabular", "Flatten L12 into frozen LightGBM on strict folds"),
        ("Top teams use operational precipitation/reanalysis forcing", "ERA5 pilot and multiple TWS papers", "Global ERA5 is not yet evaluated", "High", "No global forcing", "Small regional or monthly-index ablation"),
        ("They separate known and masked regimes", "Test has missing TWS_t; local dual expert works", "Exact top code unavailable", "High", "Already mostly covered", "Keep dual experts fixed"),
        ("They pool all locations while encoding location", "DrivenData winners used common weights plus site_id", "Different target and only 26 sites", "Medium", "Coordinates may underfit location identity", "Coordinate Fourier basis"),
        ("They validate both time and unseen space", "Challenge robustness goal", "Leaderboard split details inaccessible", "High", "Temporal validation dominates", "Frozen space-time gate"),
        ("They decompose seasonal/trend/residual components", "GRACE-FCast design", "Stable extrapolation can fail under climate change", "Medium", "Partial seasonal features", "OOF-only decomposition ablation"),
        ("They include climate teleconnection indices", "Ethiopia paper and subseasonal forecasting", "Global local effects heterogeneous", "Medium", "No ENSO/NAO/PDO", "Tiny monthly index join"),
        ("They use spatially structured models", "BCNN/MTGNN TWS papers", "Competition coordinates may be sparse and graph costly", "Medium", "Spatial correction is weak", "Cheap neighbor summaries before GNN"),
        ("They ensemble mechanism-diverse models", "Hydrology winners and local blend gain", "Correlated errors limit value", "High", "Canonical and GRU residual r=0.858", "OOF nonnegative blend"),
        ("They preserve TWS persistence as a hard anchor", "TWS has long memory; canonical strength", "Masked rows lack anchor", "High", "Known expert strong", "Delta-vs-absolute expert check"),
        ("They model water-budget changes rather than raw levels", "ERA5 variables and TWS physics", "Measurement/reanalysis errors accumulate", "Medium", "Limited forcing", "Lagged P-ET-runoff proxies"),
        ("They use regularized flattened sequences rather than deep nets", "Tree winners and GRU pilot signal", "Flattening loses recurrent inductive bias", "High", "Not isolated at L12 dual", "CPU LightGBM one controlled config"),
        ("They condition errors on climate/location regimes", "Regional skill differences in literature", "Regime splits reduce samples", "Medium", "Experts only by missingness", "OOF residual clusters with fixed bins"),
        ("They avoid public-LB tuning and optimize robustness", "30/70 split and top-10 review in official spec transcript", "Direct leaderboard data unavailable", "High", "Many submissions create selection risk", "Predeclare gates and one submission"),
        ("They incorporate human-water-use/static basin context", "TWS includes groundwater and human effects", "Free global operational data is difficult", "Low", "Climate alone misses pumping/reservoirs", "Static aridity/basin proxy audit"),
    ]
    hypotheses = df(hypotheses_text, ["Hypothesis", "Evidence", "Counter_evidence", "Confidence", "Our_gap", "How_to_test_cheaply"])

    critique_rows = [
        ["Tabular dependence","Most successful pipeline uses boosted snapshots","High","May miss multi-month storage memory","Flatten 12-month window","Single strict-fold pilot"],
        ["Target-derived prior risk","Prior logic is powerful and repeatedly audited","High","Distribution shift/leakage if cutoff handling slips","Retain row-level provenance assertions","Mutation test"],
        ["External forcing absence","ERA5 extended pilot gained 0.009427 overall","High","Likely leaves causal information unused","Eventually global operational forcing","Pilot already supports"],
        ["Temporal memory gap","GRU-L12 pilot gained 0.076232 over pilot E0","High","Potentially large, but pilot optimism likely","CPU flattened L12 surrogate","Frozen three-fold validation"],
        ["Spatial generalization","Spatial correction transfer score was weak","High","Private spatial shift risk","Space-time acceptance gate","Leave-location-block-out"],
        ["Masked expert","Masked error dominates and improves less reliably","High","Large leaderboard influence","Historical summaries without current TWS","Masked-only RMSE gate"],
        ["Sequence pilot reliability","Only 7,577 rows/two folds; architecture selected on pilot","High","Winner's curse","Full frozen validation","No new architecture search"],
        ["ERA5 pilot scope","Only three regions","High","Geographic transfer unknown","Global data later or more micro-regions","Small cached-region stress test"],
        ["Public LB overfitting","Many candidates and user-reported ranks","High","30% public selection noise","Submission budget and preregistration","Do not submit Notebook34"],
        ["Validation mismatch","Temporal strong; spatial/irregular evidence incomplete","High","Robustness goal may not match local mean RMSE","Worst-fold + masked + space-time gates","Re-score frozen OOF"],
        ["Submission experiment count","Repeated variants increase multiple-testing risk","Medium","False discovery on public LB","One mechanism per submission","Maintain submission ledger"],
        ["Compute/internet constraints","CUDA torch and 8.91GB ERA5 deferred","Medium","Blocks strongest signals, but prevents uncontrolled expansion","CPU surrogate first; cloud only after gate","Measure runtime on one fold"],
    ]
    critique = df(critique_rows,["Critique","Evidence","Severity","Leaderboard_effect","Remedy","Low_cost_test"])

    ideas_raw = [
        ("Flattened L12 LightGBM dual experts","Flatten past 12 observed input rows; known delta and masked absolute","Yes","0","<1 GB","CPU","existing LightGBM","1–2 days","Low","Medium","0.003–0.015 hypothesis","High",95),
        ("HistGradientBoosting flattened L12","Same panel window with sklearn HGB","Yes","0","<1 GB","CPU","existing sklearn","1 day","Low","Medium","0–0.008 hypothesis","Medium",78),
        ("MLPRegressor sequence flattening","Scaled L12 vectors into regularized sklearn MLP","Yes","0","<1 GB","CPU","existing sklearn","1 day","Low","High","0–0.008 hypothesis","Medium",70),
        ("Coordinate Fourier basis","Sin/cos harmonics of lat/lon without learned embeddings","Weakly","0","tiny","CPU","none","hours","Low","Medium","0–0.004 hypothesis","Medium",72),
        ("Fourier temporal basis","Annual/semiannual/trend basis fitted cutoff-safely","Weakly","0","tiny","CPU","none","hours","Low","Low","0–0.003 hypothesis","High",76),
        ("Autoregressive delta windows","Lagged TWS differences and robust summaries","Yes","0","tiny","CPU","none","hours","Medium","Medium","0–0.008 hypothesis","High",83),
        ("Multi-output direct forecasting","Predict several horizons jointly only if labels are cutoff-safe","Potentially","0","medium","CPU/GPU","existing","2–3 days","Medium","High","uncertain","Low",45),
        ("Masked historical summaries","Last-observed, age, rolling robust trend for masked rows","Yes","0","tiny","CPU","none","hours","Medium","Medium","0–0.010 hypothesis","High",87),
        ("GRU pilot distillation","Train CPU tabular model on OOF GRU predictions only","No new external information","0","medium","CPU","existing","1 day","Medium","High","uncertain; can copy pilot bias","Low",52),
        ("Small ONNX sequence runtime","Export/run frozen sequence model efficiently","No","small package","<0.5 GB","CPU/GPU","onnxruntime","1 day","Low","Low","runtime only","Medium",55),
        ("Offline CUDA wheel","Download compatible wheel elsewhere and transfer","No","2.6 GB elsewhere","3 GB","GPU","PyTorch wheel","hours","Low","Low","enables validation, no direct gain","High",74),
        ("Google Colab full GRU","Upload code/data and run frozen GRU validation","No","upload dataset","cloud","GPU","preinstalled PyTorch","hours","Low","Medium","tests 0.003–0.020 hypothesis","High",82),
        ("Kaggle Notebook full GRU","Private dataset + frozen GRU validation","No","upload dataset","cloud","GPU","preinstalled PyTorch","hours","Low","Medium","tests 0.003–0.020 hypothesis","High",86),
        ("ERA5 via alternate network","Complete manifest download outside metered connection","Yes","8.91 GB elsewhere","9+ GB","CPU","existing","days","Low","Medium","0.003–0.015 hypothesis","High",67),
        ("Smaller external climate source","Use point/monthly variables with tiny transfer","Yes","MB-scale","small","CPU","requests optional","1–2 days","Low","Medium","0–0.006 hypothesis","Medium",70),
        ("CHIRPS precipitation","Monthly precipitation aggregated to competition points","Yes","medium unless point API","medium","CPU","xarray optional","2 days","Low","Medium","0–0.008 hypothesis","Medium",65),
        ("NASA POWER monthly","Point/regional monthly climate API","Yes","MB-scale","small","CPU","none","1 day","Low","Medium","0–0.006 hypothesis","Medium",79),
        ("NOAA climate indices","Join globally available monthly indices","Yes","KB","tiny","CPU","none","hours","Low","Medium","0–0.004 hypothesis","Medium",80),
        ("ENSO/NAO/PDO indices","Lagged teleconnection indices and interactions","Yes","KB","tiny","CPU","none","hours","Low","Medium","0–0.005 hypothesis","Medium",84),
        ("Precomputed climate datasets","Use licensed small tables if prediction-time valid","Yes","varies","small","CPU","none","1 day","Medium","Medium","uncertain","Low",58),
        ("OOF regime residual audit","Rank climate/missingness/location failure regimes","No","0","tiny","CPU","none","hours","Low","Low","indirect","High",81),
        ("Robust clipped delta target","Winsorize training delta by fold only","No","0","tiny","CPU","none","hours","Low","Medium","0–0.003 hypothesis","Medium",68),
        ("Latitude-band model calibration","OOF-only affine correction by broad band","No","0","tiny","CPU","none","hours","Medium","High","0–0.002 hypothesis","Low",46),
        ("Nonnegative OOF ensemble","Constrained weights for canonical and flattened sequence","No","0","tiny","CPU","scipy existing","hours","Low","Medium","0–0.005 hypothesis","High",85),
        ("Training-only location climatology","Smoothed month/location baseline with backoff","Weakly","0","tiny","CPU","none","hours","Medium","Medium","0–0.004 hypothesis","Medium",71),
    ]
    ideas = df(ideas_raw,["Name","Technical_description","Carries_new_information","Internet_cost","Disk_cost","Compute_need","Package_need","Development_time","Leakage_risk","Validation_risk","Expected_public_gain","Confidence","Priority_score"])

    compute = df([
        ["Google Colab","Yes, availability not guaranteed","Dynamic; no fixed published free limit","Drive/runtime; limits dynamic","Upload uses user's connection; cloud-side package fetch does not","Usually yes","Yes","Disconnects and fluctuating GPU", "https://research.google.com/colaboratory/faq.html","High"],
        ["Kaggle Notebooks","Yes, P100 documented","About 30 GPU hours/week, can vary","Private datasets supported; exact upload ceiling not verified here","Initial upload uses user's connection","Yes","Best fit","Quota and phone/account verification", "https://www.kaggle.com/docs/efficient-gpu-usage","High"],
        ["Lightning Studio","Credits provide up to ~80 T4 hours/month; pricing-dependent","Free Studio restarts every 4 hours","50 GB listed on free pricing","Initial upload uses user's connection","CUDA/PyTorch images available","Suitable with checkpoints","Availability, credits, forced restarts", "https://lightning.ai/pricing/","High"],
        ["Paperspace/Gradient","No current free GPU entitlement verified","N/A","N/A","Initial upload uses user's connection","Images may include PyTorch","Not recommended without verified credit","Historical free claims may be stale", "https://docs.digitalocean.com/products/paperspace/", "Low"],
        ["Hugging Face Spaces","ZeroGPU free use is quota-limited; hosting eligibility constrained","Free account usage listed as 5 min/day for ZeroGPU use","50 GB ephemeral default","Initial upload uses user's connection","CUDA torch may need configuration","Poor fit for long notebook training","Gradio-only ZeroGPU and short quota", "https://huggingface.co/docs/hub/spaces-zerogpu", "High"],
        ["GitHub Codespaces","No generally available GPU machine verified","N/A","Codespace storage quotas depend on plan","Initial push uses user's connection","No GPU assumption","No","GPU not part of standard offering", "https://docs.github.com/en/codespaces/reference/using-github-codespaces-in-your-organization", "Medium"],
        ["University resources","Institution-dependent","Institution-dependent","Institution-dependent","Could avoid personal quota on campus network","Often available in research clusters","Potentially best if authorized","Access, queue, data policy", "Institution-specific", "Low"],
    ], ["Provider","Free_GPU","Session_limit","Storage","Internet_charged_to_user","PyTorch_ready","Notebook33_fit","Risks","Source","Confidence"])

    tables = {
        "discussion_analysis": discussions, "leaderboard_gap": leaderboard,
        "public_solution_inventory": inventory, "analog_solution_patterns": analogs,
        "technical_pattern_frequency": patterns, "top_solution_hypotheses": hypotheses,
        "current_solution_critique": critique, "low_bandwidth_research_ideas": ideas,
        "free_compute_options": compute,
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)

    plan = """# Notebook 35 plan — CPU-friendly flattened sequence boosting

## Decision
Choose **B) CPU-friendly flattened sequence boosting**. This is the only selected next experiment. It directly tests whether Notebook 33's strong L12 signal is temporal information rather than a GRU-specific effect, requires no download/new framework, and can reuse the canonical dual-expert LightGBM pipeline.

## Main hypothesis
A leakage-safe 12-month flattened panel window, trained as known-delta and masked-absolute LightGBM experts, improves the frozen canonical OOF system and masked RMSE without degrading worst-fold or space-time robustness.

## Required data/packages and internet cost
Existing train/test tables, Notebook 11/25 canonical OOF artefacts, LightGBM/pandas/numpy already installed. Internet cost: zero. No external data and no package installation.

## Implementation
1. Freeze Notebook 11 folds, feature availability rules, seed and baseline predictions.
2. Sort by location/date; construct only past/current-origin L12 features. Never use target or future rows.
3. Add raw lags, differences, missingness/age and robust window summaries; use float32.
4. Train one conservative LightGBM configuration, separately for known-delta and masked-absolute targets.
5. Persist per-fold row-level OOF predictions and provenance assertions.
6. Compare canonical, flattened model, and a nonnegative OOF blend. Do not tune on leaderboard.
7. Run strict temporal, irregular-time and space-time audits before any full-test fit.

## Validation and acceptance gate
Accept only if overall RMSE improves by at least 0.002, masked RMSE does not worsen, worst-fold improves or is within 0.001, every evaluated fold improves in at least one primary regime, and space-time RMSE does not worsen by more than 0.002. Confirm zero future timestamps and fold-training-only statistics.

## Submission policy
Produce a candidate only after all gates pass. Compare IDs/hash/schema to canonical. Make at most one leaderboard submission and record it before seeing the score.

## Stop condition
Stop after the single frozen configuration if overall gain is below 0.002, masked/worst-fold gates fail, runtime is disproportionate, or any provenance assertion fails. Do not start architecture or hyperparameter search.
"""
    recommendation = """# Notebook 34 final recommendation

## Evidence boundary
No public winning solution exists while the challenge is active, and the Zindi discussion/complete leaderboard could not be retrieved as a stable unauthenticated corpus on 2026-08-04. Individual discussion rows and rank scores were therefore not invented. The current RMSE/rank are user-supplied context. Public-solution statements are hypotheses unless tied to a listed source.

## Finding
The most consistent external pattern is not a particular deep architecture: it is hydrologically meaningful lagged state/forcing, pooled spatial learning, honest issue-date validation, and mechanism-diverse ensembles. This repository already has strong boosting and missingness experts. Its two strongest unresolved signals are ERA5-Land forcing and L12 sequence memory, but both full tests are blocked by intentionally deferred downloads.

## One next experiment
Select **B) CPU-friendly flattened sequence boosting**: one frozen L12 LightGBM dual-expert experiment. It costs zero internet, uses installed packages, tests the GRU pilot's core information hypothesis, and avoids treating the optimistic pilot as leaderboard evidence. Do not select cloud GRU first: it validates an architecture but needs dataset upload and has a larger pilot-selection risk. Do not select a small climate index first: evidence is useful but its likely effect is weaker and spatially heterogeneous.

## Confidence
High confidence that this is the cheapest decisive test; medium confidence it improves public RMSE. Expected-gain ranges in the ideas table are prioritization hypotheses, never measured results or promises.
"""
    (OUT / "notebook35_plan.md").write_text(plan, encoding="utf-8")
    (OUT / "final_recommendation.md").write_text(recommendation, encoding="utf-8")
    return tables


def build_notebook() -> None:
    cells = [
        {"cell_type":"markdown","metadata":{},"source":["# 34 — Competitor solution reverse engineering\n",
         "Research-only audit. No model is trained, no external dataset/package is downloaded, and no submission is created. Claims are separated into verified evidence, repository evidence, user-supplied context, and hypotheses. Research access date: 2026-08-04.\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Method and evidence hierarchy\n",
         "Primary official pages and peer-reviewed papers outrank blogs; official winner write-ups outrank participant hints. The active competition has no public winning solution. Zindi discussion and complete leaderboard were not stably accessible without authentication, so missing content remains missing. Notebook 27's transcribed specification is used only as repository evidence.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
         "from pathlib import Path\nimport sys, pandas as pd\nsys.path.insert(0, str(Path('../scripts').resolve()))\nfrom build_notebook34 import generate_artifacts, OUT\ntables = generate_artifacts()\nprint('Generated', len(tables)+2, 'required artifacts in', OUT)\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
         "# Repository audit: verify the requested notebook range without modifying it.\nrepo = Path('..').resolve()\nnotebooks = sorted(repo.joinpath('notebooks').glob('*.ipynb'))\nnumbered = {int(p.name[:2]): p.name for p in notebooks if p.name[:2].isdigit() and 1 <= int(p.name[:2]) <= 33}\nprint('Notebook coverage:', len(numbered), '/ 33')\nprint('Missing:', sorted(set(range(1,34))-set(numbered)))\nfor n in [27,31,32,33]: print(n, numbered.get(n))\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Competition discussion and leaderboard gap\n",
         "The following tables preserve the access limitation. Blank leaderboard scores mean unavailable, not zero. No score density is calculated without rank-point observations.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["display(tables['discussion_analysis'])\ndisplay(tables['leaderboard_gap'])\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Public solutions and analogous hydrology evidence\n",
         "Direct competition repositories/notebooks were not verified by exact-title/column searches. The inventory therefore includes the official specification plus genuinely related TWS and hydrology sources; transferability is not equivalence.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["display(tables['public_solution_inventory'])\ndisplay(tables['analog_solution_patterns'])\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Technical patterns and top-solution hypotheses\n",
         "Source counts are conservative counts within the curated inventory, not bibliometric frequencies. Every top-team statement is explicitly a hypothesis because private code is unavailable.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["display(tables['technical_pattern_frequency'])\ndisplay(tables['top_solution_hypotheses'])\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Hard critique and low-bandwidth options\n",
         "Expected public-gain ranges are subjective prioritization hypotheses. They are not validation results. Priority favors evidence, zero bandwidth, speed, compatibility and leakage safety.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["display(tables['current_solution_critique'])\ndisplay(tables['low_bandwidth_research_ideas'].sort_values('Priority_score', ascending=False))\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Free compute options\n",
         "Service limits change. Rows are dated 2026-08-04 and link to provider documentation; institution-specific availability remains unknown. Uploading local data still consumes the user's network, while runtime package traffic occurs in the provider environment.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["display(tables['free_compute_options'])\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## Final selection\n",
         "Only **B) CPU-friendly flattened sequence boosting** is selected. It isolates L12 memory using the existing LightGBM stack with zero download. The exact Notebook 35 protocol, gates, submission policy and stop condition are written to `artifacts/notebook34/notebook35_plan.md`; the evidence-qualified conclusion is in `final_recommendation.md`.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
         "required=['discussion_analysis.csv','leaderboard_gap.csv','public_solution_inventory.csv','analog_solution_patterns.csv','technical_pattern_frequency.csv','top_solution_hypotheses.csv','current_solution_critique.csv','low_bandwidth_research_ideas.csv','free_compute_options.csv','notebook35_plan.md','final_recommendation.md']\nmissing=[x for x in required if not (OUT/x).exists()]\nassert not missing, missing\nassert len(tables['top_solution_hypotheses']) >= 15\nassert len(tables['low_bandwidth_research_ideas']) >= 25\nassert not list(Path('../submissions').glob('submission_34*.csv'))\nprint('All required outputs verified; no Notebook 34 submission exists.')\nprint((OUT/'final_recommendation.md').read_text(encoding='utf-8'))\n"]},
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"nb34-{i:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3"}}, "nbformat":4, "nbformat_minor":5}
    NB.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    generate_artifacts()
    build_notebook()
    print(NB)
