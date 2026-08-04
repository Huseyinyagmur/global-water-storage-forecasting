from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"notebooks/35_flattened_sequence_boosting.ipynb"

def build():
    cells=[
      {"cell_type":"markdown","metadata":{},"source":["# 35 — Flattened Sequence Boosting\n","This notebook tests one hypothesis only: whether leakage-free L12 history supplied as flattened tabular features can reproduce the recorded Notebook 33 GRU signal. LightGBM family, dual known/masked experts, canonical parameters, seed and target remain fixed. No external data/package download is permitted.\n"]},
      {"cell_type":"markdown","metadata":{},"source":["## Frozen gates and leakage contract\n","Before results: A requires strict overall gain ≥0.002 with no worst-fold degradation; B masked gain ≥0.004 with known degradation ≤0.001; C blend gain ≥0.0015 with strict/space-time agreement; D reproduces ≥60% of recorded GRU gain and passes robust full validation. Every lag is location-wise `shift(k)` and must have source date strictly earlier than prediction date. Same-date values, validation targets, target priors, teacher forcing and future rows are forbidden. Full execution stops if estimated runtime exceeds 30 minutes or peak RAM exceeds 8 GB.\n"]},
      {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["%run ../scripts/notebook35_runtime.py\n"]},
      {"cell_type":"markdown","metadata":{},"source":["## Results\n","Notebook 33 GRU entries are recorded pilot results, not recomputed observations. `full_validation_results.csv` explicitly distinguishes computed metrics from stopped work; missing full metrics are never inferred.\n"]},
      {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Markdown\nNB_ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'data/raw/Train.csv').exists()); NB_ART=NB_ROOT/'artifacts/notebook35'\nfor name in ['window_definition','lag_feature_report','cold_start_analysis','pilot_results','full_validation_results','diversity_analysis','gru_comparison','feature_importance','submission_report']:\n display(Markdown('### '+name)); display(pd.read_csv(NB_ART/f'{name}.csv').head(30))\n"]},
      {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["required=['window_definition.csv','lag_feature_report.csv','cold_start_analysis.csv','pilot_results.csv','full_validation_results.csv','diversity_analysis.csv','gru_comparison.csv','feature_importance.csv','submission_report.csv','final_recommendation.md']\nassert not [x for x in required if not (NB_ART/x).exists()]\nassert not list((NB_ROOT/'submissions').glob('submission_35*.csv'))\ndisplay(Markdown((NB_ART/'final_recommendation.md').read_text(encoding='utf-8')))\nprint('All required outputs verified; no submission created.')\n"]},
    ]
    for i,c in enumerate(cells): c['id']=f'nb35-{i:02d}'
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
    NB.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8'); print(NB)
if __name__=='__main__': build()
