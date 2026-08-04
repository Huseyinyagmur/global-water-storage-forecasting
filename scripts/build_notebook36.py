import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; NB=ROOT/'notebooks/36_f1_l12_frozen_full_validation.ipynb'
def build():
 cells=[
  {'cell_type':'markdown','metadata':{},'source':['# 36 — F1-L12 Frozen Full Validation\n','This notebook performs no search. F1-L12 means canonical LightGBM dual experts plus raw location-wise lags 1–12 only. Parameters, seed=42, native missing-value policy, cold-start policy and candidate blend weight 0.05 are frozen. Spatial serious-degradation tolerance is fixed at 0.003 before results.\n']},
  {'cell_type':'markdown','metadata':{},'source':['## Fail-fast and causality contract\n','The Notebook 35 pilot must reproduce before full work. Each fold enforces disjoint dates and `train max date < validation min date`. Raw covariate/TWS observations may enter only through location-wise `shift(k)`; targets, target priors, same-date rows, future rows, scaling and imputation are absent. Each completed fold is checkpointed with SHA256.\n']},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['%run ../scripts/notebook36_runtime.py\n']},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display,Markdown\nR=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'data/raw/Train.csv').exists()); A=R/'artifacts/notebook36'\nfor n in ['reproduction_gate','leakage_audit','fold_results','validation_summary','blend_results','diversity_analysis','stability_analysis','performance_report','submission_report']:\n display(Markdown('### '+n)); display(pd.read_csv(A/f'{n}.csv').head(50))\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["req=['reproduction_gate.csv','leakage_audit.csv','fold_results.csv','validation_summary.csv','blend_results.csv','diversity_analysis.csv','stability_analysis.csv','performance_report.csv','submission_report.csv','final_recommendation.md']\nassert not [x for x in req if not (A/x).exists()]\nassert pd.read_csv(A/'leakage_audit.csv').All_passed.all()\ndisplay(Markdown((A/'final_recommendation.md').read_text(encoding='utf-8')))\n"]}
 ]
 for i,c in enumerate(cells):c['id']=f'nb36-{i:02d}'
 NB.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=1),encoding='utf-8'); print(NB)
if __name__=='__main__':build()
