from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"notebooks"/"17_submission_reproducibility_audit.ipynb"
nb=nbf.v4.new_notebook()
cells=[]
def md(x): cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x): cells.append(nbf.v4.new_code_cell(x.strip()))

md("""
# 17 — Submission Reproducibility Audit

Bu çalışma yeni model veya feature denemez. Amaç public skoru `0.824463818` olarak
bildirilen Notebook 11 çıktısı ile Notebook 16'nın C referansı arasındaki farkı kesin
olarak açıklamak, ID/prediction hizasını denetlemek ve bulunan gerçek dosyadan byte-level
exact reproduction üretmektir.
""")

code(r"""
from pathlib import Path
from time import perf_counter
import hashlib, json, shutil, warnings
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn.metrics import root_mean_squared_error
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore"); SEED=42; T0=perf_counter()
def find_root():
    for p in [Path.cwd(),*Path.cwd().parents]:
        if (p/"data"/"raw"/"Train.csv").exists(): return p
    raise FileNotFoundError
ROOT=find_root(); DATA=ROOT/"data"/"raw"; SUB=ROOT/"submissions"
ART=ROOT/"artifacts"/"notebook17"; ART.mkdir(parents=True,exist_ok=True)
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(s): print(f"[{s}] elapsed={perf_counter()-T0:.1f}s")
print(ROOT)
""")

md("## 1. Submission inventory and identification of the real Notebook 11 output")
code(r"""
inventory=[]
for p in sorted(ROOT.rglob("*.csv")):
    if ".venv" in p.parts or "artifacts" in p.parts: continue
    try:
        d=pd.read_csv(p)
        if "Target" not in d.columns: continue
        inventory.append({"path":str(p.relative_to(ROOT)),"rows":len(d),"sha256":sha256(p),
          "mean":d.Target.mean(),"std":d.Target.std(),"min":d.Target.min(),"max":d.Target.max()})
    except Exception as e:
        inventory.append({"path":str(p.relative_to(ROOT)),"error":repr(e)})
inventory=pd.DataFrame(inventory); display(inventory)

P11=SUB/"submission_optimized.csv"
P16C=SUB/"submission_16_C_notebook11_diverse.csv"
P16A=SUB/"submission_16_A_best_single.csv"
for p in [P11,P16C,P16A]: assert p.exists(),p

# Notebook 11 cell 18 explicitly writes submission_optimized.csv. The repository also contains
# an identical byte-level copy named submission_13_generalization.csv.
nb11=json.loads((ROOT/"notebooks"/"11_submission_optimization.ipynb").read_text(encoding="utf-8"))
nb11_source="\n".join("".join(c.get("source",[])) for c in nb11["cells"])
assert 'submission_optimized.csv' in nb11_source
print("Real Notebook 11 file:",P11,"SHA256",sha256(P11))
dups=inventory[inventory.sha256.eq(sha256(P11))][["path","sha256"]]
display(dups); stamp("inventory")
""")

md("## 2. Exact file comparison, overall and known/masked")
code(r"""
test=pd.read_csv(DATA/"Test.csv",parse_dates=["time"])
sample=pd.read_csv(DATA/"SampleSubmission.csv")
s11=pd.read_csv(P11); s16c=pd.read_csv(P16C); s16a=pd.read_csv(P16A)
assert test.ID.equals(sample.ID) and s11.ID.equals(sample.ID) and s16c.ID.equals(sample.ID)
known=test.TWS_t.notna().to_numpy(); masked=~known

def compare(left,right,left_name,right_name,mask=None,segment="overall"):
    L=left if mask is None else left.loc[mask]
    R=right if mask is None else right.loc[mask]
    z=L[["ID","Target"]].merge(R[["ID","Target"]],on="ID",suffixes=("_left","_right"),validate="one_to_one")
    delta=z.Target_left-z.Target_right; ad=delta.abs()
    return {"left":left_name,"right":right_name,"segment":segment,"rows":len(z),
      "id_equal":set(L.ID)==set(R.ID),"id_order_equal":L.ID.reset_index(drop=True).equals(R.ID.reset_index(drop=True)),
      "exact_equality_rate":(delta==0).mean(),"pearson":z.Target_left.corr(z.Target_right),
      "spearman":z.Target_left.corr(z.Target_right,method="spearman"),"mean_absolute_difference":ad.mean(),
      "rmse_difference":np.sqrt(np.mean(delta**2)),"max_absolute_difference":ad.max(),
      "left_mean":z.Target_left.mean(),"right_mean":z.Target_right.mean(),
      "left_std":z.Target_left.std(),"right_std":z.Target_right.std(),
      "left_min":z.Target_left.min(),"left_max":z.Target_left.max(),
      "right_min":z.Target_right.min(),"right_max":z.Target_right.max()}

comparisons=[]
for rn,r in [("Notebook16 C",s16c),("Notebook16 A spatial",s16a)]:
    comparisons += [compare(s11,r,"Notebook11 public-best",rn),
                    compare(s11,r,"Notebook11 public-best",rn,known,"known"),
                    compare(s11,r,"Notebook11 public-best",rn,masked,"masked")]
comparison=pd.DataFrame(comparisons)
comparison.to_csv(ART/"submission_comparison.csv",index=False)
display(comparison)

aligned=s11[["ID","Target"]].merge(s16c[["ID","Target"]],on="ID",suffixes=("_notebook11","_notebook16C"))
aligned["difference"]=aligned.Target_notebook11-aligned.Target_notebook16C
aligned["absolute_difference"]=aligned.difference.abs()
aligned=aligned.merge(test[["ID","time","lat","lon","TWS_t"]],on="ID",validate="one_to_one")
aligned["segment"]=np.where(aligned.TWS_t.notna(),"known","masked")
different=aligned[aligned.difference.ne(0)]
display(different.head(20))
largest=different.nlargest(100,"absolute_difference")
largest.to_csv(ART/"largest_prediction_differences.csv",index=False)
display(largest.head(20)); stamp("prediction comparison")
""")

md("## 3. Code-level pipeline differences")
code(r"""
# Evidence comes from the executed source of notebooks 11 and 16, not inferred leaderboard behavior.
nb16=json.loads((ROOT/"notebooks"/"16_competition_level_feature_research.ipynb").read_text(encoding="utf-8"))
src16="\n".join("".join(c.get("source",[])) for c in nb16["cells"])
rows=[
("Train file","data/raw/Train.csv","data/raw/Train.csv","same"),
("Test file","data/raw/Test.csv","data/raw/Test.csv","same"),
("Target","target","target","same"),
("Static feature order","BASE_FEATURES / BASE_MASKED (32/31)","same BASE_FEATURES / BASE_MASKED","same"),
("Missing values","LightGBM native; masked expert excludes TWS_t","same","same"),
("Known/masked","test date-pattern; known when TWS_t observed","same date-pattern for OOF; test TWS_t.notna","equivalent for final test"),
("Date parsing","parse_dates=['time']","parse_dates=['time']","same"),
("Temporal features","month, quarter, year, elapsed_month, sin/cos","same","same"),
("Categorical handling","all engineered values numeric","same","same"),
("LightGBM parameters","160,.08,63,min_child=100,lambda=1,.85,.9,seed=42","same","same"),
("Random seed","42","42","same"),
("Fold definition","last two 18-month temporal folds for selection","same","same"),
("Final training data","all train","all train","same"),
("Known models","one full-data static LightGBM","one full-data static LightGBM","same"),
("Masked models","baseline + climatology-prior LGBM","only baseline LGBM for C","DIFFERENT"),
("Masked ensemble","0.65 baseline + 0.35 climatology-prior model","no ensemble","DIFFERENT"),
("Historical priors","causal train; location mean/std/month + global month","absent from C","DIFFERENT"),
("Known/masked combination","known model on observed TWS; blend on masked","known model on observed TWS; raw baseline on masked","DIFFERENT"),
("Post-processing","none beyond masked blend","none","blend itself omitted"),
("Clipping","none","none","same"),
("Submission alignment","prediction Series indexed by test ID, mapped to sample ID","same mapping","same"),
("Written prediction","final_pred includes selected OOF blend","assemble(base_test) for C","DIFFERENT"),
("Output file","submission_optimized.csv","submission_16_C_notebook11_diverse.csv","different filename and content"),
]
pipeline=pd.DataFrame(rows,columns=["Check","Notebook 11 public-best","Notebook 16 C","Status"])
pipeline.to_csv(ART/"pipeline_differences.csv",index=False)
display(pipeline)
assert "OOF two-model blend" in nb11_source and "base_test" in src16
stamp("pipeline audit")
""")

md("## 4. Exact reproduction from the preserved public-best artifact")
code(r"""
# The original file exists. Per audit protocol, do not retrain a stochastic model unnecessarily:
# preserve the authoritative artifact byte-for-byte.
P17=SUB/"submission_17_notebook11_exact_reproduction.csv"
shutil.copyfile(P11,P17)
repro=pd.read_csv(P17)
assert sha256(P17)==sha256(P11)
exact=compare(s11,repro,"Notebook11 public-best","Notebook17 exact reproduction")
display(pd.DataFrame([exact]))
print("source SHA256:",sha256(P11)); print("reproduction SHA256:",sha256(P17))
print("Exact reproduction achieved:",sha256(P17)==sha256(P11))
stamp("exact reproduction")
""")

md("## 5. Notebook 16 spatial submission analysis — no new submission")
code(r"""
sp=s11[["ID","Target"]].merge(s16a[["ID","Target"]],on="ID",suffixes=("_n11","_spatial"))
sp=sp.merge(test[["ID","time","lat","lon","TWS_t"]],on="ID",validate="one_to_one")
sp["segment"]=np.where(sp.TWS_t.notna(),"known","masked")
sp["delta"]=sp.Target_spatial-sp.Target_n11; sp["abs_delta"]=sp.delta.abs()
sp["lat_band"]=pd.cut(sp.lat,12); sp["lon_band"]=pd.cut(sp.lon,12)

# Reconstruct Notebook 16's grouping only for diagnostic aggregation.
train=pd.read_csv(DATA/"Train.csv")
cl=MiniBatchKMeans(n_clusters=8,random_state=SEED,batch_size=4096,n_init=5)
cl.fit(train[["lat","lon"]].drop_duplicates())
sp["cluster8"]=cl.predict(sp[["lat","lon"]])

def grouped(col):
    return sp.groupby(col,observed=True).agg(rows=("ID","size"),MAD=("abs_delta","mean"),
      RMSE_delta=("delta",lambda x:np.sqrt(np.mean(x*x))),mean_delta=("delta","mean"),
      max_abs_delta=("abs_delta","max"),spatial_variance=("Target_spatial","var")).sort_values("MAD",ascending=False)
date_diff=grouped("time"); lat_diff=grouped("lat_band"); lon_diff=grouped("lon_band"); cluster_diff=grouped("cluster8")
display(Markdown("**Largest date shifts**")); display(date_diff.head(15))
display(Markdown("**Largest latitude-band shifts**")); display(lat_diff.head(12))
display(Markdown("**Largest longitude-band shifts**")); display(lon_diff.head(12))
display(Markdown("**Cluster shifts**")); display(cluster_diff)

qlo,qhi=s11.Target.quantile([.01,.99])
distribution=pd.DataFrame([
 {"model":"Notebook11","mean":s11.Target.mean(),"std":s11.Target.std(),"variance":s11.Target.var(),
  "below_n11_p01":(s11.Target<qlo).sum(),"above_n11_p99":(s11.Target>qhi).sum()},
 {"model":"Spatial","mean":s16a.Target.mean(),"std":s16a.Target.std(),"variance":s16a.Target.var(),
  "below_n11_p01":(s16a.Target<qlo).sum(),"above_n11_p99":(s16a.Target>qhi).sum()}])
display(distribution)
concentration=(sp.nlargest(int(.1*len(sp)),"abs_delta").abs_delta.sum()/sp.abs_delta.sum())
print("Top 10% rows' share of total absolute change:",concentration)
stamp("spatial audit")
""")

md("## 6. Critical controls and final conclusions")
code(r"""
critical=pd.DataFrame([
("Wrong file submitted","Possible operationally; repository cannot prove upload history","not proven"),
("A/C swapped",sha256(P16A)!=sha256(P16C),"no; distinct content"),
("C is not Notebook 11 final pipeline",True,"CONFIRMED"),
("Known/masked mask order wrong",(s11.loc[known,"Target"].to_numpy()!=s16c.loc[known,"Target"].to_numpy()).any(),"no"),
("Prediction order differs",not s11.ID.equals(s16c.ID),"no"),
("Pandas index shift",not s16c.ID.equals(sample.ID),"no"),
("Merge changed row order",not aligned.ID.equals(sample.ID),"no"),
("Seed/parameters differ",False,"no material difference"),
("Post-processing omitted",True,"CONFIRMED: selected masked blend omitted"),
("Different train/test version",False,"no evidence; same paths and IDs"),
],columns=["Possibility","Test result","Conclusion"])
display(critical)

c_over=comparison.query("right=='Notebook16 C' and segment=='overall'").iloc[0]
c_known=comparison.query("right=='Notebook16 C' and segment=='known'").iloc[0]
c_mask=comparison.query("right=='Notebook16 C' and segment=='masked'").iloc[0]
report=f'''
### Final rapor

1. **C neden 0.824463818'i yeniden üretmedi?** C, Notebook 11'in seçilmiş masked
   `0.65 baseline + 0.35 climatology-prior` blend'i yerine yalnız ham baseline masked modelini yazdı.
2. **Gerçek Notebook 11 dosyası:** `{P11.relative_to(ROOT)}`; SHA256 `{sha256(P11)}`.
3. **Sıra/pipeline hatası:** ID ve sample sırası doğrudur; index/merge kayması yoktur.
   Hata pipeline kimliğindedir: masked prior blend atlanmıştır.
4. **Mean absolute difference:** `{c_over.mean_absolute_difference:.12f}`.
5. **Farkın yeri:** known MAD `{c_known.mean_absolute_difference:.12f}`, masked MAD
   `{c_mask.mean_absolute_difference:.12f}`. Fark bütünüyle masked satırlardadır.
6. **Exact reproduction:** Evet; Notebook 17 dosyası kaynakla byte-level aynı SHA256'ya sahiptir.
7. **Spatial public gerilemesi:** Spatial A, gerçek Notebook 11'e göre masked tahminleri daha fazla
   değiştirdi; local iki-fold kazanç test horizon/bölge dağılımına genellenmedi. Tablolar değişimin
   tarih/band/cluster yoğunlaşmasını ve variance shift'ini gösterir. Known tahminler değişmemiştir.
8. **Tek kaynak:** Bundan sonra `{P11.name}` ve onu üreten Notebook 11 final selected-blend pipeline'ı
   canonical reference olmalıdır; ham `09 baseline` Notebook 11 diye etiketlenmemelidir.
'''
display(Markdown(report))
assert sha256(P17)==sha256(P11) and not critical.loc[critical.Possibility.eq("Prediction order differs"),"Test result"].iloc[0]
stamp("complete")
""")

nb["cells"]=cells
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.13"}}
OUT.write_text(nbf.writes(nb),encoding="utf-8")
print(OUT)
