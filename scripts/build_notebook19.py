import nbformat as nbf
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"notebooks"/"19_challenge_technical_specification.ipynb"
nb=nbf.v4.new_notebook()
c=[]
def md(x): c.append(nbf.v4.new_markdown_cell(x))
def code(x): c.append(nbf.v4.new_code_cell(x))

md("""# 19 — Challenge Technical Specification and Research Audit

## Koddan önce: kısa teknik şartname özeti

Bu challenge sıradan bir i.i.d. tabular regresyon problemi değildir. Resmî amaç, GRACE tabanlı
**global Total Water Storage (TWS)** değerini en son kullanılabilir GRACE gözleminden bir adım
sonrası için tahmin ederek 2–3 aylık ürün gecikmesini azaltmaktır. TWS; groundwater, soil moisture,
surface water ve snow bileşenlerini birlikte temsil eden entegre bir storage state'idir.

“Robustness across space and time” bu nedenle yalnız düşük ortalama RMSE değildir:

1. farklı iklim/hidroloji rejimlerinde ve coğrafyalarda hata kararlılığı,
2. tarihsel dönemin sonrasına gerçek temporal extrapolation,
3. `TWS_t` sağlanan ve kasıtlı maskelenen prediction rejimlerinde çalışma,
4. operasyon anında bir ay içinde gerçekten bulunabilen covariate'larla tekrar üretilebilirlik,
5. bias, şeffaflık, verimlilik ve pratik kullanım

birlikte değerlendirilmelidir.

Canonical referans değiştirilmeyecektir: Notebook 11 final selected blend,
`submissions/submission_optimized.csv`, public RMSE `0.824463818`.

İlk karar: **yeni model kodu eklemek için henüz gerekçe yoktur.** Önce veri sözleşmesi,
prediction-time availability, leakage yüzeyi ve validation–deployment uyumu denetlenir.

Kaynaklar:

- [Resmî challenge açıklaması ve kurallar](https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge)
- Repository `StarterNotebook.ipynb`
- Repository notebook'ları 01–18; özellikle 08, 11, 17 ve 18
""")

md("""## 1. Challenge description → uygulanabilir teknik sözleşme

| Şart | Teknik anlamı | Pipeline sonucu |
|---|---|---|
| Target | Satır zamanı `t` için bir sonraki TWS, yani `TWS(t+1)` | One-step global regression |
| Prediction time | Zaman indeksi gerçek bugünden değil, en son kullanılabilir GRACE gözleminden tanımlanır | Real-time görülemeyen GRACE/TWS kullanılmamalı |
| TWS masking | Test `TWS_t` değerlerinin yaklaşık üçte ikisi kasıtlı NaN | Missingness doğal eksiklik değil, leakage önleme protokolü |
| Covariates | Provided SPEI, soil moisture, seasonal encoding; dış veri yalnız operasyonel olarak ≤1 ayda mevcutsa | Availability her feature için belgelenmeli |
| Metric | RMSE | Büyük hatalar kuadratik cezalandırılır; uç rejimler önemlidir |
| Robustness | Global space/time adaptability ve practicality | Mean RMSE yanında region/time/mask stress testleri gerekir |
| Reproducibility | Seed, çalışan kod, açık kaynak araçlar; AutoML yasak | Artifact kimliği ve deterministic refit zorunlu |
| Evaluation | Public yaklaşık %30, private yaklaşık %70 | Public leaderboard model seçimi için güvenilir tek validation değildir |

Fiziksel yorum: TWS bir flux değil, bütünleşik bir **state/anomaly** değişkenidir. Soil moisture bu
state'in yalnız sığ ve hızlı bileşenini; SPEI meteorolojik kuraklık/zorlamayı; ay ve koordinatlar
mevsimsellik/coğrafyayı temsil eder. Groundwater, rivers/lakes/reservoirs, snow/ice, canopy,
irrigation, pumping, dam operations, runoff, evapotranspiration, geology/topography ve basin
connectivity doğrudan temsil edilmez.
""")

code("""from pathlib import Path
import hashlib, json, re
import numpy as np, pandas as pd, nbformat
from sklearn.model_selection import TimeSeriesSplit
ROOT=Path.cwd().resolve()
if not (ROOT/"data/raw/Train.csv").exists(): ROOT=ROOT.parent
DATA=ROOT/"data/raw"; NBDIR=ROOT/"notebooks"; SUB=ROOT/"submissions"
print("ROOT",ROOT)
""")

md("## 2. 01–18 notebook envanteri ve araştırma izi")
code("""rows=[]
for i in range(1,19):
    matches=sorted(NBDIR.glob(f"{i:02d}_*.ipynb"))
    for p in matches:
        n=nbformat.read(p,4)
        src="\\n".join(x.source for x in n.cells)
        rows.append({"No":i,"Notebook":p.name,"Cells":len(n.cells),
          "Code":sum(x.cell_type=="code" for x in n.cells),
          "Executed":sum(x.cell_type=="code" and x.execution_count is not None for x in n.cells),
          "Temporal":bool(re.search("temporal|TimeSeriesSplit|fold",src,re.I)),
          "Known/masked":bool(re.search("known|masked|TWS_t_masked",src,re.I)),
          "Target-derived history":bool(re.search("prior_|expanding|rolling|shift\\(",src,re.I)),
          "Submission":bool(re.search("to_csv|submission",src,re.I))})
inventory=pd.DataFrame(rows)
display(inventory)
assert set(range(1,19)).issubset(set(inventory.No))
""")

md("""### Notebook araştırma özeti

- `01–07`: EDA, baseline, boosting ve ensemble evrimi; çoğu erken yaklaşım prediction-time
  mask rejimini ve gerçek temporal extrapolation'ı tam temsil etmez.
- `08`: zaman sıralı validation'a geçiş için doğru motivasyon, fakat satır-bazlı split aynı ayı
  train ve validation'a böler.
- `09–10`: dual known/masked uzman, güçlü static features ve explainability/residual analizi.
- `11`: canonical selected blend; masked baseline ile causal climatology-prior modelini
  `0.65/0.35` OOF ağırlığıyla birleştirir.
- `12–15`: daha düşük local skorların public'e taşınmaması, local-selection overfit riskini
  somutlaştırır.
- `16`: geniş feature araştırması; artifact kimliği karışıklığı oluşmuştur.
- `17`: canonical dosyayı SHA256 ile kesin tanımlar ve Notebook 16 C'nin selected masked blend'i
  atladığını kanıtlar.
- `18`: fiziksel feature ablation; `drying_transition` tek başına OOF kazanç sağlasa da birleşik
  feature seti kötüleşir. “Daha çok domain feature” otomatik olarak daha iyi değildir.
""")

md("## 3. Veri ve prediction-time availability sözleşmesi")
code("""train=pd.read_csv(DATA/"Train.csv",parse_dates=["time"])
test=pd.read_csv(DATA/"Test.csv",parse_dates=["time"])
sample=pd.read_csv(DATA/"SampleSubmission.csv")
contract=pd.DataFrame([
 ("TWS_t","GRACE-derived current state","Yalnız NaN olmayan test satırında","Known uzman güvenli; masked uzman kesinlikle kullanmamalı"),
 ("SPEI_01/03/06/12_t","Multi-scale precipitation–PET anomaly","Provided at prediction time","Meteorolojik forcing; storage component değil"),
 ("SOIL_MOISTURE_t","Near-surface soil state","Provided at prediction time","TWS'nin yalnız sığ bileşeni"),
 ("month_sin/cos","Season","Deterministic","Güvenli"),
 ("lat/lon","Location","Static","Güvenli; target encoding ayrı denetlenmeli"),
 ("train target history","Past TWS(t+1) labels","Sadece forecast origin öncesi","Fold ve row-time causality zorunlu"),
],columns=["Feature group","Physical role","Availability","Decision"])
display(contract)
print("Train",train.shape,train.time.min(),train.time.max(),train.time.nunique())
print("Test ",test.shape,test.time.min(),test.time.max(),test.time.nunique())
mask_by_date=test.groupby("time").agg(rows=("ID","size"),known=("TWS_t","count"),masked_flag=("TWS_t_masked","mean"))
display(mask_by_date)
print("Known/masked rows",test.TWS_t.notna().sum(),test.TWS_t.isna().sum())
assert test.TWS_t.isna().equals(test.TWS_t_masked.astype(bool))
""")

md("""### Availability kararı

Canonical final inference yolu **provided-data düzeyinde** kurala uyar:

- observed `TWS_t` satırları known modele,
- masked satırlar `TWS_t` içermeyen modele,
- historical priors yalnız train target geçmişinden,
- dış veri kullanılmadan

tahmin edilir. Test target, recursive pseudo-label veya gelecekteki test satırı kullanılmaz.

Ancak “tamamen sağlanıyor” demek için iki çekince vardır:

1. Validation, testin düzensiz calendar gap yapısını tam simüle etmez.
2. Target-derived training feature'larının her training satırının kendi forecast origin'ine göre
   causal olması gerekir; fold train içinde bulunmak tek başına yeterli değildir.
""")

md("## 4. Validation denetimi: challenge hedefiyle uyum")
code("""# Notebook 08'in gerçek split sınırlarını yeniden denetle.
t=train[["time"]].sort_values("time").reset_index(drop=True)
split_rows=[]
for fold,(a,b) in enumerate(TimeSeriesSplit(5).split(t),1):
    split_rows.append({"fold":fold,"train_end":t.loc[a[-1],"time"],"valid_start":t.loc[b[0],"time"],
      "valid_end":t.loc[b[-1],"time"],"same_date_boundary":t.loc[a[-1],"time"]==t.loc[b[0],"time"]})
split08=pd.DataFrame(split_rows)
display(split08)

# Notebook 11'in kodundan fold ve blend sözleşmesini doğrula.
n11=nbformat.read(NBDIR/"11_submission_optimization.ipynb",4)
s11="\\n".join(x.source for x in n11.cells)
checks=pd.DataFrame([
 ("08 row TimeSeriesSplit",split08.same_date_boundary.all(),"REJECT: every boundary splits one date across train/valid"),
 ("11 expanding temporal blocks","for i in range(2)" in s11,"ACCEPT as canonical local reference"),
 ("11 test-mask pattern","KNOWN_POS" in s11,"Useful deployment mimic; ordinal/calendar mismatch remains"),
 ("11 causal location priors","shift(1)" in s11 and "expanding()" in s11,"Location history is row-causal"),
 ("11 selected masked blend","0.65" in s11 and "0.35" in s11,"Canonical final identity"),
],columns=["Check","Observed","Interpretation"])
display(checks)
""")

md("""### Validation sonucu

`08` challenge hedefiyle uyumlu değildir: aynı tarih bazı spatial satırlarda train, diğerlerinde
validation olur. Bu, global aynı-ay sinyalini validation sınırından geçirir ve gerçek gelecek
extrapolation değildir.

Notebook 11'in expanding-window, date-block ve test-benzeri known/masked routing yaklaşımı mevcut
seçenekler arasında daha uygundur. Fakat “zaten optimal” olduğu kanıtlanmış değildir:

- Test yalnız 18 düzensiz tarih içerir ve 2015-09…2018-12 aralığına yayılır.
- Canonical foldlar 24 observed-date bloklarıdır; `KNOWN_POS` ordinal olarak aktarılır.
- Calendar horizon/gap, GRACE outage ve son dönem drift'i tam eşleşmez.
- Spatial robustness ayrı unseen-region holdout ile ölçülmez.

Bu nedenle gereksiz model karmaşıklığı yerine önce **origin-aligned irregular-date hindcast** ve
**blocked spatial stress test** eklenmelidir. Canonical seçimi değiştirmek için bu testlerde de
kazanım şart olmalıdır.
""")

md("## 5. Canonical Notebook 11: kimlik, varsayımlar ve leakage yüzeyi")
code("""p=SUB/"submission_optimized.csv"
raw=p.read_bytes(); canonical=pd.read_csv(p)
sha=hashlib.sha256(raw).hexdigest()
artifact=pd.DataFrame([{
 "path":str(p.relative_to(ROOT)),"sha256":sha,"rows":len(canonical),
 "columns":list(canonical.columns),"id_unique":canonical.ID.is_unique,
 "id_set_ok":set(canonical.ID.astype(str))==set(test.ID.astype(str)),
 "finite":np.isfinite(canonical.Target).all(),"mean":canonical.Target.mean(),
 "std":canonical.Target.std(),"min":canonical.Target.min(),"max":canonical.Target.max()}])
display(artifact.T)
assert sha=="b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968"
""")

md("""Notebook 11'in temel varsayımları:

1. Aynı global LightGBM yapısı farklı bölgelerde ortak nonlinear ilişkiler öğrenebilir.
2. `TWS_t` varsa absolute-target known uzmanı en güvenilir yoldur.
3. `TWS_t` maskeliyse climate/soil/space/time static feature'ları ve geçmiş climatology kısmi
   state proxy'si olabilir.
4. Testteki mask pattern validation tarihlerine ordinal olarak taşınabilir.
5. İki fold üzerindeki `0.65 baseline + 0.35 climatology-prior` kazancı geleceğe genellenir.
6. Latitude/longitude ve elapsed time, unseen future için stabil coğrafya/trend temsilidir.
7. Global RMSE'deki kazanç, region ve extreme-regime bias'ını kabul edilebilir düzeyde tutar.

### Leakage/risk register

| Risk | Durum | Açıklama |
|---|---|---|
| Test target kullanımı | Yok | Canonical kodda test target yok |
| Masked `TWS_t` kullanımı | Yok | Masked feature listesinden çıkarılır |
| Validation target ile prior | Kontrol edilmiş | Validation priors fold train'den |
| Location expanding history | Güvenli | `shift(1)` ve location-time order |
| `prior_global_month` training order | **Denetim gerekli** | Data location-major sıralı iken month-group expanding, training satırına göre gelecek tarihleri içerebilir |
| OOF blend weight selection | Orta risk | Yalnız iki fold; ağırlık hassasiyeti/private drift |
| Known/masked mask transfer | Orta risk | Ordinal pattern var, gerçek calendar gaps yok |
| Latitude/longitude interpolation | Orta risk | Aynı gridde güçlü; unseen-region robustness kanıtlamaz |
| Public LB seçimi | Canonicalde sınırlı | Sonraki notebook'larda local/public uyumsuzluğu gözlendi |
| Artifact identity | Çözülmüş | Notebook 17 SHA256 ile canonical dosyayı doğruladı |
""")

md("## 6. Challenge'dan türeyen, henüz yeterince test edilmemiş araştırma fikirleri")
code("""ideas=pd.DataFrame([
("Irregular-origin hindcast","Test dates span 39 months but only 18 timestamps","Recreate exact relative calendar gaps and mask schedule at multiple historical origins","Validation first"),
("Spatial block CV","Official text requires robustness across space","Hold out contiguous basins/large lat-lon blocks, not random rows","Validation first"),
("Strict time-sorted target encoding","Global-month prior order risk","Sort by time before cross-fitted global/regional encodings","Leakage audit"),
("Component-aware residual diagnostics","TWS includes unobserved groundwater/snow/surface stores","Use climate/latitude regimes to identify missing-component bias; do not invent unavailable states","Analysis"),
("Snow/cold-region proxy from allowed Copernicus data","Snow is a TWS component absent from current columns","Only data operationally available within one month; document latency","Conditional external data"),
("Runoff/ET/water-balance forcing","SPEI compresses P-PET and soil moisture is partial state","Test operational Copernicus P, ET, runoff with explicit publication latency","Conditional external data"),
("Basin/topography/geology static context","Groundwater and surface routing are spatially structured","Open static, reproducible basin/elevation/aquifer descriptors","Conditional external data"),
("Bias-aware reporting","Phase-two rubric explicitly asks bias/transparency","Report region, climate, mask, extreme and recent-period error with uncertainty","Required reporting"),
("Mask-aware calibration","Known and masked are different information regimes","Calibrate separately using cross-fitted OOF only","Low-cost experiment"),
("Efficiency frontier","Sustainability/practicality are judged","RMSE vs runtime/memory/model-size Pareto report","Required reporting"),
],columns=["Idea","Concrete challenge evidence","Leakage-safe test","Priority"])
display(ideas)
""")

md("""## 7. Araştırma kararı

Mevcut evidence, yeni ve daha karmaşık bir model ailesi eklemeyi desteklemiyor:

- Notebook 14 alternatif aileleri canonical LightGBM'i tek başına geçmedi.
- Notebook 12/14 local kazançları public leaderboard'a taşınmadı.
- Notebook 18'de fiziksel olarak anlamlı feature'ların çoğu negatif; üç pozitif feature'ın
  birlikte kullanımı canonical'dan kötü.
- Notebook 17 canonical artifact/pipeline kimliğinin model yeniliğinden önce çözülmesi gerektiğini
  gösterdi.

Dolayısıyla Notebook 11 **canonical baseline olarak korunmalıdır**. Bununla birlikte mevcut
validation “nihai olarak en uygun” ilan edilemez. Bir sonraki araştırma, model değiştirmeden:

1. exact test-gap/mask schedule ile çoklu historical-origin hindcast,
2. contiguous spatial block stress test,
3. `prior_global_month` dahil tüm target-derived feature'ların strict time-order/cross-fit auditi,
4. region × time × known/masked bias raporu

olmalıdır. Ancak bu testlerde istikrarlı kazanç gösteren bir hipotez yeni submission'a taşınmalıdır.

Bu notebook yeni submission üretmez ve canonical pipeline'ı değiştirmez; amacı challenge
description'ı uygulanabilir bir model-governance ve validation spesifikasyonuna dönüştürmektir.
""")

nb.cells=c
nb.metadata={"kernelspec":{"display_name":"GWS project environment","language":"python","name":"gws"},
             "language_info":{"name":"python","version":"3.13"}}
nbf.write(nb,OUT)
print(OUT,len(c))
