import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "21_spatial_residual_correction.ipynb"
RUNTIME = ROOT / "scripts" / "notebook21_runtime.py"

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("""# 21 — Spatial Residual Correction

Amaç yeni bir ana model veya feature seti geliştirmek değildir. Notebook 11'in immutable canonical
selected-blend tahminleri korunur; Notebook 20'nin strict temporal, irregular hindcast, contiguous
spatial-block ve space-time tasarımları altında yalnızca küçük ve kanıta dayalı residual intercept
correction'ları sınanır.

`residual = target - prediction`; pozitif residual underprediction demektir.

## Önceden sabit araştırma protokolü

- Correction istatistikleri yalnız üç **daha eski**, date-grouped temporal OOF kalibrasyon fold'undan öğrenilir.
- Değerlendirilen validation target'ları correction hesabına girmez.
- Yalnız hemisphere, 10° latitude band, 20° coarse cell, hierarchical fallback ve prediction-time
  mevcut SPEI tabanlı climate regime denenir.
- Shrinkage grid'i `alpha={5000,1000,200}`; cap grid'i `±{0.01,0.02,0.03,0.05}` olarak sonuçlardan
  önce sabittir.
- Stable bias: 3 fold, en az %75 aynı yön, her fold'da en az 100 satır, mutlak bias en az 0.01 ve
  pooled bootstrap %95 CI'ın sıfırı dışlaması.
- Kabul toleransları sonuçlardan önce sabittir: strict/irregular en fazla 0.0005, space-time ve
  worst-fold en fazla 0.0002 bozulabilir. Spatial mean ve southern block mutlaka iyileşmelidir.
- Known test satırları değiştirilemez; yalnız bütün gate'ler ve test-distribution safety geçerse tek
  candidate submission yazılır. Canonical dosyanın üzerine asla yazılmaz.

Notebook 20'de spatial proxy'nin yalnız dört public adayla Spearman 0.40 ve pairwise %66.7 vermesi
zayıf kanıttır. Bu nedenle seçim spatial skora tek başına göre yapılmaz."""),
    nbf.v4.new_markdown_cell("""## Canonical reproduction, residual map, stability, correction ablation and submission gate

Aşağıdaki hücre bütün hesapları sırasıyla yapar, her ana aşamada elapsed time yazdırır ve zorunlu
artefaktları `artifacts/notebook21/` altına kaydeder. Kaynak kod builder tarafından notebook içine
gömülür; notebook dış runtime state'e bağımlı değildir."""),
    nbf.v4.new_code_cell(RUNTIME.read_text(encoding="utf-8")),
]
nb.metadata = {
    "kernelspec": {"display_name": "GWS project environment", "language": "python", "name": "gws"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, OUT)
print(OUT, len(nb.cells))
