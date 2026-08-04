# Notebook 35 final recommendation

1. Flattened lag pipeline leakage-free kuruldu mu? **Evet, pilot kapsamında.** Tüm laglar location içi `shift(k)` ile üretildi; `max(source_date) < prediction_date` doğrulandı. Target-derived feature yoktur.
2. En iyi window length hangisi? Pilot F3 ablation sonucu **L=6**; ana L12 karşılaştırması önceden sabitti.
3. Raw lag mı summary feature mı daha faydalı? L12 pilotunda **raw lag** (F1=0.640994, F2=0.647039).
4. Flattened LightGBM canonical modeli geçti mi? Aynı satırlardaki recorded canonical E0'a göre **True** (E0=0.772760, F1_L12=0.640994). Bu yalnız pilot kanıtıdır.
5. Masked RMSE ne kadar değişti? 0.808143 -> 0.659080; gain=+0.149063.
6. Robust validation kapıları geçildi mi? **Hayır.** Full strict/irregular/spatial/space-time validation çalıştırılmadı; tahmin 93.8 dakika ve 16.6 GB peak RAM, durum `not_run_performance_gate`.
7. GRU pilot kazancının ne kadarı yeniden üretildi? Gösterge oranı **172.8%**; farklı eğitim protokolleri nedeniyle doğrudan nedensel karşılaştırma değildir.
8. GRU sequence mimarisinin ayrı katkısı var mı? **Belirlenemedi.** Aynı full foldlarda kontrollü GRU/flattened karşılaştırması gerekir.
9. Blend işe yaradı mı? Pilot OOF'ta **True**; RMSE=0.640591, canonical weight=0.05. Tek fold olduğu için ağırlık stabilitesi kanıtlanmadı.
10. Submission üretildi mi? **Hayır.** Robust acceptance zorunludur.
11. Full GRU validation hâlâ öncelikli mi? **Evet, ancak önce seçilmiş flattened konfigürasyonun ayrı frozen full validation'ı tamamlanmalı.**
12. İnternet kullanmadan sonraki en güçlü deney nedir? **F1_L12 için tek, frozen strict/irregular/spatial/space-time validation koşusu; yeni feature veya parametre araması yok.**

Pilot promising=True; formal acceptance=False. Notebook 33 GRU değerleri yeniden hesaplanmadı ve `recorded result` olarak etiketlendi. Canonical/current-best submission dosyaları değiştirilmedi.
