# Notebook 20 final recommendation

1. **prior_global_month leakage:** Evet, canonical *training feature construction* içinde zaman sırası riski doğrulandı.
2. **Mekanizma:** Data `lat, lon, time` sıralı iken month-group `shift(1).expanding()` uygulanıyor; önceki satır, önceki zaman demek değil. Future timestamp target'ları daha eski satırların feature'ına girebiliyor. Validation target'ları aggregate'a girmiyor.
3. **Canonical local skor yapay mı?** Risk vardır fakat skorun ne kadarının bundan geldiği strict ablation tablosuyla ölçülür; validation target doğrudan sızmadığı için tüm kazanımı leakage'e atfetmek doğru değildir.
4. **En güvenli prior:** strict date-grouped expanding/OOF ve hierarchical yöntemler causality bakımından güvenlidir. Performans seçimi `prior_ablation.csv` içindedir.
5. **Public sıralamayla uyum:** Yalnız dört aday ve farklı kayıtlı OOF satırları nedeniyle sonuç zayıf kanıttır; `model_ranking_consistency.csv` belirsizliği açıkça taşır.
6. **Irregular hindcast:** Tarih-gap benzerliği canonical ordinal foldlardan daha yüksektir; exact test offsets raporlandı.
7. **Spatial robustness:** Contiguous future-region holdout skorları `spatial_block_folds.csv` içindedir; same-grid temporal skorla eşdeğer değildir.
8. **Canonical korunmalı mı?** Evet. Public-best artifact immutable kaldı ve strict alternatif bütün kapıları geçmeden değiştirilmedi.
9. **Yeni submission:** Hayır. Varsayılan politika uygulandı.
10. **Tek sonraki öncelik:** strict time-ordered prior pipeline'ını bağımsız ikinci run ile yeniden üretip aynı irregular + space-time foldlarda seed-stability doğrulamak.

Canonical SHA256: `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`.
