# Notebook 33 final recommendation

1. Panel sequence structure correct: True; chronological location panels and training-only scaling verified.
2. Best sequence length: 12.
3. Selected family: GRU; structure: dual.
4. Sequence better than canonical on pilot: True (E0 0.772760, S1 0.696528).
5. Masked RMSE improved: True (E0 0.808143, S1 0.727309).
6. Cold start fraction: 0.013%; left padding was used.
7. Residual diversity: Pearson 0.8577, Spearman 0.8229.
8. Blend useful: True; canonical weight 0.15, RMSE 0.694025.
9. Acceptance gate passed: False. Pilot thresholds A/B/C=True/True/True, but formal acceptance requires full strict/irregular/space-time validation. Full training was skipped because CUDA-capable PyTorch is unavailable, not because the pilot lacked signal.
10. Submission produced: No.
11. Expand sequence approach: Yes—first install a CUDA-capable PyTorch build, then run the selected GRU-L12 dual expert on the frozen full validation systems only.
12. Highest-potential next experiment without global ERA5: full validation of this single selected configuration after the CUDA environment is repaired; do not reopen architecture search.

CUDA could not be used because installed PyTorch is CPU-only despite the RTX 4050 being present. Mixed precision and GPU-memory metrics are therefore unavailable. This is a bounded pilot, not full validation or leaderboard evidence.
