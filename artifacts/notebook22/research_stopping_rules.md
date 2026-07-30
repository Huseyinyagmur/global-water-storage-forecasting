# Research stopping rules

These are **risk-control policies, not universal scientific laws**.

- Change direction after **two consecutive public-worse submissions** from the same idea family.
- Require at least **0.001 spatial mean RMSE gain** and positive southern gain before considering a correction.
- Keep submission-level prediction **MAD ≤0.01** versus the current base; lower is preferred.
- Permit at most **0.0002 worst-fold** deterioration and **0.0005 strict/irregular** deterioration.
- Correct at most **25% of all rows** unless independent support evidence justifies wider coverage.
- Treat a public gain below **0.0005** as weak evidence requiring independent replication, not proof.
- Test at most **three predeclared variants** within one idea family before stopping or redesigning.
- Preserve the current best and move to the competition report when two high-priority families fail,
  when gains are below practical thresholds, or when remaining ideas require leakage risk/large drift.
