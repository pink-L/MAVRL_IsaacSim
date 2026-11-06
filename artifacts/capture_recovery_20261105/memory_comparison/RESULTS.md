# Paired reconstruction-objective ablation

One shared-data/shared-VAE training seed with two reconstruction objectives and equally budgeted PPO retraining. Current-only still uses an LSTM. Descriptive paired results; no multi-seed or statistical-significance claim.

| Density | Past + current arrival | Current-only arrival | Past/current-only wins | Current-only wins |
|---|---:|---:|---:|---:|
| sparse | 1.000 | 1.000 | 0 | 0 |
| medium | 0.980 | 0.990 | 1 | 2 |
| dense | 0.890 | 0.890 | 6 | 6 |
