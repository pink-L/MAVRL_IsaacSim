# 自适应速度与目标速度奖励对照

Single training seed, one declared target/reward coefficient. Actual speeds may differ. No Pareto curve, equal-speed dominance, or cross-seed claim.

| 密度 | 基线到达率 | 目标速度对照到达率 | 基线碰撞率 | 对照碰撞率 | 基线平均速度 | 对照平均速度 |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 1.000 | 1.000 | 0.000 | 0.000 | 2.232 | 2.023 |
| medium | 0.980 | 0.980 | 0.020 | 0.020 | 2.151 | 2.010 |
| dense | 0.890 | 0.810 | 0.110 | 0.190 | 1.908 | 1.950 |

目标速度为2.0m/s，final速度项系数为-0.05。这些是本项目事先声明的参数，不冒称官方精确配置；目标速度奖励不保证每步恒速。
