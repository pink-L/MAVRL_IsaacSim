# 三组完成实验的结果汇总

One training seed, 100 paired scenes per density, three completed 8m PPO runs. Current-only retains LSTM; reset-memory tests are inference diagnostics. Fixed uses one project-selected 2 m/s target. No multi-seed, equal-speed Pareto, original AvoidBench numerical reproduction, or real-flight claim.

| 配置 | 密度 | 到达/100 | 碰撞/100 | 平均速度m/s | 清空记忆到达率 |
|---|---|---:|---:|---:|---:|
| baseline | sparse | 100 | 0 | 2.232 | 95% |
| baseline | medium | 98 | 2 | 2.151 | 73% |
| baseline | dense | 89 | 11 | 1.908 | 32% |
| current_only | sparse | 100 | 0 | 2.098 | 100% |
| current_only | medium | 99 | 1 | 2.015 | 98% |
| current_only | dense | 89 | 11 | 1.779 | 82% |
| fixed | sparse | 100 | 0 | 2.023 | 90% |
| fixed | medium | 98 | 2 | 2.010 | 73% |
| fixed | dense | 81 | 19 | 1.950 | 20% |

权重及SHA256：`WEIGHTS.json`。原始统计、配对成败计数、源文件摘要：`summary.json`。

每组速度先按episode求均值，再对100个任务等权平均。到达率区间为场景任务上的95% Wilson区间，不能表示跨训练种子的误差。

结论须区分历史状态使用、过去帧重建监督和目标速度奖励三个问题；不以训练完成代替消融收益成立。
