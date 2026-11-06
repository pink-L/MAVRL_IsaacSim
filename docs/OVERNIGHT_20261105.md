# 2026-11-05夜间至11月6日上午安排

**实际执行更新（2026-11-06）**：current-only于02:04完成，fixed随即接续；fixed、正常/清零/重复评估、产物验证及两组配对检查于08:29全部结束。GPU已空闲，09:00本地报告仍按计划生成。以下是保留的训练前安排。

用户已明确今晚及明天上午GPU可用，并要求安排fixed。复用正在运行的训练监督进程PID917386，不中断current-only，也不重复启动训练队列。实际状态为`artifacts/capture_recovery_20261105/status.json`。

| 北京时间预计窗口 | 工作 | 完成产物 |
|---|---|---|
| 现在至11月6日02:00–03:00 | current-only余下PPO，随后三密度正常/记忆清零/重复评估、产物验证和图表 | `artifacts/capture2_current_seed0` |
| 约02:00–03:00开始，08:00–10:00结束 | fixed目标速度PPO全预算8m，随后同样的评估和验证 | `artifacts/capture2_fixed_speed_seed0` |
| 两组流水线完成后 | 自动比较baseline/current-only及baseline/fixed的配对结果 | `artifacts/capture_recovery_20261105/memory_comparison`、`speed_comparison` |
| 09:00 | 独立CPU进程生成本地阶段报告；GPU训练若未结束则继续 | `artifacts/overnight_20261105/MORNING_REPORT.md`及JSON |

截至20:38，current-only记录1472000/8000000步，混合密度阶段约352步/秒；上述时间依据当前速度与已完成主基线约6.1小时训练耗时估算。不同策略的episode长度、场景重置数量和后续检查会改变耗时，不保证某时刻已经完成。两组结果齐备后再完成最终交付文档的结果解释与核对。

正式配置保持64环境×1000 steps、minibatch4000、10 epochs，每组PPO800万步、seed0。fixed共享baseline的past+current表征，final阶段速度项为`-0.05*abs(v_horizontal-2.0)`，其他训练设置保持配对。今晚不再增加并行规模试验或新的训练种子。

训练队列在独立阶段失败时保留失败状态与日志；不会将未通过检查的结果标记为完成。09:00报告进程只读训练产物，不启动、停止或重启GPU进程，也不向外部发送消息。

```bash
cat artifacts/capture_recovery_20261105/status.json
cat artifacts/overnight_20261105/progress.md
cat artifacts/overnight_20261105/report_status.json
```

报告计划与启动记录保存在`artifacts/overnight_20261105`。报告脚本的4项调度/配对/只读报告测试通过，包括未开始的fixed不能生成虚构评估结果。
