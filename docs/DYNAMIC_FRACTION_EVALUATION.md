# 30% / 50% 原有圆柱动态比例实验

用户要求提高动态圆柱比例，分别用已有 baseline 和 current-only 权重直接评估。三档密度各 100 个任务，每模型每比例 300 个任务，总计 1200 次正式导航。沿用原任务 seed=200、16 个环境、10 Hz 控制、原运动学和奖励，不启动训练。

每张地图将 `floor(N×比例+0.5)` 个原圆柱设为动态。保留所有圆柱的半径、高度、总数、初始位置及任务起终点。运动仍为 1 m/s、4 m 线段往返，从轨迹中点出发，按 episode 仿真时间推进。每张地图先确定完整 50% 集合，30% 使用同一集合及轨迹的前缀；两模型使用完全相同的地图和运动。

上一轮包围圆不相交的保守筛选无法容纳部分密集地图的高比例。本轮新增 `mavrl_isaac/fraction_tracks.py`，以完整周期的相对运动距离检查代替包围圆筛选，保持圆柱实体不相交，表面间隔至少 1 mm。每条轨迹也避开所有其他圆柱的初始位置，因此未被选中的圆柱保持静止时仍然相容。边界及起终点净空继续保留。初始可行性不等于动态全过程的可行性保证。

原 3 个圆柱配置及结果保留。它与本轮的轨迹筛选规则不同，不能把其性能差异全部解释为比例效应。30% 与 50% 之间则共用严格嵌套的运动。

配置：`configs/isaac_capture2_dynamic_30pct.json`、`configs/isaac_capture2_dynamic_50pct.json`。两份配置引用同一个不可变的 `artifacts/dynamic_fraction_20261106/master_tracks.json` 及 SHA256；运行时检查轨迹清单与配置、原始地图、任务的一致性。

证据目录：`artifacts/dynamic_fraction_20261106/`。`protocol.json` 固定评估设置；`geometry_audit.json` 记录全部 300 张地图及实际动态数量；`cpu_tests.log`、`smoke_replay_checks.json` 记录实现与重复性检查；各 `pct*_full/` 保存原始任务、轨迹、配置、源码快照和权重摘要；`status.json` 是当前队列状态。完成后的报告为 `RESULTS.md` 和 `comparison.json`。

1200 次正式评估已全部完成，到达率如下：

| 动态比例 | 模型 | 稀疏 | 中等 | 密集 |
|---|---|---:|---:|---:|
| 30% | baseline | 83% | 62% | 34% |
| 30% | current-only | 88% | 69% | 27% |
| 50% | baseline | 76% | 47% | 22% |
| 50% | current-only | 76% | 46% | 14% |

所有失败均为碰撞。baseline 在本轮密集场景分别高 7、8 个百分点；其他条件没有一致优势，两组实际速度也不同。结论限定于当前单训练种子和这一运动设置。35 项 CPU 测试、高比例小规模重复评估、完整场景配对及比例间轨迹嵌套检查通过；原静态和上一轮交付文件保持完整。完整指标和配对计数见[实验报告](../artifacts/dynamic_fraction_20261106/RESULTS.md)。

独立复评时使用新的输出目录：

```bash
scripts/python.sh scripts/run.py evaluate \
  --config configs/isaac_capture2_dynamic_30pct.json \
  --policy artifacts/capture2_baseline_seed0/final/latest.pt \
  --out artifacts/dynamic_fraction_repeat_new/pct30_baseline \
  --episodes 100 --num-envs 16 --seed 200
```

50% 改用对应配置；current-only 改用 `artifacts/capture2_current_seed0/final/latest.pt`、独立输出目录，并加 `--current-only`。当前轨迹清单覆盖本轮固定的 300 个任务，其他任务应另行准备轨迹清单，不能静默回退到较少移动圆柱。
