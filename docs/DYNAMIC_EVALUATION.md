# 原有圆柱运动时的 MAVRL 直接迁移评估

本轮按用户要求，让部分原有圆柱移动，并分别评估已交付的 baseline 和 current-only。不重训策略、表征或奖励，不加入安全过滤器。current-only 仍保留 LSTM 历史状态。

配置为 `configs/isaac_capture2_dynamic.json`。每张原始 Poisson 地图选取 3 个圆柱，以 1.0 m/s 在长度 4 m 的直线段上往返。圆柱从线段中点出发，方向由地图种子和独立的动态种子派生；原半径、高度、总数、任务起终点和初始位置保留。三档密度沿用静态交付的 100 个任务，评估 seed=200，16 个并行环境，两组使用相同地图、移动圆柱索引和运动轨迹。运动由每个 episode 的仿真时间决定，与墙钟耗时和策略动作无关。

仅从具有移动空间的原有圆柱中选择：完整轨迹必须位于地图内、避开其他原有圆柱和起终点净空；已选轨迹的包围圆互不相交。无法放置时明确报错，不删掉任务或重新生成地图。该选择偏向有移动空间的障碍物。初始地图保留原连通性检查；没有证明整个动态过程都存在可行通路。

`CylinderScene.cylinders` 保存原始地图，`at_time(t)` 返回当前几何，`static_cylinders` 排除已选移动圆柱。渲染直接移动原 USD 圆柱，没有额外创建同位置障碍物。碰撞检测保留原无人机步内线段近似，并对无人机与圆柱的相对运动进行连续检测；在圆柱折返时刻分段，避免只检测最终位置导致穿透。原 TRAV 保留用于奖励日志；本次主要指标是到达率、终止原因、速度和配对任务结果。

实现检查包括 CPU 几何测试、全部 300 张地图的数量/半径/初始位置核对、真实 RTX 移动深度检查，以及每组 9 个任务的两次独立小规模评估。重复检查不增加独立样本数。三档各 100 个任务的正式评估另行保存。

本轮证据目录为 `artifacts/dynamic_existing_20261106/`：

- `protocol.json`：运行前固定的设置和结论边界。
- `geometry_audit.json`、`cpu_tests.log`、`render_probe/`：实现检查。
- `baseline_full/`、`current_only_full/`：原始评估、配置、源码快照、权重摘要及逐秒轨迹。
- `scenes.json`（各评估目录中）：原始圆柱、静止圆柱、移动索引及轨迹参数。
- `status.json`、`smoke_replay_checks.json`：队列及小规模重复检查。
- `RESULTS.md`、`comparison.json`、`comparison.png`：最终结果、配对统计及对比图。

正式评估已完成。稀疏/中等/密集到达率为 baseline **94%/89%/81%**，current-only **98%/89%/81%**；所有失败均为碰撞。两组在中等场景各有 7 个任务独自成功，在密集场景各有 10 个任务独自成功。本轮没有观察到过去帧重建提高到达率；baseline 平均速度更高，不能将两者视为等速度对照。完整指标与原静态任务的配对变化见[动态评估报告](../artifacts/dynamic_existing_20261106/RESULTS.md)。

独立复评示例，输出必须改为新的目录：

```bash
scripts/python.sh scripts/run.py evaluate \
  --config configs/isaac_capture2_dynamic.json \
  --policy artifacts/capture2_baseline_seed0/final/latest.pt \
  --out artifacts/dynamic_repeat_new/baseline \
  --episodes 100 --num-envs 16 --seed 200
```

current-only 改用 `artifacts/capture2_current_seed0/final/latest.pt`、独立输出目录，并加 `--current-only`。配置不含 `dynamic_obstacles.enabled=true` 时仍运行原静态环境。

结果只支持这个移动数量、速度、轨迹长度和单训练种子下的直接迁移表现。不得将两组差异解释为普遍的记忆收益或动态障碍泛化保证。用户选择之前启动的“额外加入 3 个圆柱”探针保留在 `artifacts/dynamic_zero_shot_20261106_100717/`，已停止，排除在本轮正式结论之外。
