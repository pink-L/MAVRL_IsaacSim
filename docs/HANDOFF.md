# MAVRL / Isaac Sim 交付与复跑指南

工作目录：`/home/lz/lzspace/MAVRL_CBF`。本项目为静态圆柱环境、简化运动学、单训练种子的算法复现。结论及证据边界见[复现报告](REPRODUCTION_REPORT.md)，完整运行环境见[ENVIRONMENT.md](ENVIRONMENT.md)。不含C3BF、动态障碍或实飞。

## 入口和主要文件

| 用途 | 文件/目录 |
|---|---|
| 主模型：过去＋当前重建、自适应速度 | `artifacts/capture2_baseline_seed0/final/latest.pt` |
| current-only：仅当前重建、自适应速度 | `artifacts/capture2_current_seed0/final/latest.pt` |
| fixed：过去＋当前重建、目标速度奖励 | `artifacts/capture2_fixed_speed_seed0/final/latest.pt` |
| 三组指标、权重SHA256、汇总图和训练曲线 | `artifacts/delivery_20261106/` |
| 每组原始训练记录、任务结果、轨迹、配置 | 各组下的`final/`及`evaluation_full/` |
| 记忆清零/独立重复评估 | 各组下的`evaluation_full_reset/`、`evaluation_full_replay/` |
| 两组训练消融的配对汇总 | `artifacts/capture_recovery_20261105/memory_comparison/`、`speed_comparison/` |
| 过去＋当前LSTM和共享VAE/数据 | `artifacts/formal_seed0/lstm/`、`vae/`、`dataset/` |
| current-only LSTM | `artifacts/ablation_current_seed0/lstm/` |
| 未修改的官方源码 | `upstream/mavrl/`、`upstream/AvoidBench/` |
| 实际执行命令和源码快照 | 每个训练/评估目录的`command.json`、`config.json`、`source/` |

`latest.pt`含完整表征、actor/critic、训练配置、优化器和随机数状态；评估时不需要重新训练VAE/LSTM。运行仍需要本项目代码、固定的上游VAE代码及兼容Isaac环境。部分共享产物目录为符号链接，跨机器迁移时不能只复制链接本身；应保留所指向的项目内原始目录。权重绝对解析路径和摘要见`WEIGHTS.json`。

## 环境与检查

```bash
cd /home/lz/lzspace/MAVRL_CBF
scripts/python.sh -m unittest discover -s tests -v
nvidia-smi
```

`scripts/python.sh`使用项目内`.venv`只读复用现有Isaac Sim 5.1 / PyTorch 2.7+cu128；缓存均写入本项目。不要在LargeMap或共享环境安装/升级依赖。需要从头配置另一台机器时，先按`ENVIRONMENT.md`准备兼容依赖，再创建使用这些依赖的本地虚拟环境。

## 加载已训练模型并复评

选择新的输出目录，避免覆盖已有评估。以下主模型命令会实际启动Isaac RTX，完成三个密度各100个固定任务：

```bash
scripts/python.sh scripts/run.py evaluate \
  --policy artifacts/capture2_baseline_seed0/final/latest.pt \
  --config configs/isaac_capture2.json \
  --out artifacts/reproduction_new/baseline_eval \
  --episodes 100 --num-envs 16 --seed 200
```

current-only：将policy路径改为`artifacts/capture2_current_seed0/final/latest.pt`，另选输出目录，并加入`--current-only`。fixed：将policy路径改为`artifacts/capture2_fixed_speed_seed0/final/latest.pt`，config改为`configs/isaac_capture2_fixed_speed.json`，另选输出目录。

`--reset-memory`表示在推理时每帧清空隐状态，只用于诊断，不能替代current-only训练消融。正常重放应使用同样的policy、config、seed与环境数，再输出到另一个新目录，比较`episodes.json`和`trajectories.json`。10Hz是仿真策略频率，每步推进0.1秒，不能与墙钟吞吐混淆。

## 使用已保存表征，复跑800万步最终PPO

```bash
scripts/python.sh scripts/run.py train --phase final \
  --representation artifacts/capture2_baseline_seed0/lstm/best.pt \
  --config configs/isaac_capture2.json \
  --out artifacts/reproduction_new/baseline_final \
  --steps 8000000 --num-envs 64 --seed 0
```

current-only使用`artifacts/capture2_current_seed0/lstm/best.pt`，增加`--current-only`；fixed共享baseline的LSTM，使用`configs/isaac_capture2_fixed_speed.json`。各自另选输出目录。三组均为64×1000样本/rollout、minibatch4000、10 epochs、lr1e-4按800万步线性衰减，前1.216m处切换奖励/场景课程。参数依据及64至512并行实测见[PARALLELISM.md](PARALLELISM.md)。

上述复跑保留已有表征和数据来源，与已交付模型的最终PPO阶段最接近；不能将已有数据标为新生成的数据。

## 从初始PPO重新运行整个流程

```bash
# 第一步：冻结随机表征的基础导航策略
scripts/python.sh scripts/run.py train --phase initial \
  --config configs/isaac_capture2.json \
  --out artifacts/reproduction_full/initial \
  --steps 512000 --num-envs 16 --seed 0

# 后续：采集65536帧、VAE、LSTM、最终PPO、评估和图表
scripts/python.sh scripts/run_long_pipeline.py --formal \
  --initial-policy artifacts/reproduction_full/initial/latest.pt \
  --root artifacts/reproduction_full/baseline \
  --config configs/isaac_capture2.json \
  --frames 65536 --steps 8000000 --episodes 100 --num-envs 64
```

初始PPO的512k是已验证可导航的早停预算，学习率仍按8m总计划计算。VAE最多1000轮、按原early stopping结束；LSTM2000轮；最终PPO冻结表征、重新初始化策略头。

这套从头命令使用双子帧配置重新采集数据；已交付三组实验复用了历史单子帧数据/VAE/LSTM，随后在双子帧设置下训练最终PPO。两者数据来源不同，不能无条件承诺训练结果逐位一致。需要严格核对已交付实验时，以其保存的命令、配置和源码快照为准。

## 中断恢复与日志

```bash
scripts/python.sh scripts/run.py train \
  --resume artifacts/reproduction_new/baseline_final/latest.pt \
  --out artifacts/reproduction_new/baseline_final --steps 8000000
```

恢复继承checkpoint阶段、配置、优化器与随机数状态，但会开始新的仿真episode，不保证从中断帧逐位续接场景。保存间隔为10个rollout，完成时也保存；中断可能损失最近一次保存之后的训练。

`final/train.jsonl`为原始rollout指标，`final/tensorboard/`用于曲线查看，`final/episodes.jsonl`保留训练任务结果。独立评估由`evaluation_full/summary.json`汇总；训练episode到达率不能替代独立评估。各组`validation.json`和`memory_figures_full/provenance.json`记录冻结、加载及重放检查。

已完成队列不要重复启动。旧失败队列和诊断证据保留，不将历史失败状态改成完成。新项目交付摘要的可重建入口为：

```bash
scripts/python.sh scripts/summarize_delivery.py --out artifacts/delivery_new_export
```

该脚本要求三组实验和配对检查已经完成，只读取已有结果，在CPU上生成表格、SHA256清单和图像，不重新训练或评估。
