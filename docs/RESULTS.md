# 实际结果与验收状态

**2026-11-06交付更新：三组800万步PPO及全部评估、重复性和配对检查已于08:29完成。**完整算法对照、训练预算、数据来源和结论见[REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)，运行命令见[HANDOFF.md](HANDOFF.md)。

| 配置 | 稀疏到达率 | 中等到达率 | 密集到达率 | 平均速度（稀疏/中等/密集，m/s） |
|---|---:|---:|---:|---|
| baseline | 100% | 98% | 89% | 2.232 / 2.151 / 1.908 |
| current-only | 100% | 99% | 89% | 2.098 / 2.015 / 1.779 |
| fixed | 100% | 98% | 81% | 2.023 / 2.010 / 1.950 |

每密度100个相同独立任务，单训练种子。当前结果没有显示过去帧重建相对current-only提高到达率；清空隐状态的诊断支持历史状态依赖；变速与fixed在密集场景的差异为本组参数下的观察，不能宣称等速度或跨种子优势。三组正常与重复评估的episode/trajectory文件SHA均一致。23项CPU测试和各组权重加载/冻结检查通过。

正式权重为`artifacts/capture2_baseline_seed0/final/latest.pt`、`artifacts/capture2_current_seed0/final/latest.pt`和`artifacts/capture2_fixed_speed_seed0/final/latest.pt`。完整路径与SHA256见[权重清单](../artifacts/delivery_20261106/WEIGHTS.json)。三组最终PPO使用双子帧设置，共享数据/VAE/LSTM保留原单子帧来源；旧异常与历史权重没有覆盖。模型评估属于Isaac静态圆柱仿真，不等同于AvoidBench数值复现，不包含实飞。

## 历史短程结果和启动记录

以下保留11月4日阶段记录。512k最终阶段三密度到达率83.3% / 53.3% / 20.0%（各30任务），清空记忆后63.3% / 36.7% / 10.0%；这两次短程权重评估的90条episode及轨迹当时全部一致。它们不替代上面的正式训练结果。

- 当前可加载导航权重：`artifacts/corrected/final/latest.pt`。
- 当前评估：`artifacts/corrected/evaluation_sync/`；记忆诊断：`evaluation_sync_reset/`；重放：`evaluation_sync_replay/`。
- 曲线：`artifacts/corrected/figures_sync/`；记忆对照：`memory_figures_sync/`。
- [实际轨迹图](../artifacts/corrected/figures_sync/trajectories.png)：每档固定取最小评估seed，保留成功、碰撞、越界；仅绘制真实1Hz位置采样点，不虚构终止位置。PDF和数据摘要同目录。
- 长预算流程已于2026-11-04 18:35:57启动，目录`artifacts/formal_seed0/`；[实时状态](../artifacts/formal_seed0/long_run/status.json)。计划重新采集65536帧、VAE最多1000 epochs（early stopping）、LSTM2000 epochs、final PPO8e6步；复用已验证的基础策略，不是额外独立初始训练seed。完成前不能把运行中的作业视为新的结果。
- 正式数据采集于18:40完成：实际保存65536帧、378个连续序列，训练/验证场景无交集；已进入VAE训练。检查见`artifacts/formal_seed0/dataset/collection_check.json`。这一阶段完成不代表后续LSTM/PPO已完成。
- VAE已在第242轮按原early stopping结束，耗时约24.4分钟；LSTM已完成2000轮及训练后记忆诊断，正式64环境PPO于22:11启动，目标8m步，rollout/minibatch/epochs保持1000/4000/10；见[并行实测报告](PARALLELISM.md)。这不是新的导航效果结果。
- 夜间队列已安排主基线完成后自动运行current-only重训消融，并于2026-11-05北京时间09:00生成阶段报告；用户明确选择09:00后继续训练。详情见`docs/OVERNIGHT_PLAN.md`，实时阶段见`artifacts/overnight_20261104/status.json`。

速度口径：使用源码存储的FLU速度分量`norm(v_xy)`，先逐步求episode均值，再对episode等权平均；这与奖励中的水平速度量一致。不是总路径长度除总时间，也不严格等于有pitch/roll时的世界系水平速度。每条episode另记录三维路径长度和持续时间，便于采用其他速度口径重算。到达/碰撞/越界/超时由仿真几何状态给出，未用模型预测作为真值。

## 已完成工程工作

- 固定两份未经修改的上游源码；审计操作系统、驱动、PyTorch和现有Isaac依赖，未安装大型依赖、未修改其他项目。
- 实现10 Hz简化加速度运动学、四维动作、深度相机、静态圆柱生成、可行性检查、原始奖励与终止条件。
- 实现六阶段训练、checkpoint保存/加载、独立种子评估、原始曲线、资源基准。
- 对uniform/poisson/clustered三种分布、三档密度各抽检30种子，共270任务，均通过端点≥1.5m净空与保守网格连通性检查；见`artifacts/scene_audit/report.json`。这不代表策略均能完成。
- 17项CPU测试通过，包含直接执行上游encoder/decoder、recurrent reset和actor/critic前向函数的比较；见`logs/tests_critic.log`。测试不是训练成功率证据。
- 16路tiled相机环境步进基准559.8 transitions/s，显存3766MiB；基础PPO实测约548 transitions/s、完整进程显存约4GB。拥挤地图频繁重置会降低吞吐。指标不是每个环境的控制频率，后者固定10 Hz。
- 显式深度同步修复后，16路相机重新测得459.6 transitions/s、3764MiB；这是当前适配层的资源基准。旧吞吐保留作为历史记录，不将更快的未同步路径作为正式训练配置。

## 旧版端到端冒烟：功能通过，导航未收敛

目录：`artifacts/smoke`，原始日志`logs/smoke_*.log`及各子目录。初始PPO8192步，采集4029帧，VAE/LSTM各10 epochs，final PPO16384步，三个密度各12任务。

| 指标 | 实际结果 | 可支持的结论 |
|---|---:|---|
| VAE验证MAE，0–255灰度 | 未训练79.93 → 最佳15.60 | 短程训练学到了所采集深度图的重建 |
| LSTM过去帧验证MSE，0–1深度 | 正常记忆0.04442；逐帧重置0.12040 | 自监督历史重建依赖记忆 |
| LSTM当前帧验证MSE | 正常0.02371；重置0.02531 | 不能单凭该差异推断导航收益 |
| 导航到达 | 稀疏0/12，中等0/12，密集0/12 | **未获得导航收敛证据** |
| 碰撞 | 2/12，2/12，3/12 | 其余为越界，不是成功或超时 |
| 全任务平均水平速度m/s | 0.580，0.501，0.491 | 失败轨迹的速度不能证明自适应导航 |

数据拆分含36个训练场景、11个验证场景，scene seed无重叠；权重保存/加载动作一致，loss有限，PPO期间表征不变。见`artifacts/smoke/validation.json`、`figures/`。旧代码使用随机采集、LSTM lr=3e-4、共享critic、64帧序列，且短预算内学习率降为零；修正后实验不覆盖这些历史记录。旧checkpoint没有运行时源码快照，事后Git提交不能补证历史精确版本。

## 旧版基础策略短程训练

`artifacts/short/initial`从随机权重训练256000 transitions，16环境，原rollout1000/batch4000/10 epochs。独立30个无障碍任务到达7/30=23.3%，Wilson95%区间11.8%–40.9%，其余23次越界。全部任务平均速度1.172m/s，成功任务1.087m/s。该基础策略尚不可靠；这不是有障碍最终策略。旧学习率在256k预算末衰减为零。

## 官方预训练权重零样本迁移

来源为固定官方仓库中的`ckpts/available_policy/iter_00540.pth`。使用官方actor，不是在Isaac训练的新模型。原转换文件`artifacts/official/iter_00540.pt`保留用于复查；补齐critic后的转换为`iter_00540_full.pt`。旧转换未保留独立critic，**actor导航动作不受影响**。

固定同一套场景种子，每密度20任务：

| 密度 | 到达 | 到达率95% Wilson区间 | 碰撞 | 越界 | 平均速度，全部/成功任务(m/s) |
|---|---:|---:|---:|---:|---:|
| 稀疏 | 18/20，90% | 69.9%–97.2% | 0/20 | 2/20 | 4.073 / 4.010 |
| 中等 | 12/20，60% | 38.7%–78.1% | 6/20 | 2/20 | 3.295 / 3.253 |
| 密集 | 1/20，5% | 0.9%–23.6% | 16/20 | 3/20 | 1.995 / 2.642 |

无超时。原始记录在`artifacts/official/evaluation/`。`evaluation_replay/`独立重启后，summary.results、全部episode记录和每秒轨迹记录与第一次运行完全相同，检查程序为`scripts/summarize_transfer.py`。这些旧地图使用第三方向y∈[0,40]；源码修正为[0,34]后不能把新地图视为同一任务。旧结果没有完整源码快照，重放一致性只证明这次实际重复运行，不是跨驱动或跨硬件确定性保证。

### 记忆诊断

同一官方权重、同一地图，每一步将LSTM隐状态清零：

| 密度 | 正常记忆到达率 | 重置记忆到达率 | 仅正常记忆成功 / 仅重置成功 |
|---|---:|---:|---:|
| 稀疏 | 90% | 80% | 3 / 1 |
| 中等 | 60% | 30% | 7 / 1 |
| 密集 | 5% | 0% | 1 / 0 |

这支持“该官方策略在这些迁移场景中利用了历史状态”的有限结论。推理时破坏记忆会改变输入分布，**不能替代等预算重新训练的current-only表征消融**；只有一个权重、每档20个任务，不宣称普遍统计收益。

### 速度诊断

真实飞行评估中密度越高、全任务平均速度越低，但碰撞/越界会改变任务持续时间，存在混杂。另用`scripts/probe_speed.py`固定其余状态、重复100帧均匀深度，扫描前向速度并计算actor平均前向加速度的正到负零点：输入深度约2.82m、5.65m、10.35m，对应零点约0.26、3.71、6.14m/s。这是**受控网络输入诊断，不是飞行速度或安全性能实验**。它表明actor会随视觉输入调整加减速指令；未完成固定速度重训对照，不能宣称已复现论文速度消融。

图及数据摘要：`artifacts/official/figures/transfer.png`、`.pdf`、`provenance.json`。不平滑或重新缩放成功率。

## 修正后实验

`artifacts/corrected/initial`已从随机权重训练512000 transitions，保留8e6步学习率计划；耗时约937秒，吞吐约546 transitions/s。独立50个无障碍任务 **50/50到达**，Wilson95%区间92.9%–100%，平均水平速度2.633m/s，碰撞/越界/超时均0。见`artifacts/corrected/initial_evaluation/summary.json`。这支持基础导航已学会，不能外推到有障碍环境。

修复包括：确定性采集、LSTM lr=1e-4、源码重载路径的独立critic、第三方向y区间、续训继承phase/config。独立修复核验通过，原权重53个张量完整映射、多步actor/value/隐状态误差为0，见`audit/FIX_VERIFICATION.md`。新运行保存`source/<timestamp>/`、`run_<timestamp>.json`及checkpoint源码SHA。

确定性采集已保存16,376帧、104个连续序列，87个训练scene seed与17个验证scene seed无重叠。VAE训练20 epochs，按验证总loss选择的第20轮权重MAE=6.6364/255、MSE=0.003371。LSTM训练30 epochs，最佳第30轮的平均逐序列past MSE=0.0134366、current MSE=0.0080562；逐帧重置记忆后分别为0.1347344、0.0142052。它们是独立场景上的自监督深度重建指标，不直接证明最终导航收益。可查看`artifacts/corrected/{vae,lstm}/reconstruction_*.png`。两阶段仍属短程表征预算，未执行官方1000/2000 epochs。

final PPO已训练512000 transitions，耗时1192秒、约429 transitions/s。PPO头从随机权重重新训练，encoder及两套LSTM保持冻结；其停止点仍处于官方1.2m步稀疏预热内，**尚未使用后续混合密度和final奖励阶段**，不能冒称完整最终策略。

正常记忆的独立评估每档30个任务：

| 密度 | 到达 | 到达率95% Wilson区间 | 碰撞 | 越界 | 全部/成功任务平均速度(m/s) |
|---|---:|---:|---:|---:|---:|
| 稀疏 | 25/30，83.3% | 66.4%–92.7% | 2/30 | 3/30 | 1.364 / 1.375 |
| 中等 | 16/30，53.3% | 36.1%–69.8% | 8/30 | 6/30 | 1.281 / 1.308 |
| 密集 | 6/30，20.0% | 9.5%–37.3% | 21/30 | 3/30 | 1.217 / 1.139 |

均无超时；当前同步修复后的真实结果为`artifacts/corrected/evaluation_sync/summary.json`与`episodes.json`。这些是**本项目新训练权重**的结果，区别于前面的官方权重零样本迁移。它表明短程记忆增强PPO已经学到一定圆柱环境导航，但密集场景仍以碰撞为主。速度变化较小、且尚未进入原最终奖励阶段，不能据此宣称充分验证自适应速度。

同权重、同地图逐帧清空LSTM隐状态后，到达率分别降为19/30=63.3%、11/30=36.7%、3/30=10.0%；相应碰撞率30.0%、53.3%、86.7%。这是推理分布扰动诊断，提供历史状态利用的初步证据，不能替代等预算current-only重新训练消融。`artifacts/corrected/validation.json`确认保存加载动作一致、表征精确冻结、全部loss有限、训练/验证数据场景无重叠，以及评估任务与已记录的完成训练任务无重叠。

### 真实策略重放发现的同步问题及修复

同步前第一次重复评估的90个终止原因一致，但少数轨迹连续量不同；某次reset首帧深度均值出现大差异，因此没有将它标记为逐轨迹可重复。适配层现显式等待Warp GPU深度生产任务结束后再由Torch读回。修复后的`evaluation_sync/`与`evaluation_sync_replay/`两次独立90任务评估，指标、episode和逐秒轨迹JSON全部一致，到达率仍为83.3%/53.3%/20.0%；证据见`replay_sync_validation.json`与`audit/CAPTURE_SYNC.md`。

短程训练和16k帧数据是在该同步修复前生成的，不能追溯重标为修复后的数据。现已用修复后的环境验证同一权重；长预算训练会重新采集数据并从头训练表征和final PPO。完全重复性结论仅覆盖本次相同硬件、驱动与配置的两次运行。

扩展相机检查也已完成：严格逐像素比较失败，差异仅限8个深度边缘大差异像素及最多1灰度的其余差异；几何位姿/排序/重置检查通过。详见`docs/ENVIRONMENT.md`，不将近似一致称为像素级等价。

## 尚未通过的验收项

1. 完成原定表征及8e6 PPO预算，并验证混合密度/final奖励阶段的收敛；目前只有短程稀疏预热策略结果。
2. 多训练种子、每密度足够任务的正式成功/碰撞/速度评估。
3. 等预算current-only与past+current重训，以及固定速度策略重训消融。
4. Flightmare与Isaac完整相机外参的逐姿态运行对照；2025最终论文PDF目前未成功获取，已阅读2024预印本并以源码核对实现。

没有将未执行代码、运行中的作业或官方预训练权重结果标记为新训练基线成功复现。完整审计与修复记录见`audit/EXPERIMENT_AUDIT.md`和`docs/SOURCE_AUDIT.md`。
