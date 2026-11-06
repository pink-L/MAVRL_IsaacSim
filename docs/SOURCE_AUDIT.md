# MAVRL 源码审计与迁移对照

审计日期：2026-11-04（Asia/Shanghai）。本项目目标是 **Isaac Sim 静态圆柱环境中的算法复现**，不是原始 AvoidBench 数值复现。未修改其他研究项目。11月5日按用户要求复核并行性能时，只读查阅了LargeMap的一套配置和传感器实现，用于解释负载差异，未移植其算法。

## 固定来源

- MAVRL：https://github.com/tudelft/mavrl ，commit `813b8875ba00c838e08286a3f8518c6167546f74`；完整原始 Git 克隆见 `upstream/mavrl`，不修改。
- AvoidBench：https://github.com/tudelft/AvoidBench ，commit `b9afb971ea85a98ee98125764e93513c9cf64457`；原始源码见 `upstream/AvoidBench`，不构建、不修改，不下载 Unity standalone。
- Yu、De Wagter、de Croon，IEEE RA-L 2025，10(2):1441–1448，DOI `10.1109/LRA.2024.3522778`。出版元数据由 TU Delft Research Portal 核实。
- 已阅读本地 `docs/mavrl_arxiv.pdf` / `.txt`，来自 https://arxiv.org/abs/2402.08381v1 （2024-02-13）。**该版本是预印本，不能冒称已阅读 2025 最终版**。TU Delft 最终版链接返回 HTTP 403；仓储页面读取失败。正文与最终版差异仍须在获取最终版后复核；实现参数以固定源码为准。

## 网络和训练代码定位

| 内容 | 官方路径 | 核实结果 |
|---|---|---|
| VAE | `models/vae.py`；`trainvae.py` | 输入 1×256×256；6 层 Conv2d，通道 8/16/32/64/128/256，kernel=4、stride=2，无 padding；flatten=1024；均值和 logσ 各 64 维 |
| VAE decoder | `models/vae.py::Decoder` | FC 64→1024，reshape 为 1024×1×1；转置卷积通道 128/64/32/16/8/1，kernel=5/5/6/4/5/4、stride=2；最后 sigmoid |
| VAE loss | `trainvae.py::loss_function` | 所有像素/样本求和的 MSE 加 `-0.5 Σ(1+2 logσ−μ²−exp(2 logσ))`；Adam 默认 lr=1e-3，batch=32；训练随机水平翻转，验证不翻转；plateau 调度、early stopping |
| LSTM | `mav_baselines/torch/recurrent_ppo/recurrent/policies.py` | 输入为 encoder **均值**，并非每帧重新采样；64→256、单层 LSTM；状态向量不进入 LSTM；初始 PPO 共享 latent，重载后的训练路径建立独立 critic LSTM，详见下文 |
| 重建 | 同上 `predict_lstm`；`train_lstm_without_env.py` | FC 256→3×64；三个目标共用一个**独立随机初始化** decoder（不是复制 VAE decoder）。encoder 固定，actor LSTM、FC、decoder 联合训练；lr=1e-4、Adam eps=1e-5；主设置 `[1,1,0]`，历史间隔 10 帧=1 秒；loss 为 past/current MSE 求和 |
| PPO actor/critic | `train_policy.py`；`policies.py::_build_mlp_extractor` | LSTM256+状态7=263 维；actor ReLU MLP [256,256]；critic [512,512]；独立输出头 |
| 动作分布 | `ppo_recurrent.py::_setup_model/collect_rollouts` | **tanh 作用于 Gaussian 均值**，不是 squashed Gaussian。采样后执行时 clip[-1,1]；PPO 使用未裁剪采样动作的 log-prob；初始 log_std=-0.5，重训重置为0 |
| PPO 冻结机制 | `collect_rollouts/train/get_ppo_need` | rollout 在 no_grad 中计算并缓存 recurrent latent；优化仅使用缓存 latent，故 CNN/LSTM 实际不被 PPO 更新。本项目显式 requires_grad=False，并检查权重摘要不变 |
| PPO 超参数 | `train_policy.py` | lr=1e-4×剩余训练比例；γ=.99，GAE λ=.95，n_steps=1000，batch=4000，epochs=10，clip=.2，entropy=0，vf=.2，max_grad_norm=.5，8e6 transitions |
| curriculum | `ppo_recurrent.py::collect_rollouts` | forest 前1.2e6 transitions 使用稀疏地图(radius=10)，之后变复杂并调用 resetRewCoeff；不是固定一套 reward |

## 环境：实际 C++ 行为

### 重要：官方初始与重载路径的 critic 不同

`ppo_recurrent.py::_setup_model` 在初始训练显式设置 `shared_lstm=True, enable_critic_lstm=False`。但 `train_lstm_without_env.py` 和 `train_policy.py --retrain` 从保存的 `data` 重建 policy；保存参数没有包含这两个 recurrent 标志，于是使用默认的 `False/True`，建立一个独立 critic LSTM。公开 `iter_00540.pth` 和 `iter_00640.pth` 均包含独立、不同于 actor 的 critic LSTM 张量。离线重建只更新 actor LSTM，critic LSTM 不参与重建；重训 PPO 仍冻结两套 recurrent 特征，只更新 MLP。

迁移版现按该**实际执行路径**处理：initial 共享；离线 LSTM 阶段建立独立 critic；final PPO 缓存两套特征。早期 `artifacts/smoke` 使用共享 critic，不能视为忠实最终算法；旧最终权重禁止直接作为修正后的正式续训起点。原官方零样本评估只使用 actor，动作不受这项修正影响；新的 `iter_00540_full.pt` 转换器保留 critic 参数。独立复核见 `audit/CRITIC_SOURCE_REVIEW.md`。不把这个源码重载行为包装成新算法。

主体为 `upstream/AvoidBench/src/avoidbench/avoidlib/src/vision_envs/avoid_vision_envs/avoid_vision_envs.cpp`；配置为 `upstream/mavrl/configs/control/config_new_out.yaml`。CLI `scene_id=1` 选择 outdoor 配置，但该配置内部 `unity.scene_id=3`。因此 **forest 使用 scene 3 的终止奖励**，不能直接套 scene 1 的 -10。

- 步长 0.1 s，动作 `[ax,ay,az,yaw_rate]`；对 [-1,1] 进行线性缩放到 `[±4,±4,±1,±0.6]`。配置 max_t=5 并非实际 episode 上限：C++ 硬编码 `t>100`。
- 世界位置 RFU、机体系速度/加速度 FLU。`body2world` 先 `R(q)·body`，再 `[−y,x,z]`。保持源码中的速度存储语义：更新 yaw 时**不**将速度重新旋转来保持惯性速度不变。
- `p_next=p+R_world(v·dt+0.5 a·dt²)`，`v_next=v+a·dt`，yaw 从上一步完整姿态提取后加 yaw rate·dt。保留源码启用的 pitch/roll：由 `[-ay,ax,az]+[0,0,9.81]` 构造倾斜四元数，组合顺序为 `q_att=q_pitch_roll*q_heading`。没有力、电机、质量、低层四旋翼控制器。
- 状态实际顺序：`[log(1+d_xy), |v_xy|, atan2(-dx,dy), atan2(vy,vx), dz, vz, yaw]`。这与论文公式列出的顺序不同，必须以代码接口为准。
- 深度：官方 stereo SGM，256×256、70° FOV、baseline .2 m，左相机 offset `[.2,-.1,.3]`；截断 12 m、线性映射到0–255，再/255输入CNN。迁移采用 Isaac RTX `distance_to_image_plane` 理想深度；SGM 噪声/缺失像素没有复制，是明确的传感器简化。
- initial reward 六项系数 `[-.001,-.001,-.002,-.003,-.0005,-.003]`，对应 log distance、超过3m/s时的速度、|dz|、|track+yaw−bearing|、|a−a_prev|、|track|。final 系数 `[0,-.05,0,-.0015,-.0001,-.0015]`。角度误差源码**没有 wrap**；保持此行为。`colli_coeff` 实际未参与逐步 reward。
- 终止优先级：越界 → 超时 → 碰撞 → 到达；到达为 xyz 各自误差<.4 m。forest 越界−2、超时−5、碰撞−2、到达20/TRAV。终止奖励**覆盖**当步总奖励（`vec_vision_env_base.cpp::perAgentStep`），不是相加。
- 碰撞原实现：沿步进线段对 point cloud 做球形查询；半径来自 `drone.width/2=.1`。迁移改为线段与膨胀实心圆柱精确相交，可避免点云/线段离散漏检。
- TRAV 原实现：1m 网格、8个水平射线、.1m步进，计算射线到障碍/地图边界的平均自由距离。本项目对圆柱计算相同量的解析近似，未使用凭密度随意指定的常数；point cloud离散误差和网格末端采样差异记录为几何适配。

## 迁移选择和影响

| 类别 | 保留/变更 | 影响与理由 |
|---|---|---|
| VAE代码 | 运行时直接加载未经修改的 `upstream/mavrl/models/vae.py` | 网络与VAE采样完全复用 |
| PPO/LSTM运行层 | 将官方 recurrent latent 缓存、网络、损失和更新方式移植到独立 PyTorch 模块 | 旧实现依赖 Gym、SB3旧接口、Flightgym、ROS bag，并有未导入sys/mav_baselines的入口错误；不改共享 Python 包，不安装旧torch覆盖Isaac |
| PPO核心 | 保留网络维度、Gaussian/tanh/clip语义、GAE、clipped surrogate、Adam eps=1e-5、系数和梯度裁剪 | 无额外算法模块；不是直接用通用 RecurrentPPO 联合训练CNN/LSTM |
| 地图 | 森林/灌木→贯穿允许飞行高度的圆柱 | 允许的几何简化；稀疏/中等/密集、半径混合、不同分布；独立地图与种子，每次任务重置更新地图 |
| 起终点 | 按原森林三种对边任务方向采样；额外连通性检查 | 去掉明显不可行任务；未清空直线路径，仍允许必须绕行 |
| 初始策略场景 | 空障碍场景 | 遵循论文/README分阶段方案；公开训练入口实际用radius=10稀疏地图，此差异明确保留记录 |
| 感知 | 单个RTX深度相机代替立体SGM | 保留图像尺寸/FOV/预处理/安装偏移；不宣称传感器噪声等价或可直接sim-to-real |
| 并行 | 单Isaac进程、独立空间偏移、每环境相机 | 不引入OmniDrones动力学；现有IsaacLab/OmniDrones仅审计，不复制其配置 |
| 数据集 | 确定性均值动作采集；按scene_seed拆分train/validation；连续序列不跨episode；最大500帧 | 原采集rollout/batch为500；迁移按真实episode截断、不加入padding目标。LSTM序列起点置零，避免使用旧随机LSTM隐藏状态。冒烟显式改为64帧；这是训练数据组织差异，并非完全相同的批次 |
| 运行错误修复 | reset时同步观测与LSTM reset mask | 原collect_rollouts存在env.reset()返回值未赋给_last_obs的风险；不保留这类状态不同步错误 |
| 短程实验 | 命令行显式降低steps/horizon/epochs和warmup | 仅功能/短程训练，配置单独保存；不能当作8e6步正式结果 |

`--steps` 现仅控制停止预算；学习率默认仍按8e6 transitions衰减。`--schedule-steps` 才显式改变衰减计划。早期 smoke 与256k基础策略的学习率在短预算末尾衰减到零，这些旧结果已单独标记。第三种起终点方向的 y 范围已从旧版[0,40]恢复源码[0,34]；旧官方权重评估仍保留原始地图与配对重放结果，不覆盖历史数据。

相机朝向目前按运动学前向与 FLU→RFU 转换实现。已验证深度距离、平移和部分静态帧模式比较；新增`check_camera_parity.py`覆盖yaw/pitch/roll、环境排序和地图重置，其执行状态见RESULTS。Flightmare的Unity端相机组合变换未运行逐姿态对照，**不宣称两平台完整相机外参/图像逐像素等价**；尤其不能用静态深度一致性替代动态姿态一致性验证。

## 当前边界

不含 C3BF、动态障碍、行为克隆、教师策略、额外奖励、势场或安全过滤器。未训练和未评估的能力不得作为完成项。验证状态以 `docs/RESULTS.md` 和磁盘中的原始日志为准。
