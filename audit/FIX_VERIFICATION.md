# b8f4e3b 修复核验

日期：2026-11-04。审查者：独立 Codex，同家族 **Type-A**；不是跨家族认证。冻结版本：`b8f4e3b1bccb3342adf3b6e5bf192a073704e8cd`。本轮读取的实现、配置和测试文件SHA均与该提交相符，记录见 `FIX_VERIFICATION_EVIDENCE.json`。

**结论：PASS，限于本次修复范围。** 初始/重载/最终阶段critic拓扑、独立状态与缓存、公开权重完整转换、续训阶段恢复及确定性采集的修复成立。本轮未发现新增实际缺陷。该结论不表示导航已收敛，也不覆盖相机/物理仿真等未重跑的验证。

没有启动GPU、Isaac或训练流程，也没有执行optimizer.step。必要验证包括CPU前向、无参数更新的反向传播、临时保存重载和现有单元测试；`torch.cuda.is_initialized()`始终为false。临时文件放在`audit/`下并自动清理。实现和此前审计文件均未修改。

## 修复逐项核验

| 核验项 | 结论与直接证据 |
|---|---|
| 初始PPO共享actor recurrence | **PASS**。`mavrl_isaac/models.py:24`–`25`初始只有actor LSTM，critic为空；`step_both`在`47`返回同一latent/状态。`training.py:70`保持此初始化，与官方`ppo_recurrent.py:257`–`266`的显式共享设置相符。 |
| 离线LSTM重载后建立独立critic | **PASS**。`representation_training.py:72`–`73`加载初始checkpoint后创建独立critic；`77`–`81`只训练actor recurrence、projection和decoder，固定encoder与critic。重建函数`52`–`65`仅调用actor LSTM，与官方`policies.py:614`的路径一致。 |
| 最终PPO使用并保留独立critic | **PASS**。`training.py:74`–`76`要求新final运行包含critic；`86`–`87`也阻止旧共享critic final checkpoint以源码匹配名义续训。这些检查不会把旧权重静默改成不同拓扑。 |
| 双序列状态及episode reset | **PASS**。`models.py:42`–`49`分别更新actor/critic隐状态，并对同一episode-start mask各自清零。`training.py:90`初始化两套状态，`98`在curriculum reset时同时清零，`102`分别传递。原始官方`_process_sequence/forward_rnn`对照测试和独立多步对照均通过。 |
| Rollout缓存、bootstrap与PPO输入 | **PASS（源码路径核验）**。`training.py:102`–`107`分别缓存两个latent；`113`–`114`使用critic状态估计bootstrap值；`115`–`119`分别展平和传递缓存；`ppo_update`的`51`明确使用critic_features。bootstrap产生的新状态被丢弃，下一rollout仍在旧状态上处理该观测，因此不会重复推进持久隐状态。 |
| 冻结与保存加载 | **PASS**。`training.py:21`按保存的`critic_lstm.*`键恢复拓扑，并在`22`冻结；`save_checkpoint`在`30`保存完整representation。实测保存重载后actor/critic及heads张量完全一致、critic仍存在且所有representation参数被冻结。`training.py:131`–`132`的摘要覆盖独立critic。 |
| 公开权重完整转换 | **PASS（给定00540权重）**。`scripts/import_official_checkpoint.py:25`–`27`现已加载全部critic LSTM参数。本轮独立建立源→目标映射：原权重53个张量全部覆盖，遗漏0、数值不相等0，包含actor、critic、encoder、decoder、projection、MLP及输出头。 |
| 续训阶段与配置 | **PASS**。`scripts/run.py:9`–`16`从checkpoint恢复phase、拒绝冲突；`20`–`21`在未显式传配置时恢复checkpoint配置；`58`不再默认强制initial。`training.py:80`再次校验phase。已有回归测试确认final phase、seed、预算及原schedule恢复。 |
| 数据采集确定性 | **PASS**。`dataset.py:31`显式`deterministic=True`，`41`记录采集模式；mock collector测试验证实际传入的标志与执行动作，而非只查字符串。旧smoke数据仍属于旧随机采集产物，不能据本次代码修复重新标注。 |

本轮还确认已有回归测试覆盖LSTM学习率`1e-4`、第三类mission的y范围`[0,34]`和运行源码快照摘要；它们均通过。本轮未扩展到新仿真评估。

## 原始公开权重与转换权重的独立数值对照

原文件：`upstream/mavrl/ckpts/available_policy/iter_00540.pth`，SHA-256：

```text
a530396e060fb31f431344d701fd27241e1afed59d8b307a6f5b0a619a1a7e80
```

转换文件：`artifacts/official/iter_00540_full.pt`，SHA-256：

```text
aef7c10c7b3809a3fe4108c088ac11892efd639312159e96618eebb3f9e30851
```

转换文件记录的`source_sha256`与原文件一致。审查独立加载原权重的官方encoder、两个LSTM和两组MLP/输出头，并提取官方`policies.py`中的`_process_sequence`和`forward_rnn`实际AST执行；不使用转换文件作为参考权重。对照输入为已有NPZ中的18组图像/状态，组织为6个连续计算步×3个并行序列，包含全部复位、不中断和部分序列复位。输入只是CPU对照fixture，不被当作新的环境轨迹。

| 对照量 | 六步中的最大绝对误差 |
|---|---:|
| Actor Gaussian均值 | **0.0** |
| 独立critic value | **0.0** |
| Actor latent | **0.0** |
| Critic latent | **0.0** |
| 两套hidden/cell state | **0.0** |
| 在参考均值动作处的log-prob | **0.0** |

比较使用`rtol=2e-5, atol=3e-7`断言；实际观察误差均为零。此前报告的value差异已在给定新转换文件和该调用路径上消除。此验证没有运行完整旧版SB3/Flightgym入口，因此不宣称整个旧运行时逐位等价。

冻结检查还验证了：两套缓存latent均`requires_grad=False`；对heads做反向时representation梯度全为None、heads存在梯度；离线重建反向时actor LSTM存在梯度，critic LSTM和encoder梯度全为None。没有进行权重更新。

## 测试结果与适用边界

`tests/test_source_parity.py`与`tests/test_run_integrity.py`共 **9项测试通过，0失败、0错误**。完整记录为`FIX_VERIFICATION_TESTS.log`，机器摘要为`FIX_VERIFICATION_TESTS.json`。测试里的临时`.runtime`路径在审查进程内重定向至`audit/`，未修改测试代码。

`tests/test_source_parity.py:44`–`64`直接比较官方forward在共享及独立critic两种拓扑下的latent与状态，覆盖了上次缺失的分支。不过，`test_collector_uses_mean_action_and_keeps_episode_splits`的mock未产生done；它证明确定性动作与scene-seed拆分，不能单独证明跨episode数据正确性。该目标仍主要由`dataset.py:27`–`36`的append/flush/group顺序支撑，本轮未发现其中新增缺陷。

本轮对训练缓存/GAE的调用连接做了直接源码核验，并保留了既有GAE终止隔离测试；没有运行完整PPO训练，也没有新增同批次optimizer更新对照。CPU前向等价不能替代未来训练表现评估。独立critic的随机初始化采用本项目seed/RNG顺序，不宣称与未提供的官方初始/离线训练checkpoint有相同随机实现。

`evaluation.py`与`dataset.py`的动作推理仍可只调用actor recurrence，并丢弃`act`返回的value；这对动作没有影响。需要验证或使用完整官方value时，必须调用`step_both`并传`critic_x`，本次对照正使用此路径。未把被丢弃的actor-latent value当作官方critic证据。

修复后的实现可以继续针对本次核验范围开展新实验。旧共享critic结果、原随机采集数据及此前审计结论应保留各自版本标识；不能将本轮PASS套用到历史训练的算法拓扑或导航能力。
