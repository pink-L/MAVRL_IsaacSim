# 审计发现与处理记录

原始独立审计报告保持不变，本文件记录执行者的修复，不替代独立复核。没有将同家族Type-A审计称为跨家族认证。

| 发现 | 修复 | 验证/历史结果处理 |
|---|---|---|
| F1：续训默认initial | 未指定phase时从checkpoint继承；显式冲突报错；配置同样继承 | `test_run_integrity`回归测试；旧smoke未使用resume |
| F2：采集随机动作 | 改用Gaussian均值，保持官方deterministic=True | mock环境验证实际传入动作；manifest记录；旧4029帧不覆盖 |
| F3：LSTM lr=3e-4 | 恢复1e-4 | 配置测试和后续运行快照；旧loss/权重标记历史变体 |
| F4：相机/源码证据不完整 | 新增可复跑相机capture/compare脚本；保存原始帧、姿态测试、文件SHA；所有新run保存启动源码快照 | 新脚本执行以camera_checks产物为准；不追溯认证旧checkpoint版本 |
| F5：第三方向y范围 | [0,40]恢复[0,34] | 多种子测试；旧官方评估和旧图保持历史分布 |
| F6：tracker滞后 | 更新RESULTS和tracker | 明确旧smoke 0/36成功，正式实验未完成 |
| 补充：最终critic结构 | 初始共享；离线LSTM时新建独立critic；final PPO分别缓存actor/critic特征；两套LSTM冻结 | 直接执行原forward_rnn与移植step_both对照；旧shared-critic final不能作为正式续训起点 |
| 补充：公开权重遗漏critic | 新转换文件保留lstm_critic，不覆盖旧actor-only文件 | 旧导航只用actor，结果保持；full转换的value需使用对应critic特征 |
| 补充：短预算LR过早归零 | 停止预算与LR计划分开；默认仍8e6衰减 | 512k新基础训练计划；旧8192/256k保留原配置 |
| 补充：序列长度 | 默认最长500帧，与官方采集rollout/batch对应；smoke显式64 | 按episode截断、起点零隐藏状态仍是数据组织适配，已明确记录 |
| 补充：真实策略重放首帧深度异常 | 在CameraView Warp→Torch数据交界显式等待GPU生产任务完成 | 修复后独立两次90任务的指标/episode/轨迹完全一致；旧训练数据保留，formal重新采集；见CAPTURE_SYNC.md |

版本锚点：`216e9e1`为第一份事后代码快照；`6c2e89c`修复F1/F2/F3/F5和运行源码记录；`b8f4e3b`补齐实际critic路径及相机核验入口。初始新训练在6c2e89c启动；其共享critic正是官方初始路径，后续critic修复不影响该阶段。源码快照和每阶段运行元数据优先于当前工作树来解释实验。
