# 实验跟踪

| Run ID | 阶段 | 目的 | 状态 | 证据 |
|---|---|---|---|---|
| R001 | M0 | 环境与来源 | 完成 | audit/environment.json；docs/SOURCE_AUDIT.md |
| R002 | M1 | Isaac深度距离与时序 | 完成 | logs/isaac_probe_sync.log |
| R003 | M1 | 动力学、奖励、场景、网络、reset、审计修复 | 17测试通过 | logs/tests_critic.log |
| R004 | M1 | 4/16相机基准 | 完成，tiled约560 transitions/s | artifacts/benchmark_4.json；benchmark_16.json；benchmark_16_tiled.json |
| R005 | M2 | 旧版六阶段冒烟 | 功能完成；36任务全部失败；已知源码偏差 | artifacts/smoke/；docs/RESULTS.md |
| R006 | M3 | 旧版256k基础导航 | 完成，独立空地图7/30到达 | artifacts/short/initial_evaluation/ |
| R007 | M3 | 正式单训练种子导航 | seed0的8m PPO和三密度评估完成，重复性问题未闭环；用户于11月5日取消seed1/2验证 | artifacts/formal_seed0/long_run/status.json；docs/DELIVERY_PLAN.md |
| R008 | M4 | 等预算记忆/速度消融 | current-only离线LSTM于11月5日11:53完成2000轮，PPO待取帧调查后执行；固定速度尚待实施 | artifacts/ablation_current_seed0/long_run/status.json |
| R009 | M3 | 官方权重零样本迁移 | 完成，3密度各20任务 | artifacts/official/evaluation/ |
| R010 | M4诊断 | 官方权重记忆清零与重复评估 | 完成；正常评估逐记录重放一致 | artifacts/official/evaluation_reset/；evaluation_replay/；figures/provenance.json |
| R011 | M4诊断 | 官方权重速度输入探针 | 完成，非物理飞行 | artifacts/official/speed_probe.json |
| R012 | 审计 | 独立完整审计与critic专项 | 原WARN保留；限定范围修复核验PASS | audit/EXPERIMENT_AUDIT.md；CRITIC_SOURCE_REVIEW.md；FIX_VERIFICATION.md |
| R013 | M3 | 修正后512k基础策略 | 完成，独立50/50空地图到达 | artifacts/corrected/initial/；initial_evaluation/ |
| R014 | M2/M3 | 修正后完整短程流水线与三密度评估 | 完成：512k final稀疏预热，到达25/30、16/30、6/30；清空记忆19/30、11/30、3/30 | artifacts/corrected/；完整预算尚未完成 |
| R015 | M1 | 扩展相机位姿/排序/重置检查 | 几何检查通过；严格像素比较失败，8个边缘大差异像素 | artifacts/camera_checks/ |
| R016 | M1 | 三种分布×三档密度场景检查 | 270任务均通过几何检查 | artifacts/scene_audit/report.json |
| R017 | M1/M3 | 真实策略重放与GPU深度同步修复 | 修复后两次90任务的episode/轨迹完全一致；当前459.6 transitions/s | artifacts/corrected/replay_sync_validation.json；artifacts/benchmark_sync/ |
| R018 | M3 | 同步修复后正式表征及8m PPO | VAE242轮、LSTM2000轮、PPO8m完成；100任务/密度到达100%/98%/89%；第二次评估15条连续记录不同，所有终止结局一致 | artifacts/formal_seed0/replay_failure_audit.json；validation.json；evaluation_full/ |
| R019 | 资源 | 单/双表征进程及16/32/64/128环境吞吐 | 完成；表征单进程、PPO64环境；正式PPO累计379.4 transitions/s | docs/PARALLELISM.md；artifacts/ppo_scaling_20261104/；artifacts/parallel_probe_20261104/ |
| R020 | 调度/M4 | 夜间串行消融与09:00阶段报告 | 主基线重复性检查失败后停止训练队列；09:00:27生成报告并退出；继续工作授权仍有效 | artifacts/overnight_20261104/status.json；MORNING_REPORT.md |
| R021 | M1/M3 | 第三次正式权重复测 | 第三次与第一次300任务episode/trajectory文件SHA完全相同；未修改取帧代码，因此不属于根因修复 | artifacts/formal_seed0/replay_three_run_audit.json |
| R022 | M1诊断 | 真实策略下同状态连续两次深度捕获 | 完成3859对，1对明显差异，16环境均受影响；额外渲染仅诊断，不替代正式评估，根因未闭环 | artifacts/formal_seed0/policy_capture_probe/status.json；capture_pairs.jsonl |
| R023 | M1诊断 | 重建异常状态 | 全部16场景/位姿在新进程连续10帧均逐像素等于原第二帧，确认原第一帧异常 | artifacts/formal_seed0/capture_fixture/result.json |
| R024 | M1诊断 | 双子帧取图 | 3858对同状态检查无异常；两次独立300任务episode/trajectory完全一致；通过本轮资格检查 | artifacts/capture_recovery_20261105/capture_qualification.json |
| R025 | M3 | 双子帧设置下主基线重训 | 8m PPO完成，22003秒；完整评估、清零记忆、重复评估和产物检查完成；到达100%/98%/89%，与原同seed主模型53张量一致；19:22修复绘图舍入错误后收尾完成 | artifacts/capture2_baseline_seed0/long_run/status.json；training_repeat_comparison.json；status_before_plot_recovery.json |
| R026 | M4 | 双子帧设置下两组机制对照 | current-only于11月6日02:04、fixed及配对检查于08:29完成；到达率分别100/99/89%和100/98/81%，记忆训练消融未显示到达率提升 | artifacts/capture_recovery_20261105/status.json |
| R027 | 资源 | 64/128/256/512环境真实RTX采样与PPO更新 | 完成，约335/328/329/335 samples/s，正式维持64环境；1024显存外推39.4GiB而未实测；非学习效果实验 | artifacts/ppo_scaling_20261105/results.json；decision.json；docs/PARALLELISM.md |
| R028 | 交付 | 三组完整结果、权重清单与复跑文档 | 完成：正常/清零/重复评估2700个episode，三组重放文件SHA一致；23项CPU测试通过；单种子结论及未支持的记忆增益明确记录 | artifacts/delivery_20261106/；docs/REPRODUCTION_REPORT.md；docs/HANDOFF.md |
