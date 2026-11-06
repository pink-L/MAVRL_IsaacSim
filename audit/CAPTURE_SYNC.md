# 深度取帧同步调查

2026-11-04。触发条件：同一512k策略、同一配置/种子重复90个任务时，所有终止原因相同，但23条episode的连续量不同；中等密度平均速度差约0.00417m/s。最早的明确异常为sparse/100019重置时刻：位置、种子相同，深度均值却从147.731变为234.733。这不是可用舍入误差解释的现象，不能报告逐轨迹可重复。

原始记录：`artifacts/corrected/evaluation/`、`evaluation_replay/`、`replay_differences.json`。这些结果保留，不覆盖。

本地Isaac Sim 5.1 `isaacsim.sensors.camera/camera_view.py`在`get_data`中用Warp CUDA kernel将tiled annotator缓冲区reshape，`get_depth`通过`wp.to_torch`提供零拷贝Torch视图。项目原先立即执行Torch CPU拷贝，没有在跨框架边界显式等待生产任务完成。官方[Warp互操作说明](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)强调共享缓冲区的stream顺序；该页面版本较新，实际执行接口依据本地Warp1.8.2核对。官方[Isaac 5.1渲染排障](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html)也记录异步渲染及场景快速变化可能引发的取帧问题。

诊断`probe_capture_sync.py`进行了160次场景/位姿更新，并在每次取帧中比较GPU同步前后拷贝及静态重复捕获。未捕获直接copy race或大图像差异，运行时`asyncRendering=false`；2 subframes诊断同样未出现异常。该短探针不能排除只在实际策略/GPU负载下触发的稀有时序问题，因而不声称已唯一锁定具体CUDA竞态。

适配层修复：CameraView返回depth后，调用Warp的device同步，再由Torch拷贝到CPU，确保共享缓冲区的生产任务已完成。没有增加运动学步、改变10Hz动作时钟或修改图像数值；仍使用1个render subframe。

实际修复验证：`evaluation_sync/`与`evaluation_sync_replay/`各独立运行三密度×30任务，两次`summary.results`、全部90条episode记录、逐秒轨迹记录完全相同。对应原始JSON的SHA一致性见`artifacts/corrected/replay_sync_validation.json`；仅summary文件中的运行时间/源码快照目录等元数据不同。到达率保持83.3%/53.3%/20.0%。该证据支持同步修复消除了本次观察到的重复性问题，但不把两次运行推广成跨硬件/驱动绝对确定性保证。

此前initial、dataset、VAE、LSTM和512k final PPO是在增加显式同步前生成的，均保留真实历史标识；它们的权重现在已在同步后的环境评估。后续formal数据将重新采集，正式表征及PPO将使用同步后的接口从头训练，不把旧数据改标成修复后产物。

## 2026-11-05正式评估中的再次出现

8m正式权重三次各300任务的评估，终止结局全部一致（到达100%/98%/89%）。第一次和第三次的episode、trajectory文件SHA完全相同；第二次有15条中等密度episode的连续量不同，同位置出现不同深度输入。第三次没有修改取帧代码，不构成根因修复。此前90任务的通过结果仍是有效的历史记录，但不能证明显式Warp同步已充分解决该问题。

证据：`artifacts/formal_seed0/replay_failure_audit.json`与`replay_three_run_audit.json`。主流程仍保留失败状态，重训消融尚未启动。新诊断`scripts/probe_policy_capture.py`在真实策略运行时连续捕获同一运动学状态两次，记录大于2灰度级的差异和前10个异常批次的图像。该诊断增加渲染次数、改变取帧时序，不能用其导航结果替代正式评估；正常训练和评估代码保持原样。

后续实测：该诊断完成3859对捕获，capture 1647出现1个异常批次，16环境均有差异，共567939个像素相差大于2灰度级，最大差异228。图像保存在`artifacts/formal_seed0/policy_capture_probe/frames_001647.npz`。这验证了未改变运动学状态时连续取帧仍可能不同，但尚未判明具体渲染/缓冲区机制。2026-11-05 10:01开始复用相同历史数据的current-only离线LSTM；此举不依赖在线相机，不构成对取帧问题的验收。

### 固定状态参考与候选子帧设置

`replay_capture_fixture.py`以异常记录中的16个场景seed、位置和完整旋转矩阵重新建立场景。新进程连续10帧均逐像素等于原第二帧，与原第一帧的MAE为14.2519灰度、567939像素相差大于2灰度。结果见`capture_fixture/result.json`。该参考确认原第一帧异常，尚未确定具体渲染器内部触发点。

运行时检查asyncRendering、asyncRenderingLowLatency、throttling enable_async、replicator asyncRendering均为false；没有证据将本次异常归因于这些开关。Isaac 5.1本地`orchestrator.step`说明渲染子帧期间暂停仿真状态，[官方排障文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html)建议针对图像残影/场景更新测试更多RTSubframes。现仅将子帧数做成可记录的配置项，候选`configs/isaac_capture2.json`设为2，原默认仍为1；不改变0.1秒运动学、奖励或网络。资格检查为同状态双帧探针及两个独立300任务评估的完整episode/trajectory一致性。

若候选通过，`recover_capture.py`将使用相同的候选取帧设置分别重新训练past+current和current-only的PPO头，避免两组消融PPO训练时采用不同取帧设置。两者仍共享历史数据和VAE，各自复用已有LSTM，原始来源配置保留；这不是两个独立训练种子，也不宣称历史数据已重新采集。旧8m基线及失败状态保留。

### 18:37资格与重训更新

双子帧3858对检查未发现超过2灰度级的异常；两个独立进程各300任务的完整episode/trajectory文件和summary.results全部一致。恢复队列于12:19通过资格检查并启动新主基线。该8m PPO已完成，53个模型张量与旧同seed训练逐元素一致，排除资源耗时字段后125条学习日志也一致；新权重首轮导航评估保持100%/98%/89%。新基线自身的完整评估和检查尚在收尾。

上述证据支持在当前测试范围采用双子帧接口设置，并不能证明渲染器内部唯一根因或跨硬件的绝对确定性。资格检查、模型比较分别保存在`artifacts/capture_recovery_20261105/capture_qualification.json`和`artifacts/capture2_baseline_seed0/training_repeat_comparison.json`。

## 2026-11-06三组完整复测结果

baseline、current-only、fixed三组均完成8m最终PPO及各300任务的正常、清零、重复评估；正常与独立重复评估的`episodes.json`及`trajectories.json`分别SHA完全一致。各组`memory_figures_full/provenance.json`记录具体摘要。本轮取帧配置的检查通过，不代表渲染器内部异常触发点已被唯一定位，也不追改历史数据/VAE/LSTM的单子帧来源。完整结果见`docs/REPRODUCTION_REPORT.md`。
