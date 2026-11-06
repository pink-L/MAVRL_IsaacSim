# 环境与隔离

正式运行参数与最新交付命令见[HANDOFF.md](HANDOFF.md)。以下环境版本为原始审计记录；早期相机吞吐与同步试验按时间保留，最终适配层使用`tiled + synchronize_depth + rt_subframes=2`，正式PPO为64环境，性能复测见[PARALLELISM.md](PARALLELISM.md)。

2026-11-04 审计，原始记录：`audit/environment.json`、`audit/cpu.txt`、`audit/memory.txt`、`audit/installed_packages.txt`。

| 组件 | 已验证版本/状态 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，kernel 6.8.0-138 |
| GPU | NVIDIA RTX 5090 D，32607 MiB；审计时无其他计算任务 |
| NVIDIA driver | 580.178.04；nvidia-smi显示CUDA 13.0是驱动支持上限 |
| 实际Torch CUDA | PyTorch 2.7.0+cu128、runtime12.8；sm_120 CUDA张量运算通过 |
| Python | `/home/lz/miniconda3/envs/lz_env/bin/python`，3.11.16 |
| Isaac Sim | 5.1.0.0；headless RTX深度已实际运行 |
| Isaac Lab | 包元数据0.47.2，指向现有drones/IsaacLab；本实现未导入其任务 |
| OmniDrones | 包元数据0.1.1，指向现有drones/OmniDrones；本实现未导入其任务或动力学 |
| Gym/Gymnasium/SB3 | 0.23.1 / 1.2.0 / 2.9.0；SB3仅用于源码一致性测试与官方权重metadata兼容，训练核心为移植的PyTorch PPO |
| nvcc | 不在PATH；本项目不编译CUDA扩展 |

创建 `.venv --system-site-packages` 复用现有大依赖，未在共享环境pip安装/升级/卸载任何包。所有脚本由 `scripts/python.sh` 启动，工作目录固定为项目根；缓存、Kit portable root、matplotlib配置、日志均定向到项目内 `.runtime` 或 `logs`。`PYTHONDONTWRITEBYTECODE=1` 避免写原仓库pycache。未修改用户shell配置或其他研究项目。

## 渲染同步问题

首次仅执行world.step()后，相机annotator返回空数组；没有用空白伪图替代。改用Replicator显式同步step后正确输出。首次shader编译约140秒，缓存后启动约5秒。探针前表面预期4m、实测4.005m；平移1m后实测3.004m。数值小误差来自圆柱渲染曲面离散和像素中心。

## 已测资源

| 相机方式 | 环境数 | transitions/s，仅环境步进 | GPU显存MiB |
|---|---:|---:|---:|
| 独立Render Product | 4 | 103.6 | 3432 |
| 独立Render Product | 16 | 114.2 | 3863 |
| CameraView tiled | 16 | 559.8 | 3766 |
| CameraView tiled，显式GPU同步修复后 | 16 | 459.6 | 3764 |

基准为100步，不含启动、shader编译、PPO优化，不能直接当作训练端到端吞吐。相同种子的前4个相机在两种渲染模式下，8位预处理深度逐像素相同，见 `artifacts/tiled_parity.json`；这不是对任意姿态/场景的普遍等价证明。

扩展检查已运行：`artifacts/camera_checks`包含16环境×5种状态的原始NPZ帧、元数据及SHA。基准位置、前移1m、地图重置三组逐像素相同；yaw旋转有5个、混合pitch/roll/yaw有3个较大差异像素，全部位于互相3×3邻域可解释的深度不连续边缘，非边缘最大差1灰度。严格逐像素测试**失败**，记录保留为`comparison_strict.json`；显式允许≤0.01%边缘像素的几何近似检查通过。精确的渲染器数值原因未确定，不将该结果称为完整图像等价。两种模式的光轴中心深度、前移、yaw朝向、环境排序、几何隐藏/替换检查均通过。

复跑命令：

```bash
scripts/python.sh scripts/check_camera_parity.py --out artifacts/camera_repeat/individual
scripts/python.sh scripts/check_camera_parity.py --out artifacts/camera_repeat/tiled --tiled
scripts/python.sh scripts/check_camera_parity.py --compare artifacts/camera_repeat/individual artifacts/camera_repeat/tiled \
  --out artifacts/camera_repeat --allow-boundary-differences
```

默认配置保留原4环境；实验可用`--num-envs 16`。不把高层策略的10Hz与墙钟帧率混淆：每个策略动作始终代表0.1s。CPU线程限4，实验串行运行。启动新训练前可用`nvidia-smi`检查其他任务；不要同时运行多个训练脚本。

## 真实策略取帧同步

随后真实策略重放发现少数reset首帧异常；静态/短动态相机探针不足以发现这一问题。项目适配层在CameraView的Warp→Torch零拷贝交界显式同步GPU生产任务后再读回。修复后两次90任务评估的逐秒轨迹及episode记录完全一致。完整调查和原始异常数据均保留于`audit/CAPTURE_SYNC.md`。修复后基准459.6 transitions/s、3764MiB，另存`artifacts/benchmark_sync/`，不覆盖旧记录。
