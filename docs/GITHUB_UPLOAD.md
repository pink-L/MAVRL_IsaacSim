# GitHub 发布范围

目标仓库：https://github.com/pink-L/MAVRL_IsaacSim 。用户选择上传代码与实验结果，继续排除大文件。

上传内容：

- `mavrl_isaac/`、`scripts/`、`configs/`、`tests/`、文档与已有源码审计记录。
- 静态交付报告、三模型的逐任务评估结果、配置、训练曲线数据、保存加载/冻结检查和配对统计。
- 3 个动态圆柱、30%/50% 动态比例的报告、逐任务结果、场景参数、几何检查、图表与小规模重复检查记录。
- `artifacts/dynamic_fraction_20261106/master_tracks.json`：动态比例配置与测试必需的共享轨迹清单。
- 原有 26 个本地提交的 Git bundle；不包含 `.git/config`、凭据或未跟踪的本地文件。

不上传训练 checkpoint、深度数据集、逐秒 `trajectories.json`、渲染帧/大型日志、运行源码快照、未修改的 upstream 克隆、虚拟环境、Isaac 依赖和缓存。`.gitignore` 中的相关目录排除规则继续保留；选定的轻量实验文件按明确清单加入版本控制。

原始证据文件按原内容上传，SHA256 不变。历史报告中的权重路径、完整轨迹路径、源码快照路径和机器绝对路径是来源记录，部分文件仅保留在实验机器上；不表示对应内容已在 GitHub 提供。若要运行训练、权重评估或需要全部历史原始数据的汇总脚本，应先补齐相应依赖与产物。

上传清单与各文件摘要见 `docs/REPOSITORY_FILES.json`。配置中的共享轨迹清单及其摘要在仓库中齐备，CPU 测试可在兼容 Python 环境和固定 upstream 源码准备后执行。GPU 复评仍需要未上传的模型权重。

本机没有 GitHub HTTPS/SSH 推送凭据，本次使用已连接账户的 GitHub API 发布代码快照；原本地提交历史通过 `history/local-history-through-e854c3a.bundle` 保留，不改写原始提交。
