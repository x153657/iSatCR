# ISACR 五种消融

五种算法为 DirectConcat、GAT、GraphSAGE、JKNet、NoSelfFeature。公共训练参数由 `train_4_5seed.build_training_config` 的 ISACR 配置生成，训练负载为 160 tasks/s、故障概率 0.18、场景种子 2020，模型种子为 0、1、2、3、4。五种消融与 ISACR 一致，先预热 151 步，再正式训练 1000 步，从 900 步起选择训练回报最佳权重。学习率为 0.001，gamma 为 0.99，repeat 为 10，batch_size 为 1024，经验池容量为 200000；各入口不再保留会被公共配置覆盖的旧参数字典。

训练固定参数直接继承 `train_4_5seed.build_training_config('ISACR_self_elimination', seed)`，仅覆盖消融结构、算法名称和输出路径。故障填充策略也直接继承，不在消融配置中单独硬编码；训练填充系数当前为 2.0。通信周期和心跳超时全部继承同一份 ISACR 配置，当前分别为 0.1 秒和 0.15 秒，GraphSAGE 不再单独覆盖它们。固定测试采用相同的任务与故障轨迹，测试场景种子为 2025，每项 300 步；350 项测试配置逐项与四算法 ISACR 的公共场景配置核对。旧零预热或旧聚合通信实现训练的模型不满足当前契约，不能通过改元数据复用。

## 训练与保存

`train_ablation.py --dry-run` 只生成计划。`train_ablation.py` 默认 10 并行，25 个模型排队执行，空闲后自动补齐，`--rerun` 将选定旧输出归档后重新训练。

目录为 `training_process_data/v1/model/算法名/seedN/`。GNN/DirectConcat 的权重目录内含 `gnn.pth` 和 `qnet.pth`；NoSelfFeature 保存 `.pth` 文件。每个种子保留最新和训练回报最佳检查点、检查点训练元数据、训练配置、训练日志、控制台日志和进程结果。完成判定同时核对最终步数、退出码、源文件/配置及权重哈希。旧五维模型不适用于本次六维实现。

## 固定测试

`test_ablation.py` 默认覆盖五算法、五种子，与四算法 ISACR 测试条件一致。负载为 100–160、步长 10，故障概率 0.18；故障扫描固定负载 160，概率为 0–0.18、步长 0.03，共 350 项。任务频率和任务/故障轨迹复用公共测试配置，输出按算法和模型种子分开保存。

测试前验证训练已完成、模型结构/参数配置、当前源文件与权重哈希。复用结果时重新解析原始日志，核对实际轨迹哈希和数量；不复用过期汇总。中断记录移入每个测试用例的 `previous_attempts`，不直接删除。表示差异和旧模型不兼容均不能通过修改元数据绕过。

```powershell
# 仅生成五算法、五种子、350 项测试计划，不加载模型、不运行仿真
& 'D:\anaconda\envs\xbb\python.exe' -B .\test_ablation.py --dry-run

# 两种测试全部执行，默认 10 并行；模型种子完成后自动接续下一个
& 'D:\anaconda\envs\xbb\python.exe' -B .\test_ablation.py --workers 10

# 分别执行负载或故障扫描
& 'D:\anaconda\envs\xbb\python.exe' -B .\test_ablation.py --experiment load
& 'D:\anaconda\envs\xbb\python.exe' -B .\test_ablation.py --experiment fault

# 指定多个消融、单个模型种子和测试点；--rerun 先归档旧记录再重测
& 'D:\anaconda\envs\xbb\python.exe' -B .\test_ablation.py --algorithms gat jknet --seed 0 --experiment load --load-point 160 --rerun
```

测试顺序与四算法入口一样，负载和故障概率均从高到低。结果根目录为 `training_process_data/v1/test/test/ablation/`，各算法的 `seedN/load` 和 `seedN/fault` 保存原始日志、配置、逐点汇总及 CSV；根目录生成两类全量 CSV、五种子均值 CSV 和 `TEST_STATUS.json`。`TEST_PLAN.json` 保存所选测试矩阵与固定轨迹信息。汇总只接纳当前代码契约及当前模型权重对应的记录，单点重测的进度仅统计本次所选测试点。

当前验证覆盖真实小拓扑信息交换、故障/恢复、实际双样本 DDQN 更新与管理器模拟调度，不代表完整训练后的性能结论。该次修复没有启动消融训练。
