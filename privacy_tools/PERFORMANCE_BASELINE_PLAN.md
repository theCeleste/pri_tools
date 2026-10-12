# 效用对比与性能对比改造规划

## 目标

微调任务在选择方法后，将原来的“对比方法”拆分为“效用对比方法”和“性能对比方法”。效用对比沿用各方法现有 baseline；性能对比统一运行 DP-SGD，只展示内存、运行时间和吞吐率。

## 实验映射

| 方法 | 效用对比（`baseline`） | 性能对比（`performance_baseline`） | Python |
| --- | --- | --- | --- |
| ClipText | 原始数据 LoRA | `dpsgd/dpsgd.py` | `python.dpsgd` |
| PTCDS | 原始数据 LoRA | `dpsgd/dpsgd.py` | `python.dpsgd` |
| Kal-DP | `Kal-DP/Peft.py` | `Kal-DP/dpsgd.py` | `python.kal_dp` |
| ZO-DPGA | LoRA + MeZO | `Kal-DP/dpsgd.py` | `python.kal_dp` |
| DADP-ICL | 原有非隐私 ICL | `dpsgd/dpsgd.py` | `python.dpsgd` |
| PrivSV | 无噪声 steering vector | `dpsgd/dpsgd.py` | `python.dpsgd` |

新增实验 ID 为 `performance_baseline`，原有 `utility`、`baseline`、直接评估和间接评估 ID 保持兼容。

## 配置和运行环境

在 JSON 中使用独立的 `python.dpsgd`、`dpsgd` 和 `kal_dp_dpsgd` 参数段。`dpsgd/dpsgd.py` 默认运行 1 个 epoch，`Kal-DP/dpsgd.py` 默认运行 2 个 epoch，二者分别从各自配置读取。两条脚本均接收当前模型、数据集路径（如适用）、输出目录和 DP-SGD 参数。Kal-DP/ZO-DPGA 性能对比复用 Kal-DP 环境及脚本；其余四种方法使用 `python.dpsgd` 和 `dpsgd/dpsgd.py`。

环境检查、依赖检查、GPU 检查和命令打印必须展示本次实验实际使用的 Python 路径。性能对比不应错误使用所选方法的 Python 环境。

## 结果规则

- `utility`：展示并记录准确率、内存、运行时间、吞吐率。
- `baseline`：展示并记录准确率，其他性能字段留空。
- `performance_baseline`：展示并记录内存、运行时间、吞吐率，准确率字段留空。

子进程原始日志可保留全部日志；“只需输出”约束正式汇总和 CSV 结果。

历史结果展示按实验类型过滤：效用对比只显示准确率，性能对比只显示 DP-SGD 的三项性能指标。真实运行失败时不展示历史结果。

## `dpsgd/dpsgd.py` 适配

保留现有 DP-SGD/Opacus/LoRA 核心逻辑，增加或适配本地 SST-2 数据路径参数，确保与统一入口的数据集配置一致；从 `dpsgd.num_epochs` 显式传入 `--num_epochs`（默认 1）。输出 `perf_summary.json` 等结构化性能报告供统一指标提取。

## `Kal-DP/dpsgd.py` 配置

Kal-DP/ZO-DPGA 性能基线从 `kal_dp_dpsgd.num_epochs` 读取 `--epochs`，默认值为 2；同一值也用于运行时副本中的 Opacus 隐私会计参数。

## 验收标准

1. 微调菜单出现“当前方法 / 效用对比方法 / 性能对比方法”。
2. Kal-DP 效用对比调用 `Kal-DP/Peft.py`。
3. Kal-DP、ZO-DPGA 性能对比调用 `Kal-DP/dpsgd.py`。
4. 其余四种方法性能对比调用 `dpsgd/dpsgd.py`，并使用 `python.dpsgd`。
5. 通用 DPSGD 命令显式包含配置中的 `--num_epochs`（默认 1），Kal-DP DPSGD 命令显式包含配置中的 `--epochs`（默认 2）。
6. 六种方法共 30 个运行项可构造、可预演并通过结构审计。
7. 三类微调结果的正式输出字段符合上述规则。
