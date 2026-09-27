# 具备MG信息保护能力的大模型微调支撑工具集

本工具把六套研究代码接入同一个命令行工具集。使用者只需按菜单选择，不需要记忆各方法原来的脚本、参数或目录。

集成层不会修改六份方法源码。遇到原脚本中的固定服务器路径时，只会在本次结果目录生成临时运行副本。

## 唯一入口

在解压后的项目根目录运行：

```bash
python3 privacy_tool.py
```

若当前环境中的命令是 `python`，也可以运行：

```bash
python privacy_tool.py
```

入口首先选择任务：

```text
1 微调
2 评估
```

微调流程为：

```text
微调
→ 梯度扰动 / 嵌入扰动 / 示例扰动
→ 具体方法
→ JSON 配置中的模型和数据集
→ 当前方法 / 效用对比方法 / 性能对比方法
```

评估流程为：

```text
评估
→ 直接评估 / 间接评估
→ 显式微调模型 / 隐式微调模型
→ 扰动类别
→ 具体方法
→ JSON 配置中的模型
```

选择“无扰动微调”时，待评估模型仅提供 ChatGLM3-6b 和 FM9G-7B（九格大模型）。两种模型的显式/隐式间接评估统一使用 evaluate/ 下的新版实现，直接评估继续预留；现有 Llama 评估接口保留在代码中，但不再通过该菜单提供选择。其他扰动分支仍使用 Llama 评估。

模型和数据集菜单只提供 JSON 中已配置的值，不提供自定义输入。其他实验参数和 Python 解释器不再询问，直接读取 JSON。立即执行前输入小写 `run`；输入 `0` 可取消。

## 六种方法的对比项

| 方法 | 本微调方法 | 对比微调方法 |
|---|---|---|
| ClipText | 脱敏文本 LoRA | 原始文本 LoRA |
| PTCDS | PTCDS 表示保护微调 | 原始文本 LoRA |
| Kal-DP | Kalman 梯度过滤差分隐私微调 | DP-SGD |
| ZO-DPGA | ZO-DPGA | LoRA + MeZO |
| DADP-ICL | 私有示例生成与 ICL 评估 | 非私有 ICL |
| PrivSV | 私有 steering vector | 无噪声 steering vector |

“本微调方法”记录准确率、峰值内存、运行时间和吞吐率；“效用对比方法”只输出准确率；“性能对比方法”统一运行 DP-SGD，只输出峰值内存、运行时间和吞吐率。Kal-DP 与 ZO-DPGA 复用 Kal-DP 的 DP-SGD，其他方法使用 `dpsgd/dpsgd.py`。

“直接评估”执行训练数据抽取评估；“间接评估”执行邻居成员推断评估。凡原始代码的 `.md` 已明确攻击步骤，统一入口均按其中规定的受保护/未保护、成员/非成员组合和评估参数执行。原始脚本中的硬编码路径仅在单次运行目录的临时副本中替换，六份源码不被改写。逐项对应关系见 [攻击流程对应表](privacy_tools/ATTACK_README_ALIGNMENT.md)。

## 第一次测试

先执行不加载大模型的交付自检：

```bash
python3 smoke_test.py
```

自检会检查六种方法共 30 个运行项的脚本、路径、参数和预演命令，不会下载模型或启动 GPU 训练。

然后启动：

```bash
python3 privacy_tool.py
```

第一次建议在“执行模式”中选择“查看最近历史结果”。该模式会生成完整命令并展示最近历史结果，但不会加载模型、占用 GPU 或写入正式实验指标。

示例路径：

```text
1 微调
→ 1 梯度扰动
→ 1 Kal-DP
→ 1 配置模型
→ 1 配置数据集
→ 1 当前方法
→ 2 查看最近历史结果
→ 确认
```

## 结果位置

```text
privacy_tools/results/<类别>/<方法>/<运行项>/
├─ experiment_metrics.csv
└─ <运行编号>/
   ├─ run_manifest.json
   ├─ stdout.log
   ├─ runtime_adapters/        # 仅在需要临时路径适配时出现
   └─ 攻击评估结果文件          # 直接/间接评估时出现
```

`experiment_metrics.csv` 每次真实训练追加一行，主要字段为：

- `privacy_budget_epsilon`
- `accuracy_pct`
- `peak_memory_gb`
- `runtime_hours`
- `throughput_samples_per_second`

不同隐私预算重复运行后，会形成“每个预算一行”的对比数据。所有指标只从本次日志和产物提取；没有实测到的值留空，不会自动填造。

### 历史结果展示

每次子进程完成后，统一入口会先读取项目目录下预先准备的历史结果目录中与当前方法对应的 TXT 文件（例如 `Kal-DP.txt`），并在终端原样展示其中的 Acc、内存、运行时间、吞吐率及攻击评估结果。默认目录为 `privacy_tools/historical_results`，也可以在配置中修改 `historical_results_dir`；历史结果不会写入或覆盖本次 `experiment_metrics.csv`。

历史结果只在运行成功或选择“查看最近历史结果”时展示。立即执行被中断、子进程返回非零状态或准备阶段发生异常时，终端输出“运行失败，请重新运行”，不展示历史结果，也不追加本次实验指标；运行目录中的日志和 manifest 会保留用于排查。

## 环境准备

工具集本身只需要 Python 3.10 或 3.11。各方法使用的解释器由 JSON 中的 `python.<method>` 配置；立即执行前会检查依赖、CUDA、GPU、模型、数据和 adapter。配置环境缺少依赖时会停止并提示，不再通过交互菜单切换或自动创建环境。

自动安装清单位于：

```text
privacy_tools/requirements/auto/
```

正式训练和实际攻击通常还需要：

- NVIDIA GPU 与匹配的 CUDA 版 PyTorch；
- Llama-2 的 Hugging Face 访问权限或本地模型目录；
- 可用网络与足够磁盘空间；
- 对应方法已经训练出的 adapter 或 steering checkpoint。

ClipText 默认参数随包提供预计算映射，不会默认下载大型 GloVe 文件；只有改用尚无映射的参数组合时，工具集才会询问如何准备 GloVe。

## 可选命令

列出全部方法与运行项：

```bash
python3 privacy_tool.py list
```

按“任务 → 扰动/评估类型 → 方法 → 实验”层级列出所有分支会使用的模型、数据集、adapter、checkpoint 等路径选项及其当前配置值：

```bash
python3 privacy_tool.py paths
```

检查某个方法：

```bash
python3 privacy_tool.py doctor --method kal_dp
```

跳过菜单，预演 Kal-DP：

```bash
python3 privacy_tool.py run --method kal_dp --experiment utility --dry-run
```

覆盖隐私预算并真实运行：

```bash
python3 privacy_tool.py --set methods.kal_dp.epsilon=4 run --method kal_dp --experiment utility
```

交付前完整审计：

```bash
python3 full_audit.py
```

六份源码顶层目录固定为 `ClipText/`、`PTCDS/`、`Kal-DP/`、`ZO-DPGA/`、`DADP-ICL/` 和 `PrivSV/`。配置中的本地文件使用项目相对路径，因此整个顶层目录移动到其他 Linux 服务器后仍可由统一入口解析。

ChatGLM / FM9G 间接评估配置、命令预览与离线重算见 [接入说明](privacy_tools/INDIRECT_EVALUATION.md)。
