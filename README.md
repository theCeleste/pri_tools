# 六种隐私保护方法统一实验工具

本工具把六套研究代码接入同一个命令行助手。使用者只需按菜单选择，不需要记忆各方法原来的脚本、参数或目录。

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

菜单顺序固定为：

```text
1 示例扰动
  1 DADP-ICL
  2 PrivSV
2 嵌入扰动
  1 ClipText
  2 PTCDS
3 梯度扰动
  1 Kal-DP
  2 ZO-DPGA
```

选择任一方法后，运行项均完全相同：

```text
1 本微调方法
2 对比微调方法
3 直接评估
4 间接评估
```

到参数步骤时，可选择“一键使用全部默认参数”，也可逐项调整。真实执行前输入小写 `run`；输错后工具会提示重新输入，不会直接退出，输入 `0` 可取消。

## 六种方法的对比项

| 方法 | 本微调方法 | 对比微调方法 |
|---|---|---|
| ClipText | 脱敏文本 LoRA | 原始文本 LoRA |
| PTCDS | PTCDS 表示保护微调 | 原始文本 LoRA |
| Kal-DP | Kalman 梯度过滤差分隐私微调 | DP-SGD |
| ZO-DPGA | ZO-DPGA | LoRA + MeZO |
| DADP-ICL | 私有示例生成与 ICL 评估 | 非私有 ICL |
| PrivSV | 私有 steering vector | 无噪声 steering vector |

“本微调方法”和“对比微调方法”都会记录本次真实运行得到的准确率、峰值内存、运行时间和吞吐率。Kal-DP 的对比项会完整记录 DP-SGD 的四项指标。

“直接评估”执行训练数据抽取评估；“间接评估”执行邻居成员推断评估。凡原始代码的 `.md` 已明确攻击步骤，统一入口均按其中规定的受保护/未保护、成员/非成员组合和评估参数执行。原始脚本中的硬编码路径仅在单次运行目录的临时副本中替换，六份源码不被改写。逐项对应关系见 [攻击流程对应表](privacy_tools/ATTACK_README_ALIGNMENT.md)。

## 第一次测试

先执行不加载大模型的交付自检：

```bash
python3 smoke_test.py
```

自检会检查六种方法共 24 个运行项的脚本、路径、参数和预演命令，不会下载模型或启动 GPU 训练。

然后启动助手：

```bash
python3 privacy_tool.py
```

第一次建议在“执行模式”中选择“安全预演”。安全预演会生成完整命令，但不会加载模型、占用 GPU 或写入正式实验指标。

示例路径：

```text
3 梯度扰动
→ 1 Kal-DP
→ 1 本微调方法
→ 1 安全预演
→ 1 一键使用全部默认参数
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

每次子进程完成后，统一入口会先读取项目目录下预先准备的历史结果目录，再扫描本次运行产物。默认目录为 `privacy_tools/historical_results`，也可以在配置中修改 `historical_results_dir`。目录中的 JSON、JSONL、CSV、LOG 和 TXT 文件会被只读解析，并在终端展示已识别的 Acc、内存、运行时间、吞吐率、攻击成功率和 AUC；历史结果不会写入或覆盖本次 `experiment_metrics.csv`。

## 环境准备

统一助手本身只需要 Python 3.10 或 3.11。选择真实执行后，助手会检查依赖、CUDA、GPU、模型、数据和 adapter，并可自动创建隔离环境、安装所选方法的依赖。

自动安装清单位于：

```text
privacy_tools/requirements/auto/
```

正式训练和实际攻击通常还需要：

- NVIDIA GPU 与匹配的 CUDA 版 PyTorch；
- Llama-2 的 Hugging Face 访问权限或本地模型目录；
- 可用网络与足够磁盘空间；
- 对应方法已经训练出的 adapter 或 steering checkpoint。

ClipText 默认参数随包提供预计算映射，不会默认下载大型 GloVe 文件；只有改用尚无映射的参数组合时，助手才会询问如何准备 GloVe。

## 可选命令

列出全部方法与运行项：

```bash
python3 privacy_tool.py list
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
