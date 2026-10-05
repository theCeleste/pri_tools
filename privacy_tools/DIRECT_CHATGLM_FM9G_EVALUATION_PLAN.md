# ChatGLM 与九格大模型直接评估接入规划

## 目标

基于 `evaluate/direct/` 现有的无扰动显式微调和隐式微调实现，补齐统一入口中 ChatGLM3-6B 与 FM9G-7B（九格大模型）的直接评估接口。本阶段只定义实现方案，不改变现有评估代码。

首期支持以下组合：

| 模型 | 显式微调 | 隐式微调 | 数据集 |
| --- | --- | --- | --- |
| ChatGLM3-6B | LoRA 生成与成员匹配 | ICL 生成与成员匹配 | SST-2、WikiText-2 |
| FM9G-7B | LoRA 生成与成员匹配 | ICL 生成与成员匹配 | SST-2、WikiText-2 |

九格脚本目前还包含 AG News 生成逻辑，但首期不纳入统一直接评估，待隐式流程、数据格式和匹配口径完整后再扩展。

## 现有实现与复用关系

- ChatGLM 显式生成：`evaluate/direct/lora/chatglm/temperature_chatglm.py`
- 九格显式生成：`evaluate/direct/lora/9g/temperature_9g.py`
- 显式成员匹配：`evaluate/direct/lora/sst2_member_test.py`
- 隐式生成：`evaluate/direct/DPICL/run_exp_sst2_local.py`
- 隐式成员匹配：`evaluate/direct/DPICL/member_test_current.py`
- 统一入口占位：`privacy_tools/unified.py` 中的 ChatGLM 显式、ChatGLM 隐式和 FM9G 直接评估分支

显式流程继续使用候选文本生成、PPL/Zlib 排序和训练集匹配。隐式流程继续使用无扰动示例生成，再对 `synthetic_text` 或 `generated_text` 做成员匹配。

## 当前阻塞点

1. 统一入口中的四个直接评估分支仍是占位提示。
2. 两个显式生成脚本的模型、适配器和输出路径主要通过命令行或默认值指定，不能直接作为统一后端安全调用。
3. `sst2_member_test.py` 含硬编码路径，并在模块顶层执行，需要改造成可调用的参数化命令或函数。
4. 隐式生成依赖 `local_lm_compat.py` 和 `prob_utils.py`，需要确认本地依赖来源、Python 环境和模型接口兼容性。
5. 隐式脚本的默认 `sigma=0.39` 与无扰动评估目标冲突，必须显式建立无噪声执行分支，并验证所有分支不会使用未初始化结果。
6. LoRA 路径缺失时，显式生成脚本可能退回基座模型；统一接口必须改为预检失败，不能把基座结果标记为微调结果。
7. 生成文本经过长度和解码过滤后，实际有效数量可能小于目标数量，统计中必须区分目标数、有效候选数和最终匹配数。

## 计划的模块设计

### 1. 新增统一直接评估模块

新增 `privacy_tools/direct_evaluation.py`，参考 `privacy_tools/indirect_evaluation.py` 的组织方式，提供：

- `run`：执行生成和匹配
- `preview`：只展示完整命令、配置和资源
- `history`：读取当前模型、微调方式和数据集的历史结果
- 离线重算：对已有候选或原始结果重新执行匹配

统一入口只负责选择任务和启动子进程，避免菜单进程直接导入 Torch、Transformers 或数据集库。

### 2. 配置和资源预检

在设置文件中新增 `evaluation.direct` 配置段，至少包含：

- ChatGLM、FM9G 的基座模型目录
- 按模型和数据集区分的 LoRA 适配器目录
- tokenizer 目录
- SST-2 与 WikiText-2 数据路径
- Python 解释器、GPU、随机种子
- 生成参数：候选数量、batch size、最大新 token、top-p、top-k、解码模式
- 匹配参数：最小长度、contains 阈值、fuzzy 阈值、WikiText 窗口长度

执行前检查模型配置、权重、tokenizer、适配器配置和数据文件；路径或资源缺失时立即失败。统一内部名称为 `chatglm3`、`fm9g` 和 `wikitext2`，兼容现有菜单中的别名。

### 3. 参数化显式 LoRA 后端

- 保留 ChatGLM 与 FM9G 各自的加载和生成兼容逻辑。
- 将路径、设备、dtype、候选数、输出目录等全部改为显式参数。
- 强制要求适配器存在且可加载，不允许静默退回基座模型。
- 输出结构化 JSON，至少记录 prompt、生成文本、PPL、Zlib 长度、ratio、模型、数据集和随机种子。
- 可继续输出 TXT 作为人工检查格式，但 JSON 作为后续匹配的唯一接口。
- 记录有效候选不足、采样异常和 greedy fallback 等状态。

排序规则仍为 `log(PPL) / log(zlib 长度)`，默认生成 1000 条并选取前 100 条；实际数量不足时在结果中明确记录。

### 4. 参数化隐式无扰动后端

- 整理 `local_lm_compat.py`、`prob_utils.py` 的本地依赖和解释器配置。
- 增加明确的无扰动模式，确保不注入 Gaussian 或 Exponential 噪声。
- 固定并记录示例数、查询数、最大生成 token、采样种子和模型路径。
- SST-2 统一读取 `synthetic_text`；WikiText 统一读取 `generated_text`。
- 保存原始生成 JSON，供匹配和离线复算使用。

无扰动模式必须通过代码路径和运行日志双重确认，不能仅依赖默认参数值。

### 5. 统一成员匹配输出

复用现有匹配规则，并参数化数据加载：

- raw exact
- normalized exact
- contains（满足最小长度）
- fuzzy；WikiText 额外支持窗口 fuzzy

每条候选保留候选编号、原文、匹配类型、分数、训练集编号、匹配原文和标签（SST-2）。汇总结果包含候选总数、匹配数、匹配率、各匹配类型计数、阈值和数据集信息。空候选保留为未匹配记录，不能改变统计分母。

### 6. 统一入口接线

将 `privacy_tools/unified.py` 中的三个占位路径改为调用新模块：

- ChatGLM 显式 → `direct_evaluation.run(..., model="chatglm3", finetune="explicit")`
- ChatGLM 隐式 → `direct_evaluation.run(..., model="chatglm3", finetune="implicit")`
- FM9G 显式或隐式 → 对应 `model="fm9g"` 分支

统一入口保留立即执行和查看历史结果两种模式；底层模块仍可通过预览模式输出命令，并沿用统一结果目录和运行清单格式：

`results/direct_evaluation/<model>/<finetune>/<dataset>/<run_id>/`

每次运行至少保存配置清单、命令、环境、日志、候选结果、匹配结果和 `summary.json`。

## 验证计划

### 静态和单元验证

- 命令构造、别名映射和配置覆盖
- 缺少基座、tokenizer、LoRA、数据集时的失败信息
- 显式候选 JSON 与隐式生成 JSON 的解析
- 空候选、短文本和重复文本的统计分母
- 匹配阈值、匹配类型和结果字段
- preview 模式不加载模型、不创建正式结果

### 小规模运行验证

对四种模型/微调组合分别使用少量候选验证：

- ChatGLM 显式、FM9G 显式
- ChatGLM 隐式、FM9G 隐式

验证模型确实加载了对应适配器，隐式流程未注入噪声，输出可被匹配阶段完整消费，失败状态能写入运行清单。

### 正式运行验证

在资源满足后执行 SST-2 和 WikiText-2 正式规模运行，检查：

- 候选数量与过滤数量
- 匹配率及匹配类型分布
- 运行可复现信息
- 历史结果筛选是否按模型、微调方式、数据集和后端版本隔离

## 结果口径

直接评估主指标为训练集文本抽取或匹配率，并分别报告精确命中和模糊命中。PPL、Zlib、ratio 作为候选排序依据，不直接等同于泄露率。隐式流程需要同时记录生成样本数量和匹配样本数量，避免把示例生成成功率与成员命中率混为一谈。
