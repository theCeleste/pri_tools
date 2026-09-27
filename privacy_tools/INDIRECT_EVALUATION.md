# ChatGLM / FM9G 新版间接评估

两种模型共用 evaluate/ 后端：显式使用 indirect_LoRA，隐式使用 indirect_ICL。
统一入口不再调用 glm/GLM_LoRA 或 glm/GLM_ICL；直接评估继续保留占位状态。

## 菜单

运行 python -B privacy_tool.py，选择：评估 → 间接评估 → 显式/隐式 → 无扰动微调 → ChatGLM/FM9G。
执行模式分为“立即执行”“查看最近历史结果”“预览评估命令”。历史查询和预览不启动计算、不创建结果目录。
离线重算使用下面的模块命令。

## 配置

settings.example.json 的 evaluation.indirect 保存新配置；可用 --config 指定覆盖文件。

- models.chatglm3.base_model / models.fm9g.base_model：本地完整底座目录。
  CHATGLM_MODEL_DIR / FM9G_MODEL_DIR 环境变量优先；未配置时分别检查 evaluate/models/chatglm3-6b、evaluate/models/9G7B_MHA。
  不将 Hugging Face 仓库标识当成本地路径，不自动下载模型。
- models.<model>.adapters.<dataset>：显式评估 adapter；ChatGLM SST-2 默认 /home/zjs/llm/glm-lora。
  其他组合默认检查 evaluate/indirect_LoRA/weights/<model>/<dataset>。
- models.<model>.tokenizers.<dataset>：可选 tokenizer 覆盖，默认与 adapter 同目录。
- models.<model>.training_manifests.<dataset>：可选训练来源证明，默认 adapter/training_manifest.json。
- lora.neighbor_model：/home/zjs/llm/roberta-base。
- icl.proposal_model：evaluate/indirect_ICL/resources/bert-base-uncased；不兼容将原 RoBERTa 配置直接传入 ICL。
- icl.input_json：可选，复用新版生成的10条示例；为空时运行原生示例生成。
- common.dataset：sst2（默认）或 wikitext2。
- common.dataset_path：SST-2 默认 privacy_tools/datasets/sst2；不改写目录内文件。
- datasets.wikitext2.lora_dir / icl_dir：分别为 LoRA HF 划分与 ICL JSON 快照。
- python.indirect_lora / python.indirect_icl：分别指向符合两套原生 requirements.txt 的 Python 环境。
  示例配置的 python3 仅是可编辑入口，不表示该环境已经具备依赖。
- common.gpu：单个物理卡号0/1/2/3。子进程统一使用映射后的逻辑 cuda:0。

旧 methods.chatglm_lora 的 adapter/邻居路径仍可作为缺省回退，但新的 evaluation.indirect 配置优先。
ICL 的实验规模和 seed 固定遵守交付方法，不继承旧 chatglm_icl 的任意样本规模。
底层 Python 环境、所有权重文件及正式样本数预检失败时，不自动降级或减少样本。

## 命令示例

以下命令在项目根目录执行，配置中的模型路径应先指向已安装的真实资源：

    python -B -m privacy_tools.indirect_evaluation --model chatglm --finetune explicit --mode preview
    python -B -m privacy_tools.indirect_evaluation --model fm9g --finetune implicit --mode preview
    python -B -m privacy_tools.indirect_evaluation --model chatglm --finetune implicit --mode run --config /path/to/local.json
    python -B -m privacy_tools.indirect_evaluation --model fm9g --finetune explicit --mode history
    python -B -m privacy_tools.indirect_evaluation --model fm9g --finetune implicit --dataset wikitext2 --mode preview

离线重算只接受模型、微调方式、数据集、后端指纹匹配的新版成功运行外层目录：

    python -B -m privacy_tools.indirect_evaluation --model chatglm --finetune explicit --mode evaluate --source-run /path/to/successful-run

它读取 source-run/backend 下的原始 score pickle 或攻击 CSV。仅有历史指标而无原始分数不能重算。
无需底座、adapter 或 GPU，但对应解释器仍需具备原生离线评估依赖。
旧 glm/ 输出不被自动当作新版结果导入。

## LoRA 成员来源证明

模型训练成员身份不能从 train.tsv 文件名推断。正式运行保留顺序取成员与非成员各前1200条，并验证：

1. 选取的样本非空、数量足够且组内无归一化重复。
2. 所有成员属于完整训练集；非成员与完整训练集及候选成员无交集。
3. 训练来源证明绑定实际 adapter 权重 SHA256、模型与数据集。

如 adapter 与新版交付清单中对应权重的 SHA256 一致，可使用经过清单校验的包内完整训练快照作为证明。
这只用于验证，不把攻击数据静默换成包内数据。
其他 adapter 需提供来自真实训练记录的 training_manifest.json：

    {
      "model": "chatglm3",
      "dataset": "sst2",
      "adapter_sha256": "实际 adapter_model.safetensors 文件的 SHA256",
      "source": "训练数据清单或训练任务的可追溯说明",
      "training_text_sha256": ["完整训练集中每条文本的规范化 SHA256"]
    }

文本规范化：去首尾空白、连续空白合并为单个空格、casefold，然后以 UTF-8 编码计算 SHA256。
列表必须来源于该 adapter 的完整真实训练集，不能从拟评估成员反向制作证明。
成员错配时应配置经过核对的划分；不得通过伪造证明或绕过检查继续。

## 输出与方法

外层目录为 results_dir/indirect_evaluation/<model>/<explicit|implicit>/<dataset>/<run_id>/。

- run_manifest.json：实际命令、白名单环境变量、资源、阶段日志位置、退出码与耗时、成功/失败状态。
- data_validation.json：运行前的数据指纹及成员来源检查记录。
- backend/：完整原生产物。LoRA 保留 PKL、metadata、CSV；ICL 保留两组提示的 CSV、metrics.json、ROC 和生成记录。
- summary.json：仅在所有步骤成功且结果完整时生成；读取实际 AUC、TPR 和样本数，不填入推测结果。

正式样本规模：LoRA 1200+1200、seed=42；ICL 10+500、seed=0。ICL 分开显示 original_text 与 synthetic_text，默认不加噪声，不声称差分隐私保证。
ICL 的 num-test=20 是 SST-2 分类效用测试数，不是攻击样本数。WikiText-2 不运行情感分类效用测试。
历史查询同时匹配模型、微调方式、数据集和后端源码指纹，忽略失败、旧版本及 legacy 结果。

## 验证与当前资源限制

    python -B -m unittest discover -s privacy_tools/tests -p test_indirect_evaluation.py -v
    python -B -m unittest discover -s privacy_tools/tests -p test_indirect_cpu.py -v

原有交付测试应在各自配置环境中运行。包内文件有改动时同步维护 checksums.json，不移除缺失资源的清单记录。

2026-09-27 的本地只读核验发现：当前统一 SST-2 的前1200条候选成员并非全部参与 /home/zjs/llm/glm-lora 的训练。
因此默认显式评估在正式攻击前会报告划分错配；本次实现没有修改统一数据或重新训练模型。
ICL 的统一 SST-2 与随包 WikiText-2 非成员检查通过。
当前底座模型和部分随包大权重尚未齐全；现有 dp-few-shot 环境的 Transformers 等版本与锁定依赖不一致，九格 tokenizer 检查还缺 datamodel_code_generator、jsonschema。
这些是实际 GPU 验收的前置条件，命令/菜单测试和合成分数测试通过不等于真实 GPU 实验通过。


本次新增接入及 CPU 离线测试共16项通过；CPU 离线测试需使用已具备原生评估依赖的环境，缺依赖时会显式跳过。
在本机可用 /home/zjs/anaconda3/envs/dp-few-shot/bin/python 运行离线测试；该环境尚不满足正式 GPU 运行的锁定依赖，不能将这两个验收层级混同。
