# 攻击流程与原始说明对应表

统一入口遵循以下规则：原始代码的 `.md` 明确写出攻击流程时，以该流程为准；硬编码模型、数据、GPU 和输出路径只在运行时临时副本中替换，不改原文件。

| 方法 | 直接评估实际流程 | 间接评估实际流程 | 原始说明依据 |
|---|---|---|---|
| ClipText | `extraction_temperature_decay.py`（受保护）→ `member_test.py`；`extraction_temperature_decay_wo.py`（未保护）→ `member_test.py` | `neighbour_attack.py` 的 train/test；`neighbour_attack_wo.py` 的 train/test；`evaluate_mia_fair_compare.py --target-fprs 0.025 0.01 --pos-neg-ratio 0.2` | `ClipText/README.md` 的“攻击”章节 |
| PTCDS | 与 ClipText 相同的两路流程：受保护权重执行 `attack/extraction_temperature_decay.py` → `member_test.py`；未保护 base 权重执行 `attack/extraction_temperature_decay_base.py` → `member_test.py` | `attack/neighbour_attack.py` 与 `neighbour_attack_wo.py` 的 train/test，再公平比较 | `PTCDS/README.md` 的直接评估、间接评估章节 |
| Kal-DP | `extraction_temperature_decay.py`（受保护）→ `member_test.py`；`extraction_temperature_decay_wo.py`（未保护）→ `member_test.py` | `neighbour_attack.py` 的 train/test；`neighbour_attack_wo.py` 的 train/test；`evaluate_mia_fair_compare.py --target-fprs 0.025 0.01 --pos-neg-ratio 0.2` | `Kal-DP/readme.md` 的“攻击方法”章节 |
| ZO-DPGA | 与 Kal-DP 相同的两路前缀抽取流程，加载 ZO-DPGA 本方法/对比模型 | 与 Kal-DP 相同的四路邻居得分及公平比较，加载 ZO-DPGA 本方法/对比模型 | `ZO-DPGA/README - new.md`：“和 kalman 方法一致，只需要换下路径” |
| DADP-ICL | 原始 `invade_prefix_with_custom.py` 运行两遍：分别用 `synthetic_text`、`original_text` 组成 prompt；两组均由原始 `member_test.py` 统计命中 | 原始 `neighbour_mia.py` 运行四遍：`synthetic_text/original_text × other/sst2`；`combine.py` 分别合并；`evaluate_attack_metrics.py --ratio 0.1` 分别评估 | 示例扰动原始 `readme.md` 的直接攻击、间接攻击章节 |
| PrivSV | 原始 `extract_topn.py`（未保护）和 `extract_topn-sv.py`（PrivSV）各运行一次；分别执行 `member_test.py`、`member_test-sv.py` | 原始 `neighbour_mia_sv.py` 分别以 steering 倍数和 0 倍运行 `other/sst2`；`combine.py` 分别合并；`evaluate_attack_metrics.py --ratio 0.1` 分别评估 | 示例扰动原始 `readme.md` 的“引导向量适配”章节 |

其中 `other` 是成员样本，`sst2` 是非成员样本；DADP-ICL 的 `synthetic_text` 是保护后提示，`original_text` 是未保护对比提示；PrivSV 的 steering 倍数 0 表示同一原始打分逻辑下关闭保护干预。
