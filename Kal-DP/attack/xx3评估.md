## 准备

cd到你的模型路径下
cp -r ../../models 到你的模型路径下(间接攻击的掩码补全模型)

## 1.直接攻击
- 有隐私保护
python extraction_temperature_decay.py

- 无隐私保护
python extraction_temperature_decay_wo.py

将得到的文件检测
python member_test.py
发现有隐私保护攻击成功率为0，无隐私保护攻击出了一条记录 => 降低100%

## 2.间接攻击
- 有隐私保护
python neighbour_attack.py --proc-id 0 --model roberta --dataset sst2  --split train
生成all_scores_sst2_roberta_0_train.pkl
python neighbour_attack.py  --proc-id 0 --model roberta --dataset sst2  --split test
生成all_scores_sst2_roberta_0_test.pkl

- 无隐私保护
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split train
生成wo_all_scores_sst2_roberta_0_train.pkl
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split test
生成wo_all_scores_sst2_roberta_0_test.pkl

- 得到对比结果（分析间接攻击生成的pkl文件）
python evaluate_mia_fair_compare.py   --base-train-pkl wo_all_scores_sst2_roberta_0_train.pkl   --base-test-pkl  wo_all_scores_sst2_roberta_0_test.pkl   --priv-train-pkl all_scores_sst2_roberta_0_train.pkl   --priv-test-pkl  all_scores_sst2_roberta_0_test.pkl   --target-fprs 0.025 0.01   --pos-neg-ratio 0.2


