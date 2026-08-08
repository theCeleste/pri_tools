import pandas as pd

# df_member = pd.read_csv("./attack_results_other_bert_0.csv")
# df_nonmember = pd.read_csv("./attack_results_sst2_bert_0.csv")
df_member = pd.read_csv(r"/home/zjs/code/x73/x73-privacy-tool/privacy_tools/results/example/DADP-ICL/indirect_attack/20260807_234850_b228ae/attack_results_other_bert_0.csv")
df_nonmember = pd.read_csv(r"/home/zjs/code/x73/x73-privacy-tool/privacy_tools/results/example/DADP-ICL/indirect_attack/20260807_234850_b228ae/attack_results_sst2_bert_0.csv")

df_member["label"] = 1
df_nonmember["label"] = 0

df_all = pd.concat([df_member, df_nonmember], ignore_index=True)
df_all.to_csv(r"/home/zjs/code/x73/x73-privacy-tool/privacy_tools/results/example/DADP-ICL/indirect_attack/20260807_234850_b228ae/attack_results_bert_combined.csv", index=False)
# df_all.to_csv("attack_results_bert_combined.csv", index=False)
print("✅ 当前评估：对比方法；已合并成员+非成员文件，保存为：/home/zjs/code/x73/x73-privacy-tool/privacy_tools/results/example/DADP-ICL/indirect_attack/20260807_234850_b228ae/attack_results_bert_combined.csv")
