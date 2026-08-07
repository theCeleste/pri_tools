import pandas as pd

df_member = pd.read_csv("/home/sch/dp-few-shot/privSV_attack/attack_results_other_bert_0_noise_member.csv")
df_nonmember = pd.read_csv("/home/sch/dp-few-shot/privSV_attack/attack_results_sst2_bert_0_noise_nomember.csv")

df_member["label"] = 1
df_nonmember["label"] = 0

df_all = pd.concat([df_member, df_nonmember], ignore_index=True)
df_all.to_csv("attack_results_bert_combined_noise.csv", index=False)
print(f"✅ 已合并成员+非成员文件，保存为 attack_results_bert_combined_noise.csv")
