import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("joyfine/llama2-7b-fine-tuning_SST2_20")
model = AutoModelForCausalLM.from_pretrained("joyfine/llama2-7b-fine-tuning_SST2_20")
