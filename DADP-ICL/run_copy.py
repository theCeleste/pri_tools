from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("benayas/llama-2-7b-sst2_v0")
model = AutoModelForCausalLM.from_pretrained("benayas/llama-2-7b-sst2_v0")
