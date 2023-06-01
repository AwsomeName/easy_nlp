from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/data/starchat", padding_side="left")

system_token = "<|system|>"
user_token = "<|user|>"
assistant_token = "<|assistant|>"
end_token = "<|end|>"

print(tokenizer.eos_token_id)
print(tokenizer(end_token)["input_ids"][0])