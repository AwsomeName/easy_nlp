from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# model_name = "bigcode/starcoder"
model_name = "THUDM/chatglm-6b"

model_path = "/root/data/glm"

#用 AutoModel.from_pretrained() 下载模型

acc_token = "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"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
# tokenizer.save_pretrained(model_path,revision="main")
# print("token done")
# del(tokenizer)

#用 PreTrainedModel.save_pretrained() 保存模型到指定位置


model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
model.save_pretrained(model_path,revision="main")