from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

model_name = "bigcode/starcoder"

model_path = "/home/lc/code/data/models/starcode"

#用 AutoModel.from_pretrained() 下载模型

acc_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=acc_token)
tokenizer.save_pretrained(model_path,trust_remote_code=True,revision="main")
del(tokenizer)

#用 PreTrainedModel.save_pretrained() 保存模型到指定位置


model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=acc_token)
model.save_pretrained(model_path,trust_remote_code=True,revision="main")