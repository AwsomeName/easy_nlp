from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import BloomTokenizerFast

# model_name = "bigcode/starcoder"
# model_name = "THUDM/chatglm-6b"
model_name = "HuggingFaceH4/starchat-alpha"
model_name = "bigscience/bloom-3b"

model_path = "/root/data/bloom"

#用 AutoModel.from_pretrained() 下载模型

acc_token = "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"

tokenizer = BloomTokenizerFast.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
tokenizer.save_pretrained(model_path,revision="main")
print("token done")
del(tokenizer)

#用 PreTrainedModel.save_pretrained() 保存模型到指定位置


model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
model.save_pretrained(model_path,revision="main")