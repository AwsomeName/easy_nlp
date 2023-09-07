from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import BloomTokenizerFast

# model_name = "bigcode/starcoder"
# model_name = "THUDM/chatglm-6b"
# model_name = "HuggingFaceH4/starchat-beta"
# model_name = "bigscience/bloom-3b"
# model_name = "bigscience/bloom-560m"
# model_name = "rahuldshetty/starchat-beta-8bit"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
model_name = "decapoda-research/llama-7b-hf"
model_name = "microsoft/speecht5_tts"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "OpenBuddy/openbuddy-llama2-70b-v10.1-bf16"

# model_path = "/root/data/bloom-560m"
# model_path = "/home/lc/models/HuggingFaceH4/starchat-beta"
# model_path = "/home/lc/models/rahuldshetty/starchat-beta-8bit"
model_path = "/home/lc/models/decapoda-research/llama-7b-hf"
model_path = "/home/lc/models/all-MiniLM-L6-v2"
model_paht = "/home/lc/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16"

#用 AutoModel.from_pretrained() 下载模型

acc_token = "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"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
# tokenizer.save_pretrained(model_path,revision="main")
# print("token done")
# del(tokenizer)

#用 PreTrainedModel.save_pretrained() 保存模型到指定位置


model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_auth_token=acc_token)
model.save_pretrained(model_path,revision="main")