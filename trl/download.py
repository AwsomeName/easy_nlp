import datasets
import json

dataset = datasets.load_dataset("Anthropic/hh-rlhf")
# dataset.save_to_disk('./data/Anthropic/hh-rlhf')

print(dataset)

print(dataset['train'][0])
print("---------")
print(dataset['train'][1]['chosen'])

# 构造instruct和PPO用的数据

train_data = dataset['train']
test_data = dataset['test']

with open("./data/train.json", 'w') as wp:
    for data in dataset['train']:
        chosen = data['chosen']
        instuct = chosen.split("Assistant")[0] + "Assistant: "
        sec = chosen.replace(instuct, "")
        ans = sec.split("Human")[0]
        value = {}
        value["instruct"] = instuct
        value["ans"] = ans
        wp.write(json.dumps(value) + "\n")

        
with open("./data/dev.json", 'w') as wp:
    for data in dataset['test']:
        chosen = data['chosen']
        instuct = chosen.split("Assistant")[0] + "Assistant: "
        sec = chosen.replace(instuct, "")
        ans = sec.split("Human")[0]
        value = {}
        value["instruct"] = instuct
        value["ans"] = ans
        wp.write(json.dumps(value) + "\n")

# 下载模型
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

model_name = "bigscience/bloom-560m"

model_path = "./model/bloom-560"

#用 AutoModel.from_pretrained() 下载模型

acc_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=acc_token)
tokenizer.save_pretrained(model_path,trust_remote_code=True,revision="main")
del(tokenizer)

#用 PreTrainedModel.save_pretrained() 保存模型到指定位置


model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=acc_token)
model.save_pretrained(model_path,trust_remote_code=True,revision="main")