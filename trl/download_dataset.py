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
