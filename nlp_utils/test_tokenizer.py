from datasets import load_dataset
from transformers import AutoTokenizer

data_path = "/Users/lc/code/easy_nlp/data/"
data_files = {'train': 'train_100.txt', 'test': 'test_10.txt'}
ds = load_dataset(data_path, data_files=data_files)

print(ds)
print(ds['test'])
print(ds['test'][0])

print("loading tokenizer...")
model_path = "/Users/lc/code/easy_nlp/models/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
print("load tokenizer done...")

print(ds['train']['text'][:10])
ds_list = ds['train']['text'][:10]
# ds_list = [x for x in ds['train'][:10]]
print(ds_list)
print(len(ds_list))

inputs = tokenizer(ds_list, return_tensors="pt", padding=True, truncation=True)

print("----------------")
print(inputs)
print("----------------")
print(inputs['input_ids'])
print(inputs['attention_mask'])
print(type(inputs))
print(type(inputs['input_ids']))
print(inputs['input_ids'].shape, inputs['attention_mask'].size())
