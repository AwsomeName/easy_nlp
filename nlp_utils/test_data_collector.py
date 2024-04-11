from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding
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

# print(ds['train']['text'][:10])
# ds_list = ds['train']['text'][:10]
# # ds_list = [x for x in ds['train'][:10]]
# print(ds_list)
# print(len(ds_list))

# inputs = tokenizer(ds_list, return_tensors="pt")
text_data = ["hello llama", "who are you?"]

# 将文本数据编码为输入张量
# inputs = tokenizer(text_data, return_tensors="pt")
inputs = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)
print(inputs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(data_collator)

# mid_res = data_collator(ds['train'][:10])
batch = data_collator(inputs)
# mid_res = data_collator(inputs)

exit()
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
