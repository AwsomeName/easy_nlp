from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# test_data = load_dataset("json", data_files="/root/data/data/openai_humaneval/HumanEval.jsonl")
test_data = load_dataset("json", data_files="/root/data/data/humaneval-x/data/python/data/humaneval.jsonl")

print(test_data)

tok = AutoTokenizer.from_pretrained("/root/data/bloom")
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained("/root/data/bloom").cuda()

def gen_code(model, tok, prompt):
    prompt = tok.tokenize(
        prompt,
        max_length=tok.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = torch.tensor(prompt['input_ids']).cuda()
    attention_mask =  torch.tensor(prompt['attention_mask']).cuda()
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=tok.max_len,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_beams=5,
        eos_token_id=tok.eos_token_id,
        num_return_sequences=1,
        early_stopping=True,
        len_penalty=2.5)
    
    return tok.decode(output[0], skip_special_tokens=True)

train_data = test_data['train']

with open("./test_result_blm.txt", 'w') as wp:
    for data in train_data:
        # print(data)
        # for d in data:
            # print("-----")
            # print(d)
            # print(data[d])
        prompt = data['prompt']
        code = gen_code(model, tok, prompt)
        print("-----------")
        print("prompt:", prompt)
        print("code:", code)
        print("ans: ", data['canonical_solution'])
        print("example_test:", data['example_text'])
        wp.write(code + "\n")
        break


print(len(train_data))