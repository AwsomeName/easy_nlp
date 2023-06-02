from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# test_data = load_dataset("json", data_files="/root/data/data/openai_humaneval/HumanEval.jsonl")
test_data = load_dataset("json", data_files="/root/data/data/humaneval-x/data/python/data/humaneval.jsonl")

print(test_data)

# tok = AutoTokenizer.from_pretrained("/root/data/bloom")
tok = AutoTokenizer.from_pretrained("/root/data/bloom-560m")
tok.pad_token = tok.eos_token
print("init tok done")

# model = AutoModelForCausalLM.from_pretrained("/root/data/bloom").cuda()
model = AutoModelForCausalLM.from_pretrained("/root/data/bloom-560m").cuda()
print("init model done")

max_length = 2048

generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
        # eos_token_id=tok.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
    )


def gen_code(model, tok, prompt):
    prompts = tok(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = torch.tensor(prompts['input_ids']).cuda()
    attention_mask =  torch.tensor(prompts['attention_mask']).cuda()
    
    output = model.generate(input_ids=input_ids, generation_config=generation_config)
    # output = model.generate(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     max_length=max_length,
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.95,
    #     num_beams=5,
    #     eos_token_id=tok.eos_token_id,
    #     num_return_sequences=1,
    #     early_stopping=True,
    #     length_penalty=2.5)
    
    return tok.decode(output[0], skip_special_tokens=True)

train_data = test_data['train']

with open("./test_result_blm.txt", 'w') as wp:
    for data in train_data:
        # print(data)
        # for d in data:
            # print("-----")
            # print(d)
            # print(data[d])
        prompt = "continue write the code below\n\n \"\"\"" + data['prompt'] + "\"\"\""
        code = gen_code(model, tok, prompt)
        print("-----------")
        print("prompt:", prompt)
        print("[code]:", code)
        # print("[code]:", code.replace(prompt, ""))
        print("ans: \n", data['canonical_solution'])
        print("example_test:", data['example_test'])
        wp.write(code + "\n")
        # break


print(len(train_data))