from transformers import BloomTokenizerFast, BloomForCausalLM, pipeline
import torch
import json

model_path = "/root/easy_nlp/trl/outputs/sft/checkpoint-2600"
# tokenizer = BloomTokenizerFast.from_pretrained(model_path)
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
tokenizer.pad_token = tokenizer.eos_token

# model = BloomForCausalLM.from_pretrained(model_path).cuda()
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").cuda()
model.eval()

def gen_res(model, tokenizer, ipstr):
    chosen_token = tokenizer(
        ipstr,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = torch.tensor(chosen_token['input_ids']).cuda()
    attention_mask = torch.tensor(chosen_token['attention_mask']).cuda()
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1024,
        do_sample=True,
        top_k=5,
        top_p=0.95,
        num_return_sequences=1,
        early_stopping=True,
        length_penalty=-0.5)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

with open("./data/dev.json", 'r') as fp, open("./generate.txt", 'a') as wp:
    for line in fp.readlines():
        line = line.strip()
        info = eval(line)
        input_text = info['instruct']
        res_text = gen_res(model, tokenizer, input_text)
        print("-------")
        print(info)
        print(res_text)
        info['res'] = res_text
        # info['version'] = "bloom-2200-sft"
        info['version'] = "raw"
        wp.write(json.dumps(info))
        