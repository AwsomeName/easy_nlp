from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers import BloomTokenizerFast, AutoConfig
import torch
import os
# import subprocess
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map
from tensor_parallel import TensorParallelPreTrainedModel, infer_sharded_device_map


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("-----device:", device)
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    n_devices = torch.cuda.device_count()
    device_ids = list(range(n_devices))
else:
    n_devices = 1
    device_ids = None

print("---device_ids:", device_ids)
test_data = load_dataset("json", data_files="humaneval.jsonl")
# test_data = load_dataset("openai_humaneval")
# test_data = load_dataset("THUDM/humaneval-x")

# print(test_data)

# tok = AutoTokenizer.from_pretrained("/root/data/bloom", padding_side="left")
# tok = AutoTokenizer.from_pretrained("/root/data/bloom-560m", padding_side="left")
tok = AutoTokenizer.from_pretrained("/hy-tmp/starchat-beta", padding_side="left")
# tok = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b", padding_side="left")
# tok = BloomTokenizerFast.from_pretrained("/hy-tmp/b3b", padding_side="left")
# tok.save_pretrained("/hy-tmp/b3b")
# tokenizer.pad_token = tokenizer.eos_token
tok.pad_token = tok.eos_token
print("init tok done")

# model = AutoModelForCausalLM.from_pretrained("/root/data/bloom").cuda()
# model = AutoModelForCausalLM.from_pretrained("/root/data/bloom-560m").cuda()
# max_memory_mapping = {0: "8GB", 1: "32GB"}

with init_empty_weights():
    model = TensorParallelPreTrainedModel(
        AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("/hy-tmp/starchat-beta")).half(),
        device_ids=["cuda:0", "cuda:1"],
    )
    # model =  AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("/hy-tmp/starchat-beta")).half()
    
# model = AutoModelForCausalLM.from_pretrained(
#     "/hy-tmp/b3b", device_map="auto", torch_dtype=torch.half).to(device)
    # "/hy-tmp/b3b", device_map="auto", torch_dtype=torch.half, max_memory=max_memory_mapping).to(device)
    # "bigscience/bloom-3b", device_map="auto", torch_dtype=torch.half, max_memory=max_memory_mapping).to(device)
    # "/hy-tmp/starchat-beta/", device_map="auto", torch_dtype=torch.half, max_memory=max_memory_mapping).to(device)
    # "/hy-tmp/starchat-beta/", device_map="auto").cuda()
# model.save_pretrained("/hy-tmp/b3b")

# blocks = torch.nn.parallel.replicate(model, device_ids)
# for block in blocks:
#     block.eval()
# model_parallel = DistributedDataParallel(model.to(device), device_ids=None)
# model_parallel.eval()
device_map = infer_sharded_device_map(model)
# device_map = infer_auto_device_map(model)
new_map = {}
# default_device = device_map["wrapped_model._sanity_check_params.1"]
# default_device = "0"
# # default_device.index = 0
# print("default_device: ", default_device)
# new_map["transformer.wte.weight"] = "0"
# new_map["transformer.wpe.weight"] = "0"
# device_map["transformer.h.0.ln_1.weight"] = default_device
# device_map["transformer.h.0.ln_1.bias"] = default_device
# device_map["transformer.h.0.attn.c_attn.weight"] = default_device
# dd = GPTBigCodeBlock
for key in device_map:
    new_key = key
    # if "wrapped_model.module_shards." in key:
    #     new_key = key.replace("wrapped_model.module_shards.", "")[2:].replace("tp_wrapped_module.", "")
    # else:
    #     new_key = key
    new_map[new_key] = device_map[key]
print("device_map: ", new_map)
# print(device_map['transformer.wte.weight'])


print("init model done")
# model = AutoModelForCausalLM.from_pretrained(
#     # "/hy-tmp/starchat-beta/", device_map="auto", torch_dtype=torch.half).to(device)
#     "/hy-tmp/starchat-beta/", device_map=new_map, torch_dtype=torch.half).cuda()
for i in range(4):
    shard_path = "/hy-tmp/starchat-beta/pytorch_model-0000" + str(i+1) + "-of-00004.bin"
    load_checkpoint_in_model(
        model,
        checkpoint=shard_path,
        device_map=new_map,
    )
# model = model.eval()  
# model = TensorParallelPreTrainedModel(model)
# subprocess.call(["rm", "-rf", shard_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("init model done_2")
# exit()

system_token = "<|system|>"
user_token = "<|user|>"
assistant_token = "<|assistant|>"
end_token = "<|end|>"
max_length = 2048
stop_token_ids = [tok.eos_token_id, tok(end_token)["input_ids"][0], tok(user_token)["input_ids"][0]]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
stop = StopOnTokens() 
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
        stopping_criteria=StoppingCriteriaList([stop])
    )


def gen_code(model, tok, prompt):
    prompts = tok(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # input_ids = torch.tensor(prompts['input_ids']).to('cuda:0')
    # attention_mask =  torch.tensor(prompts['attention_mask']).to('cuda:0')
    print("------current_device:", torch.cuda.current_device())
    input_ids = torch.tensor(prompts['input_ids']).to(torch.cuda.current_device())
    attention_mask =  torch.tensor(prompts['attention_mask']).to(torch.cuda.current_device())
    
    
    output = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        generation_config=generation_config)
    
    return tok.decode(output[0], skip_special_tokens=True)

    
system_msg = "Below are a series of dialogues between various people and an AI assistant.\
The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.\
The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed.\
It also tries to avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.\
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.\
-----\n"
train_data = test_data['train']

with open("./test_problem_top6_sc_prompt.txt", 'w') as wp:
    for idx, data in enumerate(train_data):
        prompt = system_token + "\n" + system_msg + end_token + "\n" + user_token + "\n" + data['prompt'] + end_token + "\n" + assistant_token + "\n"
        # prompt = "continue write the code below\n\n \"\"\"" + data['prompt'] + "\"\"\""
        code = gen_code(model, tok, prompt)
        # code = gen_code_multi_gpu(blocks, tok, prompt)
        print("-----------")
        print("prompt:", prompt)
        # print("prompt:", data['prompt'])
        # print("[code]:", code)
        print("[code]:", code.replace(prompt, ""))
        print("ans: \n", data['canonical_solution'])
        print("example_test:", data['test'])
        wp.write("----------------\n")
        wp.write(code + "\n")
        if idx > 6:
            break
        # break


print(len(train_data))