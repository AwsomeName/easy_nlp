# 这一版是可以运行的，除了一个问题，就是accelerate包的load_checkpoint_and_dispatch， 会报错，load的时候，可以改为如下这个样子：
# return torch.load(checkpoint_file, map_location={'cuda:2': 'cuda:0','cuda:3': 'cuda:1'})

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers import BloomTokenizerFast, AutoConfig
import torch
import os
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map, load_checkpoint_and_dispatch
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
tok = AutoTokenizer.from_pretrained("/hy-tmp/starchat-beta", padding_side="left")
tok.pad_token = tok.eos_token
print("init tok done")


with init_empty_weights():
    model =  AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("/hy-tmp/starchat-beta")).half()
    
check_point = "/hy-tmp/starchat-beta/"
model = load_checkpoint_and_dispatch(
    model, check_point, device_map="auto", no_split_module_classes=["GPTBigCodeBlock"]
)

print("init model done_")

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