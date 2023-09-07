# -*- coding: utf-8 -*-
 
import sys
import copy
import os
import gc
import fire
import torch
import time
import re
 
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
 
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
 
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.
You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.
You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI."""
 
 
def load_model(
        load_8bit: bool = False,
        base_model: str = '/data/Workspace/TEMP/DATA/ckpt/OpenBuddy-openbuddy-llama2-70b-v10.1-bf16'
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # 初始化模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
         max_memory = {0: '10Gib',1: '10Gib',2: '10Gib',3: '11Gib', 'cpu': '128Gib'},
        use_safetensors=True
    )
    # unwind broken decapoda-research config
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
 
    # if not load_8bit:
 
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
 
    return tokenizer, model
 
 
 
cutoff_len = 2048
IGNORE_INDEX = -100
 
if __name__ == "__main__":
    tokenizer, model = load_model()
 
    from torch.utils.data import Dataset, DataLoader
 
    from tqdm import tqdm
 
    generation_config = GenerationConfig(
        temperature=0.3,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        repetition_penalty=1.3,
    )
 
    def _clean_prompt(sentence):
        return sentence.strip()
 
    def get_prompt(tokenizer, sentence, history):
        sentence = _clean_prompt(sentence)
        if history == '':
            # print("history:", history)
            return tokenizer.bos_token + DEFAULT_SYSTEM_PROMPT + f"\n\nUser: {sentence}.\nAssistant:"
        else:
            if history[-5:] == '\n</s>':
                history = history[:-5] + '</s>'
            elif history[-4:] != '</s>':
                history = history + '</s>'
            # print("history:", history)
            return history + f"\n\nUser: {sentence}.\nAssistant:" 
 
            
 
    history = ''
    output = ''
    clean_flag = "clean"
    while 1:
        user_input = input("User:")
        if 'clean' == user_input:
            history = ''
            output = ''
            print('clean history')
            continue
        time0 = time.time()
        prompt = get_prompt(tokenizer, user_input, history)
        # print("real prompt:", prompt)
        sentence_ids = tokenizer(prompt, add_special_tokens=False)
        sentence_ids['input_ids'] = sentence_ids['input_ids'][-cutoff_len:]
        sentence_ids['attention_mask'] = sentence_ids['attention_mask'][-cutoff_len:]
        input_ids = torch.as_tensor(sentence_ids["input_ids"],dtype=torch.long).reshape(1, -1).to(device)
        attention_mask = torch.as_tensor(sentence_ids["attention_mask"],dtype=torch.long).reshape(1, -1).to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                use_cache=False
            )
            s = generation_output
            for i in s[0]:
                output = tokenizer.decode(list(i.detach().cpu().numpy()))
            res = output.split("Assistant:")[-1].replace(tokenizer.eos_token, '').strip()
            print("Assistant:", res)
        time_use = time.time() - time0
        print("time use:", time_use)
        history = output