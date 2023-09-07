from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
debug = True
model_path = "/root/data/model"
model_path = "/home/lc/models/codellama/CodeLlama-7b-Instruct-hf"
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
system_msg = "Below are a series of dialogues between various people and an AI assistant.\
The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.\
The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed.\
It also tries to avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.\
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.\
-----\n"
system_token = "<|system|>"
user_token = "<|user|>"
assistant_token = "<|assistant|>"
end_token = "<|end|>"
max_length = 2048
stop_token_ids = [tokenizer.eos_token_id, tokenizer(end_token)["input_ids"][0], tokenizer(user_token)["input_ids"][0]]
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
stop = StopOnTokens()

stop_token_ids.append(tokenizer("\n")["input_ids"][0])
stop_by_line = StopOnTokens()


generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # eos_token_id=tok.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
        stopping_criteria=StoppingCriteriaList([stop])
    )

def gen_result(prompt, generation_config):
    global model, tokenizer
    prompts = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = torch.tensor(prompts['input_ids']).cuda()
    attention_mask =  torch.tensor(prompts['attention_mask']).cuda()
    
    output = model.generate(
        input_ids=input_ids, 
        generation_config=generation_config,
        attention_mask = attention_mask,)
    code = tokenizer.decode(output[0], skip_special_tokens=True)
    return code


@app.post("/api/v1/test")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

# use less
@app.post("/api/v1/multilingual_code_generate_block")
async def multilingual_code_generate_block(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print("----- raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
   
    prompts = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = torch.tensor(prompts['input_ids']).cuda()
    attention_mask =  torch.tensor(prompts['attention_mask']).cuda()
    
    output = model.generate(
        input_ids=input_ids, 
        generation_config=generation_config,
        attention_mask = attention_mask,)
    code = tokenizer.decode(output[0], skip_special_tokens=True)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        # "response": response,
        # "history": history,
        "status": 0,
        "result": {
            "process_time": time,
            "output":[
                code
            ]
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer


@app.post("/api/v1/multilingual_code_generate_adapt")
async def multilingual_code_generate_adapt(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print("----- raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
   
    prompts = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = torch.tensor(prompts['input_ids']).cuda()
    attention_mask =  torch.tensor(prompts['attention_mask']).cuda()
    
    output = model.generate(
        input_ids=input_ids, 
        generation_config=generation_config,
        attention_mask = attention_mask,)
    code = tokenizer.decode(output[0], skip_special_tokens=True)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        # "response": response,
        # "history": history,
        "status": 0,
        "result": {
            "process_time": time,
            "output":[
                code
            ]
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer


#  line by line
@app.post("/api/v1/multilingual_code_genline")
async def multilingual_code_genline(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print("-----code_genLine raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    context = json_post_list.get('context')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    gc = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # eos_token_id=tok.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=2,
        max_new_tokens=96,
        stopping_criteria=StoppingCriteriaList([stop_by_line])
    )
    
    start = datetime.datetime.now()
    # system_msg = "Below are a series of dialogues between various people and an AI assistant.\
    system_msg = "The AI assistant tries to complete the code. "
    # continue_prompt = "Human: ```\n" + context + "```\n"
    continue_prompt = ""

    prompt = system_token + "\n" + system_msg + "" + \
        end_token + "\n" + user_token + "\n" + continue_prompt + \
        end_token + "\n" + "Assistant: \n```" + context
        # end_token + "\n" + assistant_token + "\n"
    code = gen_result(prompt, gc)
    code = code[len(prompt):].split("\n\n\n\n")[0]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    print("len:", len(prompt), len(code))
    log = "[" + time + "] " + '", prompt:"' + prompt + code + '"'
    print(log)
    torch_gc()
    return answer


#  continue write
@app.post("/api/v1/multilingual_code_fim")
async def multilingual_code_fim(request: Request):
    json_post_raw = await request.json()
    print("----- code_continue_1 raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prefix = json_post_list.get('prefix')
    suffix = json_post_list.get('suffix')
    
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    global generation_config
    gc = generation_config
    start = datetime.datetime.now()
    continue_prompt = "Human: fill the following code between the prefix code and the suffix code, according to the context, give the full solution\n"
    # 
    continue_prompt += "<|fim_prefix|>```\n" + prefix + "```\n" + end_token
    continue_prompt += "<|fim_suffix|>```\n" + suffix + "```\n" + end_token

    prompt = system_token + "\n" + system_msg + \
        end_token + "\n" + user_token + "\n" + continue_prompt + \
        end_token + "\n" + "Assistant: "
        # end_token + "\n" + assistant_token + "\n"
    code = gen_result(prompt, gc)
    code = code[len(prompt):]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer

@app.post("/api/v1/multilingual_code_free")
async def multilingual_code_free(request: Request):
    json_post_raw = await request.json()
    print("----- code_free raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    instruct = json_post_list.get('instruct', "")
    context = json_post_list.get("context", "")
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    global generation_config
    gc = generation_config
    start = datetime.datetime.now()

    if len(context) > 0:
        human_instruct = "Human: " + instruct +  "\n```\n" + context + "```\n"
    else:
        human_instruct = "Human: " + instruct 
        

    prompt = system_token + "\n" + system_msg + \
        end_token + "\n" + user_token + "\n" + human_instruct +\
        end_token + "\n" + "Assistant: "
        # end_token + "\n" + assistant_token + "\n"

    code = gen_result(prompt, gc)
    code = code[len(prompt):].split("\n\n\n\n")[0]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer



# write module test
@app.post("/api/v1/multilingual_gen_test")
async def multilingual_gen_test(request: Request):
    json_post_raw = await request.json()
    print("----- gen_test raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    # instruct = json_post_list.get('instruct')
    instruct = " write test function for the code below"
    context = json_post_list.get("context")
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    global generation_config
    gc = generation_config
    start = datetime.datetime.now()

    if len(context) > 0:
        human_instruct = "Human: " + instruct +  "\n```\n" + context + "```\n"
    else:
        human_instruct = "Human: " + instruct 
        

    prompt = system_token + "\n" + system_msg + \
        end_token + "\n" + user_token + "\n" + human_instruct +\
        end_token + "\n" + "Assistant: "
        # end_token + "\n" + assistant_token + "\n"

    code = gen_result(prompt, gc)
    code = code[len(prompt)+1:].split("\n\n\n\n")[0]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer

# explain code 
@app.post("/api/v1/multilingual_code_explain")
async def multilingual_code_explain(request: Request):
    json_post_raw = await request.json()
    print("----- code_explain raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    # instruct = json_post_list.get('instruct')
    instruct = " What is the purpose of the code below? explain in 中文 "
    context = json_post_list.get("context")
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    global generation_config
    gc = generation_config
    start = datetime.datetime.now()

    if len(context) > 0:
        human_instruct = "Human: " + instruct +  "\n```\n" + context + "```\n"
    else:
        human_instruct = "Human: " + instruct 
        

    prompt = system_token + "\n" + system_msg + \
        end_token + "\n" + user_token + "\n" + human_instruct +\
        end_token + "\n" + "Assistant: "
        # end_token + "\n" + assistant_token + "\n"

    code = gen_result(prompt, gc)
    code = code[len(prompt)+1:].split("\n\n\n\n")[0]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer


# code debug
@app.post("/api/v1/multilingual_code_debug")
async def multilingual_code_debug(request: Request):
    json_post_raw = await request.json()
    print("----- code_debug raw request:")
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    # instruct = json_post_list.get('instruct')
    instruct = " the code below is not working, check whats maybe wrong "
    context = json_post_list.get("context")
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    global generation_config
    gc = generation_config
    start = datetime.datetime.now()

    if len(context) > 0:
        human_instruct = "Human: " + instruct +  "\n```\n" + context + "```\n"
    else:
        human_instruct = "Human: " + instruct 
        

    prompt = system_token + "\n" + system_msg + \
        end_token + "\n" + user_token + "\n" + human_instruct +\
        end_token + "\n" + "Assistant: "
        # end_token + "\n" + assistant_token + "\n"

    code = gen_result(prompt, gc)
    code = code[len(prompt):].split("\n\n\n\n")[0]
    end = datetime.datetime.now()
    time = end.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "status": 0,
        "result": {
            "process_time": end-start,
            "output": {
                "code" :[
                    code
                ]}
        },
        "message": "No message"
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + code + '"'
    print(log)
    torch_gc()
    return answer

# history continue
@app.post("/api/v1/multilingual_history_continue")
async def multilingual_history_continue(request: Request):
    pass


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    # model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=11073, workers=1)
