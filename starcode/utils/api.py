from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
debug = True
model_path = "/root/data/starchat"
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        # stopping_criteria=StoppingCriteriaList([stop])
    )

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


@app.post("/api/v1/multilingual_code_generate")
async def multilingual_code_generate(request: Request):
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

    

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=11073, workers=1)
