from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList
import torch

test_data = load_dataset("json", data_files="/root/data/data/openai_humaneval/HumanEval.jsonl")
# test_data = load_dataset("json", data_files="/root/data/data/humaneval-x/data/python/data/humaneval.jsonl")

print(test_data)

# tok = AutoTokenizer.from_pretrained("/root/data/bloom", padding_side="left")
tok = AutoTokenizer.from_pretrained("/root/data/bloom-560m", padding_side="left")
# tok = AutoTokenizer.from_pretrained("/root/data/starchat", padding_side="left")
# tokenizer.pad_token = tokenizer.eos_token
tok.pad_token = tok.eos_token
print("init tok done")

# model = AutoModelForCausalLM.from_pretrained("/root/data/bloom").cuda()
model = AutoModelForCausalLM.from_pretrained("/root/data/bloom-560m").cuda()
# model = AutoModelForCausalLM.from_pretrained("/root/data/starchat").half().cuda()
model = model.eval()
print("init model done")

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
        # stopping_criteria=StoppingCriteriaList([stop])
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



with open("./test_problem_top6_sc_prompt.txt", 'w') as wp:
    for idx, data in enumerate(train_data):
        # print(data)
        # for d in data:
            # print("-----")
            # print(d)
            # print(data[d])
        
        system_msg = "Below are a series of dialogues between various people and an AI assistant.\
The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.\
The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed.\
It also tries to avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.\
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.\
-----\n"

        prompt = system_token + "\n" + system_msg + end_token + "\n" + user_token + "\n" + data['prompt'] + end_token + "\n" + assistant_token + "\n"
        # prompt = "continue write the code below\n\n \"\"\"" + data['prompt'] + "\"\"\""
        code = gen_code(model, tok, prompt)
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