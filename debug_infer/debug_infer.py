# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# checkpoint = "bigcode/starcoder"
# model_path = "/hy_tmp/Qwen2.5-0.5B-Instruct"
device = "cuda" # for GPU usage or "cpu" for CPU usage
# model_path = "/hy-tmp/Qwen2.5-0.5B-Instruct"
model_path = "/hy-tmp/YJT2.5-0.5B-Instruct"

# acc_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).half().to(device)

# inputs = tokenizer.encode("Humman: 完善以下函数，并生成测试调用\n\ndef print_hello_world(): \n\nAssistent: ", return_tensors="pt").to(device)
inputs = tokenizer.encode("Humman: 用Python实现一个快排 \n\nAssistent: ", return_tensors="pt").to(device)

outputs = model.generate(
            inputs,
            # attention_mask=attention_mask,
            max_length=2000,
            do_sample=True,
            top_k = 5,
            top_p=0.95,
            num_return_sequences=1)
# outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
