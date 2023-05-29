# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

acc_token = "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"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=acc_token)
model = AutoModelForCausalLM.from_pretrained(checkpoint, use_auth_token=acc_token).half().to(device)

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
