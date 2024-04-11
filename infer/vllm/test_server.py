from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.

# llm = LLM (model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
llm = LLM (model="/home/lc/models/codellama/CodeLlama-7b-Instruct-hf",
           tensor_parallel_size=2,
           dtype="float16")  # Create an LLM.

outputs = llm.generate (prompts)  # Generate texts from the prompts.
print(outputs)