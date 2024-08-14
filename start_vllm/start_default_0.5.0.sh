source /data/miniconda3/bin/activate
conda activate vllm0.5.0

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model /data/models/chatglm3-6b \
    --trust-remote-code \
    --dtype auto \
    --served-model-name chatglm3-6b \
    --port 3332 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 6000 \
    --tensor-parallel-size=1
    --chat-template ./template_chatglm.jinja \
    2>&1 1>vllm-3332.log &
    # --enable_lora \
    # --lora_modules glm3="/data/xxx/" \

