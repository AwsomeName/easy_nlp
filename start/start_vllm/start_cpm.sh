source /data/miniconda3/bin/activate
conda activate vllm-p39

# nohup \ 
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
# CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server \
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
        /data/models/test/minicpm-v-2_6 \
        --trust-remote-code \
        --dtype auto \
        --served-model-name VQA \
        --port 33335 \
        --max-model-len 6000 \
        --tensor-parallel-size=2 \
        # --chat-template ./template_chatglm.jinja \
        # 2>&1 1>vllm-3335.log  &
        #--lora-modules chatglm2="/data/models/epoch-2-step-1250" \
        # --gpu-memory-utilization 0.4 \
        # --lora-modules glm4lora="/data/models/checkpoint-3000" \
        # --model /data/models/glm-4-9b-chat \
