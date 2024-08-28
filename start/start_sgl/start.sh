source /data/miniconda3/bin/activate
conda activate sgl

CUDA_VISIBLE_DEVICES=2 nohup python -m sglang.launch_server \
    --model-path /data/models/glm3_raw/chatglm3-6b/ \
    --port 7771 \
    --trust-remote-code 