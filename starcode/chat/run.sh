pip install -r requirements.txt
sudo apt install git-lfs
huggingface-cli login
torchrun --nproc_per_node=8 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json