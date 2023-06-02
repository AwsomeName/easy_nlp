# spongbobliu/test

from huggingface_hub import Repository
from transformers import AutoModel, AutoTokenizer

token = "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"
# repo = Repository("./", clone_from="spongbobliu/test")
# repo.git_pull()

# model_path = "/root/easy_nlp/trl/outputs/sft/checkpoint-200"
model_path = "/root/data/bloom-560m"

model = AutoModel.from_pretrained(model_path)
print('Init done')


model.push_to_hub("spongbobliu/test", private=False, use_auth_token=token, create_pr=1)
