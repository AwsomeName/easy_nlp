# spongbobliu/test

from huggingface_hub import Repository
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoTokenizer
from utils import StarChatArgumentParser, hf_login
from config import DataArguments, ModelArguments, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

token = "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"
# repo = Repository("./", clone_from="spongbobliu/test")
# repo.git_pull()

# 提交模型
# model_path = "/root/easy_nlp/trl/outputs/sft/checkpoint-200"
if True:
    # model_path = "/root/data/bloom-560m"
    x_path = "/hy-tmp/starchat-alpha/checkpoint-500"
    tokenizer = AutoTokenizer.from_pretrained(x_path)

    # model = AutoModel.from_pretrained(x_path)
    print('Init done')
    # model.push_to_hub("spongbobliu/test_2", private=False, use_auth_token=token, create_pr=1)
    tokenizer.push_to_hub("spongbobliu/test_2", private=False, use_auth_token=token, create_pr=1)

# 提交全部
if False:
    import sys, os
    model_path = "/hy-tmp/bloom-560m"
    parser = StarChatArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("----init done")
    xpath = "/hy-tmp/starchat-alpha/checkpoint-500"
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            pass
    print("---- get last ckp done")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    token = "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"
    trainer.push_to_hub("spongbobliu/test_2", private=False, use_auth_token=token, create_pr=1)