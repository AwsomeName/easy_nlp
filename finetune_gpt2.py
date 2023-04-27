# 旨在零基础的入门
from transformers import GPT2Tokenizer,  GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import transformers
from loguru import logger
import argparse
import sys

assert(
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "need llama"
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
import os
import torch
from transformers.trainer import TRAINING_ARGS_NAME


def train_or_pred():
    logger.info(args)
    logger.info("token...")
    tokenizer = AutoTokenizer.from_pretrained(args.pre_train, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("token done")
    
    logger.info("model...")
    model = AutoModelForCausalLM.from_pretrained(args.pre_train).cuda()
    logger.info("model done")
    
    # logger.info("train data ...")
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.train_file, block_size=128)
    # logger.info("test data...")
    test_dataset = TextDataset(tokenizer=tokenizer, file_path=args.test_file, block_size=128)
    
    # logger.info("colla data...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    logger.info("train data ...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = "steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs_test",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        disable_tqdm=False,
        save_total_limit=3,
        save_steps=100,
        save_strategy="steps",
        save_on_each_node=False,
        run_name="run_name",
    )
    
    logger.info("Trainer...")
    if True:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=test_dataset
        )
    
        logger.info("start...")
        
        # model.config.use_cache = False
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        (global_step, training_loss, metrics) = trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
        logger.info("Save checkpointing ...")
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Traing succeeded")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_file", default="data/instruct_train_1k.txt", type=str)
    parser.add_argument("--train_file", default="data/train_100.txt", type=str)
    parser.add_argument("--test_file", default="data/test_10.txt", type=str)
    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2", type=str)
    parser.add_argument("--pre_train", default="sshleifer/tiny-gpt2", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--output_dir", default="./outputs_test/")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    train_or_pred()
    
    
    
    
    
    
    
    
    