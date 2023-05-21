from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import transformers
from loguru import logger
import argparse
import sys

import os
import torch
from transformers.trainer import TRAINING_ARGS_NAME
    
def train_or_pred(
        train_on_inputs: bool = False, # if False, masks out inputs in loss
    ):
    
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors=None,
        )
    
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < args.max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
            
        if add_eos_token and len(result["input_ids"]) >= args.max_length:
            result["input_ids"][args.max_length - 1] = tokenizer.eos_token_id
            result['attention_mask'][args.max_length - 1] = 1
            
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    def generate_and_tokenize_prompt(data_point):
        input_text = data_point['instruct']
        input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
        target_text = data_point["ans"] + tokenizer.eos_token
        full_prompt = input_text + target_text
        tokenized_full_prompt = tokenize(full_prompt)
        # print(tokenized_full_prompt)
        # print("----------")
        if not train_on_inputs:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
            
        return tokenized_full_prompt
    
    logger.info(args)
    tokenizer = AutoTokenizer.from_pretrained(args.pre_train, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.pre_train).cuda()
    logger.info("init done")
    
    
    logger.info("process data ...")
    data = load_dataset("json", data_files=args.train_file)
    train_val = data["train"].train_test_split(
        test_size=0.1, shuffle=False, seed=42
    )
    train_dataset = train_val['train'].shuffle().map(generate_and_tokenize_prompt)
    test_dataset = train_val['test'].shuffle().map(generate_and_tokenize_prompt)
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    
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
        save_total_limit=2,
        save_steps=100,
        save_strategy="steps",
        save_on_each_node=False,
        run_name="run_name",
    )
    
    logger.info("Trainer...")
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if args.do_train:
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
        # logger.info("trainer.train")
        (global_step, training_loss, metrics) = trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
        logger.info("Save checkpointing ...")
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Traing succeeded")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/train.json", type=str)
    parser.add_argument("--test_file", default="data/dev.json", type=str)
    # parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--model_name", default="bigscience/bloom-560m", type=str)
    parser.add_argument("--pre_train", default="bigscience/bloom-560m", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--output_dir", default="./outputs/sft")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    train_or_pred()