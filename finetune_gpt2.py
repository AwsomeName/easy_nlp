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

class FinetuneTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs['position_ids'],
            labels=inputs['labels'],
        ).loss
        
    def save_model(self, output_dir=None, _internal_call=False, lora_name="adapter_model.bin"):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        # return super().save_model(output_dir, _internal_call)
        torch.save(saved_params, os.path.join(output_dir, lora_name))
    
    
def train_or_pred(
        train_on_inputs: bool = True, # if False, masks out inputs in loss
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
            result["input_ids"][-1] != tokenize.eos_token_id
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
        instruction = data_point['instruction']
        input_text = data_point["input"]
        input_text = "Human: " + instruction + input_text + "\n\nAssistant: "
        # print("---------")
        # print(input_text)
        input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
        target_text = data_point["output"] + tokenizer.eos_token
        full_prompt = input_text + target_text
        # print(full_prompt)
        # print("---------1-")
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
    logger.info("token...")
    tokenizer = AutoTokenizer.from_pretrained(args.pre_train, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("token done")
    
    logger.info("model...")
    model = AutoModelForCausalLM.from_pretrained(args.pre_train).cuda()
    logger.info("model done")
    
    # logger.info("train data ...")
    # train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.train_file, block_size=128)
    # logger.info("test data...")
    test_dataset = TextDataset(tokenizer=tokenizer, file_path=args.test_file, block_size=128)
    
    # logger.info("colla data...")
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    logger.info("train data ...")
    data = load_dataset("json", data_files=args.train_file)
    # print(type(data))
    # print(data)
    # print("-------")
    # exit()
    train_val = data["train"].train_test_split(
        test_size=0.1, shuffle=True, seed=42
    )
    # print(train_val)
    # print("--------")
    # train_dataset = train_val['train'].shuffle()
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
        save_total_limit=3,
        save_steps=100,
        save_strategy="steps",
        save_on_each_node=False,
        run_name="run_name",
    )
    
    logger.info("Trainer...")
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if True:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=test_dataset
        )
    
        logger.info("start...")
        # trainer.train()
        # (global_step, training_loss, metrics) = trainer.train()
        
        # model.config.use_cache = False
        # if args.use_lora:
        #     old_state_dict = model.state_dict
        #     model.state_dict = (
        #         lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        #     ).__get__(model, type(model))
        
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
    parser.add_argument("--train_file", default="data/instruct_train_1k.txt", type=str)
    # parser.add_argument("--train_file", default="data/train_100.txt", type=str)
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
    
    
    
    
    
    
    
    
    