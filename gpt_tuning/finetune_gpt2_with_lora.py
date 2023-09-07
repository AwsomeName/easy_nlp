from transformers import GPT2Tokenizer,  GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import transformers
from loguru import logger
import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
from peft import TaskType
import os
import torch
from transformers.trainer import TRAINING_ARGS_NAME

class FinetuneTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     return model(
    #         input_ids=inputs["input_ids"],
    #         # attention_mask=inputs["attention_mask"],
    #         # position_ids=inputs['position_ids'],
    #         labels=inputs['labels'],
    #     ).loss
        
    def save_model(self, output_dir=None, _internal_call=False, lora_name="adapter_model.bin"):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        # return super().save_model(output_dir, _internal_call)
        torch.save(saved_params, os.path.join(output_dir, lora_name))
    
    
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
    
    logger.info("train data ...")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = "steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs_test_lora",
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

    cuda_device = -1
    if args.use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                device = torch.device("cuda")
            else:
                device = torch.device(f"cuda:{cuda_device}")
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                "Make sure CUDA is available or set `use_cuda=False`."
            )
    else:
        device = "cpu"
    logger.debug(f"Device: {device}")
    logger.info("Trainer...")
    lora_loaded = False
    if args.do_train:
        if args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
            lora_loaded = True
        # move_model_to_device
        model.to(device)
    
        logger.info("start...")
        
        # model.config.use_cache = False
        # if args.use_lora:
            # old_state_dict = model.state_dict
            # model.state_dict = (
                # lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
            # ).__get__(model, type(model))

        # trainer = Trainer(
        trainer = FinetuneTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=test_dataset
        )
        
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        logger.info("trainer.train")
        (global_step, training_loss, metrics) = trainer.train()
        logger.info("Save checkpointing ...")
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Traing succeeded")
        
    elif args.do_predict:
        # nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)
        text = "你好吗？今天天气不错呢"
        max_length = 1024
        chosen_token = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        # input_ids = chosen_token['']
        input_ids = torch.tensor(chosen_token['input_ids']).cuda()
        attention_mask = torch.tensor(chosen_token['attention_mask']).cuda()
        if not lora_loaded:
            # load_lora()
            if args.use_lora:
                lora_path = os.path.join(args.output_dir, args.lora_name)
                if lora_path and os.path.exists(lora_path):
                    # infer with trained lora model
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=True,
                        r=args.lora_rank,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                    )
                    model = get_peft_model(model, peft_config)
                    model.load_state_dict(torch.load(lora_path), strict=False)
                    logger.info(f"Loaded lora model from {lora_path}")
                    lora_loaded = True
        # _move_model_to_device()
        model.to(device)
        model.eval()
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=200,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            num_return_sequences=1)
        
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("".join(output))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/instruct_train.txt", type=str)
    # parser.add_argument("--train_file", default="data/train_100.txt", type=str)
    parser.add_argument("--test_file", default="data/test_10.txt", type=str)
    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2", type=str)
    parser.add_argument("--pre_train", default="sshleifer/tiny-gpt2", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--output_dir", default="./outputs_test_lora_2/")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--lora_name", default="adapter_model.bin", type=str)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    train_or_pred()