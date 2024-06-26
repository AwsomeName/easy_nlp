# 旨在零基础的入门
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from datasets import load_dataset
import argparse
import torch

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
        (global_step, training_loss, metrics) = trainer.train(
            resume_from_checkpoint = args.resume_from_checkpoint)
        logger.info("Save checkpointing ...")
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Traing succeeded")
    elif args.do_predict:
        # nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)
        text = "你好吗？今天天气不错"
        max_length = 1024
        chosen_token = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = torch.tensor(chosen_token["input_ids"]).cuda()
        attention_mask = torch.tensor(chosen_token['attention_mask']).cuda()
        checkpoint_path = args.output_dir + "checkpoint-100/pytorch_model.bin"
        checkpoint = torch.load(checkpoint_path)
        print("checkpoint:", checkpoint)
        model.load_state_dict(checkpoint)
        model.eval()
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=200,
            do_sample=True,
            top_k = 5,
            top_p=0.95,
            num_return_sequences=1)
        
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("".join(output))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_file", default="data/instruct_train_1k.txt", type=str)
    parser.add_argument("--train_file", default="/home/lc/code/easy_nlp/data/train_100.txt", type=str)
    parser.add_argument("--test_file", default="/home/lc/code/easy_nlp/data/test_10.txt", type=str)
    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--model_name", default="/home/lc/models/sshleifer/tiny-gpt2", type=str)
    # parser.add_argument("--model_name", default="sshleifer/tiny-gpt2", type=str)
    parser.add_argument("--pre_train", default="/home/lc/models/sshleifer/tiny-gpt2", type=str)
    # parser.add_argument("--pre_train", default="sshleifer/tiny-gpt2", type=str)
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
    