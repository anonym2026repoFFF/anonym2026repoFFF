import os
import yaml
import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, load_from_disk
from datasets import Dataset, concatenate_datasets

import wandb
from new_fff import FFF
from fastOPT import FastOPT  # Ensure fastOPT.py is on your PYTHONPATH

import torch

import math
import random
from tqdm import tqdm

def train(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    big_dataset: bool,
    num_shards: int,
    use_fast_tokenizer: bool,
    block_size: int,
    output_dir: str,
    overwrite_output_dir: bool,
    evaluation_strategy: str,
    eval_steps: int,
    logging_strategy: str,
    logging_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: int,
    weight_decay: float,
    save_strategy: str,
    save_steps: int,
    fp16: bool,
    report_to: list,
    wandb_project: str = None,
    wandb_group: str = None,
    local_data_dir: str = "",
    fff_depth: int = 3,
    strategy: str = "new_FFF",
):
    """
    Train an OPT model on a language modeling task. Optionally uses wandb
    if 'wandb' appears in report_to.

    Args:
        report_to: list of strings, e.g. ["wandb"] or [].
        wandb_project: required if "wandb" in report_to.
        wandb_group: optional name for the wandb run.
    """
    # 1. Optionally initialize Weights & Biases
    if "wandb" in report_to:
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "use_fast_tokenizer": use_fast_tokenizer,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "fp16": fp16,
                "fff_depth": fff_depth,
                "strategy": strategy,
                "output_dir": output_dir,
            },
        )

    def sanitize(name):
        return name.replace("/", "_").replace("-", "_")

    if local_data_dir:
        print("Loading from local paths...")
        if big_dataset:
            dataset_dir = dataset_name
        else:
            dataset_dir = os.path.join(local_data_dir, f"{sanitize(dataset_name)}_{sanitize(dataset_config)}")
            
        tokenizer_dir = os.path.join(local_data_dir, sanitize(model_name) + "_tokenizer")
        model_dir = os.path.join(local_data_dir, sanitize(model_name))

        for path in [dataset_dir, tokenizer_dir, model_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Offline path not found: {path}")


        if big_dataset:
            print("Big dataset mode: loading random shards...")

            shards_dir = os.path.join(dataset_dir, "train")
            all_shards = [
                os.path.join(shards_dir, f) for f in os.listdir(shards_dir)
                if f.endswith(".arrow")
            ]

            if len(all_shards) < num_shards:
                raise ValueError(f"Requested {num_shards} shards, but only {len(all_shards)} available.")

            # random.seed(42)
            selected_shards = random.sample(all_shards, num_shards)

            train_datasets = []
            for path in tqdm(selected_shards, desc="Loading training shards"):
                ds = Dataset.from_file(path)
                train_datasets.append(ds)

            train_dataset = concatenate_datasets(train_datasets)

            # Load a separate validation set
            eval_dataset_path = os.path.join(dataset_dir, "validation")
            eval_dataset = load_from_disk(eval_dataset_path)

        else:
            print("Standard dataset mode.")
            # 2. Load dataset
            raw_datasets = load_from_disk(dataset_dir)
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]

        # 3. Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir , use_fast=use_fast_tokenizer
        )
        model = FastOPT.from_pretrained(model_dir, depth=fff_depth, strategy=strategy)

    else:
        print("Loading from Hugging Face hub...")
        # 2. Load dataset from Hugging Face Hub
        if big_dataset:
            raise ValueError("Big dataset mode is not supported when loading from Hugging Face Hub.")
        else :
            raw_datasets = load_dataset(dataset_name, dataset_config)
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]

        # 3. Load tokenizer & model from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer,
        )
        model = FastOPT.from_pretrained(model_name, depth=fff_depth, strategy=strategy)

    # just keep the text column
    # train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "text"])
    # eval_dataset  = eval_dataset.remove_columns([c for c in eval_dataset.column_names  if c != "text"])

    def preprocess_batch(examples):
        # 1) tokenize without truncation
        tok = tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            truncation=False,
        )

        # 2) concatenate lists of tokens across the batch
        concatenated = {k: sum(tok[k], []) for k in tok}

        # 3) truncate to a multiple of block_size
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        # 4) split into blocks and set labels
        result = {
            k: [
                concatenated[k][i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k in concatenated
        }
        result["labels"] = result["input_ids"].copy()

        # 5) record how many tokens each example contributes
        result["num_tokens"] = [len(seq) for seq in result["input_ids"]]

        return result

    train_tokenized = train_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        num_proc=48,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )

    eval_tokenized = eval_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        num_proc=48,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
    )

    total_train_tokens = sum(train_tokenized["num_tokens"])
    print(f"Total train tokens: {total_train_tokens:,}")

    # total_eval_tokens = sum(eval_tokenized["num_tokens"])
    # print(f"Total eval tokens: {total_eval_tokens:,}")

    # 6. Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7. Verify model has lm_head
    assert hasattr(model, "lm_head"), "Model is missing `lm_head`!"
    print(f"Model head: {model.lm_head}")
    output_dir = output_dir + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8. Build TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy=evaluation_strategy,
        # eval_steps=eval_steps,
        dataloader_num_workers=48,   # or as many CPU cores as you can spare
        # pin_memory=True,     
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=500,
        lr_scheduler_type="cosine",  # or "linear"
        save_strategy=save_strategy,
        # save_steps=save_steps,
        save_total_limit=2, 
        # fp16=fp16,
        bf16=fp16, fp16=False,
        torch_compile=True, 
        report_to=report_to,  # either ["wandb"] or []
    )

    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 10. Train & save final model
    trainer.train()

    metrics = trainer.evaluate()
    print("Eval loss:", metrics["eval_loss"])
    print("Perplexity:", math.exp(metrics["eval_loss"]))

    trainer.save_model(output_dir)

    # 11. Finish WandB run if it was started
    if "wandb" in report_to:
        wandb.log({"loss_in build": metrics["eval_loss"]})
        wandb.log({"ppl_in build": math.exp(metrics["eval_loss"])})
        wandb.finish()


def main():
    # Parse command-line argument
    parser = argparse.ArgumentParser(description="Train a language model with specified YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    local_data_dir = config.get("local_data_dir", "").strip()
    model_name = config["model_name"]
    dataset_name = config["dataset"]["name"]
    big_dataset = bool(config.get("dataset", {}).get("big_dataset", False))
    num_shards = int(config.get("dataset", {}).get("num_shards", 10))
    dataset_config = config["dataset"]["sub_dataset"]
    use_fast_tokenizer = bool(config.get("use_fast_tokenizer", False))
    block_size = int(config.get("block_size", 512))

    training_cfg = config["training"]
    output_dir = training_cfg["output_dir"]
    overwrite_output_dir = bool(training_cfg["overwrite_output_dir"])
    evaluation_strategy = training_cfg["evaluation_strategy"]

    eval_steps = int(training_cfg["eval_steps"])
    logging_strategy = training_cfg.get("logging_strategy", "no")
    logging_steps = training_cfg.get("logging_steps", None)
    if logging_steps is not None:
        logging_steps = int(logging_steps)

    learning_rate = float(training_cfg["learning_rate"])
    per_device_train_batch_size = int(training_cfg["per_device_train_batch_size"])
    per_device_eval_batch_size = int(training_cfg["per_device_eval_batch_size"])
    num_train_epochs = int(training_cfg["num_train_epochs"])
    weight_decay = float(training_cfg["weight_decay"])

    save_strategy = training_cfg["save_strategy"]
    save_steps = int(training_cfg["save_steps"])
    fp16 = bool(training_cfg["fp16"])

    model_cfg = config.get("model", {})
    fff_depth = int(model_cfg.get("fff_depth", 3)) 
    strategy = model_cfg.get("strategy", "new_FFF")  

    report_to_cfg = training_cfg.get("report_to", None)
    if report_to_cfg and str(report_to_cfg).lower() != "none":
        report_to = [report_to_cfg]
    else:
        report_to = []

    wandb_project = training_cfg.get("wandb_project", None)
    wandb_group = training_cfg.get("group", None)

    train(
        local_data_dir=local_data_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        big_dataset=big_dataset,
        num_shards=num_shards,
        use_fast_tokenizer=use_fast_tokenizer,
        block_size=block_size,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_strategy=save_strategy,
        save_steps=save_steps,
        fp16=fp16,
        report_to=report_to,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        fff_depth=fff_depth,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
