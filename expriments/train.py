import os, time, glob
import yaml
import argparse
from datetime import datetime

from transformers import AutoConfig

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

from transformers.trainer_utils import get_last_checkpoint

_DEFAULT_WANDB_LOGDIR='.'


def cleanup_rng_files(ckpt_dir):
    rng_files = glob.glob(os.path.join(ckpt_dir, "rng_state*.pth"))
    for f in rng_files:
        print(f"⚠️ Removing RNG state file: {f}")
        os.remove(f)

def is_rank_zero():
    rank = os.environ.get("RANK")
    return rank is None or rank == "0"

def train(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    big_dataset: bool,
    num_shards: int,
    use_fast_tokenizer: bool,
    block_size: int,
    dataloader_num_workers: int,
    output_dir: str,
    overwrite_output_dir: bool,
    evaluation_strategy: str,
    eval_steps: int,
    logging_strategy: str,
    logging_steps: int,
    learning_rate: float,
    optimizer: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    save_strategy: str,
    save_steps: int,
    fp16: bool,
    report_to: list,
    wandb_project: str = None,
    wandb_group: str = None,
    wandb_logdir: str = None,
    local_data_dir: str = "",
    fff_depth: int = 3,
    strategy: str = "new_FFF",
    num_train_tokens: int = 1000000,
    from_scratch: bool = False,
    resume_from_checkpoint: str | None = None,
    output_date: str | None = None,
    full_config: dict | None = None,   # <— add this
    config_path: str | None = None,    # <— optional: to attach the YAML file
    seed: int | None = None,
    data_seed: int | None = None,
    max_steps: int = -1,
):
    """
    Train an OPT model on a language modeling task. Optionally uses wandb
    if 'wandb' appears in report_to.

    Args:
        report_to: list of strings, e.g. ["wandb"] or [].
        wandb_project: required if "wandb" in report_to.
        wandb_group: optional name for the wandb run.
    """

    # make config consistent
    if full_config['model']['strategy'] in ['new_FFF', 'old_FFF']:
        full_config['model']['moe_experts'] = 0
        full_config['model']['moe_k'] = 0
    elif full_config['model']['strategy'] == 'moe':
        full_config['model']['fff_depth'] = 0
    elif full_config['model']['strategy'] == 'FF':
        full_config['model']['fff_depth'] = 0
        full_config['model']['moe_experts'] = 0
        full_config['model']['moe_k'] = 0

    # 1. Optionally initialize Weights & Biases
    if "wandb" in report_to and is_rank_zero():
        if wandb_logdir is None:
            wandb_logdir = _DEFAULT_WANDB_LOGDIR

        cfg = dict(full_config or {}) 
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            dir=wandb_logdir,
            config=cfg,
        )

        if config_path:
            wandb.save(config_path)

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

            # Load a separate validation set
            train_dataset_path = os.path.join(dataset_dir, "train")
            train_dataset = load_from_disk(train_dataset_path)
            eval_dataset_path = os.path.join(dataset_dir, "validation")
            eval_dataset = load_from_disk(eval_dataset_path)

            num_samples =   num_train_tokens // block_size
            if num_samples > len(train_dataset):    
                print(f"Warning: num_train_tokens ({num_train_tokens}) is larger than available training samples ({len(train_dataset)}). Using all available samples.")
                num_samples = len(train_dataset)
            train_tokenized = train_dataset.shuffle(seed=seed).select(range(num_samples))
            eval_tokenized = eval_dataset
            
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

        if from_scratch:
            config = AutoConfig.from_pretrained(model_dir)
            model = FastOPT(
                config,
                depth=fff_depth,
                strategy=strategy,
                moe_n_experts=full_config['model']['moe_experts'],
                moe_k=full_config['model']['moe_k']
            )
        else:
            model = FastOPT.from_pretrained(
                model_dir,
                depth=fff_depth,
                strategy=strategy,
                moe_n_experts=full_config['model']['moe_experts'],
                moe_k=full_config['model']['moe_k']
            )

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
        model = FastOPT.from_pretrained(
            model_name,
            depth=fff_depth,
            strategy=strategy,
            moe_n_experts=full_config['model']['moe_experts'],
            moe_k=full_config['model']['moe_k']
        )

    if gradient_checkpointing:
        # transformers requires use_cache=False when training with checkpointing
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)
        # Some models expose an explicit enabler; safe to call if present
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # 4. Tokenize datasets
    total_train_tokens = sum(train_tokenized["num_tokens"])
    print(f"Total train tokens: {total_train_tokens:,}")

    # total_eval_tokens = sum(eval_tokenized["num_tokens"])
    # print(f"Total eval tokens: {total_eval_tokens:,}")

    # 6. Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7. Verify model has lm_head
    assert hasattr(model, "lm_head"), "Model is missing `lm_head`!"
    print(f"Model head: {model.lm_head}")

    # Only append timestamp if NOT resuming and there is no existing checkpoint
    existing_ckpt = None
    if os.path.isdir(output_dir):
        existing_ckpt = get_last_checkpoint(output_dir)

    if resume_from_checkpoint is None and existing_ckpt is None:
        output_dir = output_dir + "_" + output_date

    # output_dir = output_dir + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8. Build TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        optim=optimizer,
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy=evaluation_strategy,
        # eval_steps=eval_steps,
        dataloader_num_workers=dataloader_num_workers,   # or as many CPU cores as you can spare
        # pin_memory=True,     
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio= warmup_ratio,
        lr_scheduler_type="cosine",  # or "linear"
        save_strategy=save_strategy,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        save_total_limit=2, 
        # fp16=fp16,
        bf16=fp16, fp16=False,
        torch_compile=True, 
        report_to=report_to,  # either ["wandb"] or []
        seed=seed,
        data_seed=data_seed,
        max_steps=max_steps,
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

    # Decide what to resume from
    checkpoint_to_resume = None
    if resume_from_checkpoint:
        checkpoint_to_resume = resume_from_checkpoint
    else:
        # auto-pick latest in output_dir if present
        latest = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        if latest:
            checkpoint_to_resume = latest

    if checkpoint_to_resume:
        print(f"Resuming from checkpoint: {checkpoint_to_resume}")
        cleanup_rng_files(checkpoint_to_resume)

    t_start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    t_end = time.time()

    metrics = trainer.evaluate()
    print("Eval loss:", metrics["eval_loss"])
    print("Perplexity:", math.exp(metrics["eval_loss"]))

    # trainer.save_model(output_dir)

    # 11. Finish WandB run if it was started
    if "wandb" in report_to and is_rank_zero():
        wandb.log({"loss_in build": metrics["eval_loss"]})
        wandb.log({"ppl_in build": math.exp(metrics["eval_loss"])})
        wandb.log({"training_time": t_end - t_start})
        wandb.config.update({"output_dir": output_dir})
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
    parser.add_argument("--runstamp", type=str, required=True)
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
    num_train_tokens = int(float(config.get("dataset", {}).get("num_train_tokens", 100000)))

    use_fast_tokenizer = bool(config.get("use_fast_tokenizer", False))
    block_size = int(config.get("block_size", 512))

    dataloader_num_workers = int(config.get("dataloader_num_workers", 8))

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
    optimizer = training_cfg.get("optim", "adamw")
    per_device_train_batch_size = int(training_cfg["per_device_train_batch_size"])
    per_device_eval_batch_size = int(training_cfg["per_device_eval_batch_size"])
    num_train_epochs = int(training_cfg["num_train_epochs"])
    weight_decay = float(training_cfg["weight_decay"])
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    gradient_checkpointing = bool(training_cfg.get("gradient_checkpointing", False))
    max_steps = int(training_cfg.get("max_steps", -1))
    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", None)
    seed = int(training_cfg.get("seed", 42))
    data_seed = int(training_cfg.get("data_seed", seed))

    save_strategy = training_cfg["save_strategy"]
    save_steps = int(training_cfg["save_steps"])
    fp16 = bool(training_cfg["fp16"])

    model_cfg = config.get("model", {})
    fff_depth = int(model_cfg.get("fff_depth", 3)) 
    strategy = model_cfg.get("strategy", "new_FFF")  
    from_scratch = bool(model_cfg.get("from_scratch", False))

    report_to_cfg = training_cfg.get("report_to", None)
    if report_to_cfg and str(report_to_cfg).lower() != "none":
        report_to = [report_to_cfg]
    else:
        report_to = []

    wandb_project = training_cfg.get("wandb_project", None)
    wandb_group = training_cfg.get("group", None)
    wandb_logdir = training_cfg.get("wandb_logdir", None)
    if wandb_logdir is None:
        wandb_logdir = os.environ.get('WANDB_LOGDIR', None)

    train(
        local_data_dir=local_data_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        big_dataset=big_dataset,
        num_shards=num_shards,
        use_fast_tokenizer=use_fast_tokenizer,
        block_size=block_size,
        dataloader_num_workers=dataloader_num_workers,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        optimizer=optimizer,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy=save_strategy,
        save_steps=save_steps,
        fp16=fp16,
        report_to=report_to,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_logdir=wandb_logdir,
        fff_depth=fff_depth,
        strategy=strategy,
        num_train_tokens=num_train_tokens,
        from_scratch=from_scratch,
        resume_from_checkpoint=resume_from_checkpoint,
        full_config=config,
        output_date=args.runstamp,
        config_path=config_path,   # optional
        seed=seed,
        data_seed=data_seed,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    main()
