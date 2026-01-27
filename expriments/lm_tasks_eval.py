import os, time, glob
import yaml
import argparse
from datetime import datetime

from transformers import AutoConfig

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    OPTConfig,
    OPTForCausalLM,
)
from datasets import load_dataset, load_from_disk
from datasets import Dataset, concatenate_datasets
from lm_eval import evaluator

import wandb
from new_fff_2 import FFF
from fastOPT_3 import FastOPT, FastOPTConfig  # Ensure fastOPT.py is on your PYTHONPATH

import torch

import math
import random
from tqdm import tqdm
import json

from lm_eval.__main__ import cli_evaluate

from transformers.trainer_utils import get_last_checkpoint

_DEFAULT_WANDB_LOGDIR='.'

def sanitize(name):
    return name.replace("/", "_").replace("-", "_")



def print_group_results(results):
    """Print plain-text results for all tasks with acc/acc_norm and stderrs."""
    res = results.get("results", {})
    for task, metrics in res.items():
        acc = metrics.get("acc,none")
        acc_stderr = metrics.get("acc_stderr,none")
        acc_norm = metrics.get("acc_norm,none")
        acc_norm_stderr = metrics.get("acc_norm_stderr,none")
        f1 = metrics.get("f1,none")
        f1_stderr = metrics.get("f1_stderr,none")

        parts = []
        if acc is not None:
            parts.append(f"acc={acc:.3f}" + (f" ±{acc_stderr:.3f}" if acc_stderr else ""))
        if acc_norm is not None:
            parts.append(f"acc_norm={acc_norm:.3f}" + (f" ±{acc_norm_stderr:.3f}" if acc_norm_stderr else ""))
        if f1 is not None:
            parts.append(f"f1={f1:.3f}" + (f" ±{f1_stderr:.3f}" if f1_stderr else ""))

        line = f"{task}: " + ", ".join(parts) if parts else f"{task}: N/A"
        print(line)


def summarize_grouped_tasks(results):
    """Automatically average multi-subtask benchmarks like MMLU and BBH."""
    grouped = {}
    for task, metrics in results["results"].items():
        prefix = task.split("_")[0]  # e.g., "mmlu_math" → "mmlu"
        val = metrics.get("acc_norm,none", metrics.get("acc,none", None))
        if val is not None:
            grouped.setdefault(prefix, []).append(val)

    printed = False
    for group, vals in grouped.items():
        if len(vals) > 1:  # multiple subtasks → average them
            mean_val = sum(vals) / len(vals)
            print(f"\n📘 {group.upper()} average over {len(vals)} subtasks: {mean_val:.4f}")
            printed = True
    if not printed:
        print("\n(no multi-subtask groups found)")

def run_eval_groups(export_dir, tokenizer_dir, device="cuda:0"):
    groups = [
        {"name": "zero-shot", "tasks": ["hellaswag", "openbookqa", "winogrande", "arc_challenge","arc_easy", "boolq", "piqa"], "shots": 0},
        {"name": "3-shot", "tasks": ["bbh", "drop"], "shots": 3},
        {"name": "5-shot", "tasks": ["mmlu"], "shots": 5},
    ]

    for g in groups:
        print(f"\n🚀 Running {g['name']} evaluation ({g['shots']}-shot) ...")

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={export_dir},tokenizer={tokenizer_dir},dtype=bfloat16",
            tasks=g["tasks"],
            device=device,
            batch_size="auto",   # automatic batch-size detection
            num_fewshot=g["shots"],
            limit=None,
        )

        print_group_results(results)
        summarize_grouped_tasks(results)


def eval(
    model_name: str,
    export_dir: str,
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
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
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
    # if "wandb" in report_to and is_rank_zero():
    #     if wandb_logdir is None:
    #         wandb_logdir = _DEFAULT_WANDB_LOGDIR

    #     cfg = dict(full_config or {}) 
    #     wandb.init(
    #         project=wandb_project,
    #         group=wandb_group,
    #         dir=wandb_logdir,
    #         config=cfg,
    #     )

    #     if config_path:
    #         wandb.save(config_path)


        
    tokenizer_dir = os.path.join(local_data_dir, sanitize(model_name) + "_tokenizer")
    model_dir = os.path.join(local_data_dir, sanitize(model_name))

    for path in [ tokenizer_dir, model_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline path not found: {path}")

    if os.path.isdir(output_dir):
        existing_ckpt = get_last_checkpoint(output_dir)

    print("🔧 Converting checkpoint...")

    # Start from OPT config and extend
    config = FastOPTConfig.from_pretrained(model_dir)
    config.model_type = "fast_opt"
    config.depth = fff_depth
    config.strategy = strategy
    config.moe_n_experts = full_config['model']['moe_experts']
    config.moe_k = full_config['model']['moe_k']

    if existing_ckpt is not None:
        print(' WARNING: resuming or fine-tuning from existing checkpoint ')
        model = FastOPT.from_pretrained(
            existing_ckpt,
            config=config
        )

    else:
        raise NotImplementedError("Evaluation only supports from scratch")

    # model.config = config
    print('###########################')
    print(config)
    print(model.config)


    model.save_pretrained(export_dir)
    config.save_pretrained(export_dir)
    print(f"✅ Saved HF-compatible model to {export_dir}")

    AutoConfig.register("fast_opt", FastOPTConfig)
    AutoModelForCausalLM.register(FastOPTConfig, FastOPT)
    print("🔗 Registered 'fast_opt' with Hugging Face Auto classes.")
    run_eval_groups(export_dir=export_dir, tokenizer_dir=tokenizer_dir)
    # print("🚀 Running evaluation...")
    # results = evaluator.simple_evaluate(
    #     model="hf",
    #     model_args=f"pretrained={export_dir},tokenizer={tokenizer_dir},dtype=bfloat16",
    #     tasks=["piqa"],
    #     device="cuda:0",
    #     batch_size= "auto" # per_device_eval_batch_size,
    #     limit=10,
    #     num_fewshot=0,
    #     trust_remote_code=True,
    # )

    # print(evaluator.make_table(results))



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
    export_dir = config.get("export_dir", "./exported_model").strip()
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
    adam_beta1 = float(training_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(training_cfg.get("adam_beta2", 0.999))
    adam_epsilon = float(training_cfg.get("adam_epsilon", 1e-8))
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

    eval(
        local_data_dir=local_data_dir,
        export_dir=export_dir,
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
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
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
