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

import wandb
from new_fff import FFF
from fastOPT import FastOPT  # Ensure fastOPT.py is on your PYTHONPATH

import torch

import math
from tqdm import tqdm

import matplotlib.pyplot as plt


# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     # logits shape: (batch_size, seq_len, vocab_size)
#     # labels shape: (batch_size, seq_len)
#     shift_logits = torch.tensor(logits[..., :-1, :])
#     shift_labels = torch.tensor(labels[..., 1:])
#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     perplexity = math.exp(loss.item())
#     return {"perplexity": perplexity}

def train(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
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
    max_samples: int,
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
        dataset_dir = os.path.join(local_data_dir, f"{sanitize(dataset_name)}_{sanitize(dataset_config)}")
        tokenizer_dir = os.path.join(local_data_dir, sanitize(model_name) + "_tokenizer")
        model_dir = os.path.join(local_data_dir, sanitize(model_name))

        for path in [dataset_dir, tokenizer_dir, model_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Offline path not found: {path}")

        # 2. Load dataset
        raw_datasets = load_from_disk(dataset_dir)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        test_dataset = raw_datasets["test"]

        # 3. Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir , use_fast=use_fast_tokenizer
        )
        model = FastOPT.from_pretrained(model_dir, depth=fff_depth, strategy=strategy)

    else:
        print("Loading from Hugging Face hub...")
        # 2. Load dataset from Hugging Face Hub
        raw_datasets = load_dataset(dataset_name, dataset_config)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        test_dataset = raw_datasets["test"]

        # 3. Load tokenizer & model from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        model = FastOPT.from_pretrained(model_name, depth=fff_depth, strategy=strategy)

    # 4. Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    tokenized_test = test_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # 5. Group texts into chunks of size `block_size`
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                concatenated[k][i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_tokenized = tokenized_train.map(group_texts, batched=True)
    eval_tokenized = tokenized_eval.map(group_texts, batched=True)

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
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_strategy=save_strategy,
        # save_steps=save_steps,
        save_total_limit=2, 
        fp16=fp16,
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
        # compute_metrics=compute_metrics, 
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Eval loss:", metrics["eval_loss"])
    print("Perplexity:", math.exp(metrics["eval_loss"]))

    if max_samples is not None:
        test_text = "\n\n".join(test_dataset["text"][:max_samples])
    else:
        test_text = "\n\n".join(test_dataset["text"])

    encodings = tokenizer(test_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    stride = 512
    max_length = model.config.max_position_embeddings

    decoder_layers = model.model.decoder.layers

    nlls = []
    prev_end_loc = 0

    device = model.device
    max_possible_steps = len(range(0, seq_len, stride))
    # Evaluate
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())

    # print(tree_load_layers[0])

    # Log to wandb
    if wandb_project and "wandb" in report_to:

        print(f"\nPerplexity: {ppl.item():.2f}")
        print("\nTree Load:")
        wandb.log({"perplexity": ppl.item()})
        wandb.log({"loss_raw": torch.stack(nlls).mean().item()})
        wandb.log({"loss_in build": metrics["eval_loss"]})
        wandb.log({"ppl_in build": math.exp(metrics["eval_loss"])})

        # for i, tree_load in enumerate(tree_load_layers):
        #     # Heatmap
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     heatmap = ax.imshow(tree_load.numpy(), aspect='auto', interpolation='nearest')
        #     plt.colorbar(heatmap, ax=ax)
        #     plt.title(f"Tree Load Heatmap - Layer {i}")
        #     plt.xlabel("Nodes")
        #     plt.ylabel("Parallel Units")
        #     wandb.log({f"tree_load_heatmap_layer_{i}": wandb.Image(fig)})
        #     plt.close(fig)

        #     # Histogram
        #     wandb.log({f"tree_load_distribution_layer_{i}": wandb.Histogram(tree_load.flatten().numpy())})

    # 10. Train & save final model
    
    trainer.save_model(output_dir)

    # 11. Finish WandB run if it was started
    if "wandb" in report_to:
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
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to use from test set.")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    local_data_dir = config.get("local_data_dir", "").strip()
    model_name = config["model_name"]
    dataset_name = config["dataset"]["name"]
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
        max_samples=args.max_samples,
    )



if __name__ == "__main__":
    main()
