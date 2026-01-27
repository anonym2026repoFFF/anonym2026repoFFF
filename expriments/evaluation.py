import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, OPTForCausalLM
from fastOPT import FastOPT
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_WANDB_LOGDIR='./wandb'


def sanitize(name):
    return name.replace("/", "_").replace("-", "_")


def evaluate(model_dir, tokenizer_dir=None, wandb_project=None, wandb_group=None,
             wandb_logdir=None, max_samples=None, local_data_dir=None,
             model_name=None, dataset_name=None, dataset_config=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if wandb_logdir is None:
        wandb_logdir = _DEFAULT_WANDB_LOGDIR

    # === Load tokenizer, dataset, and model ===
    if local_data_dir:
        print("Loading from local paths...")

        dataset_path = os.path.join(
            local_data_dir, f"{sanitize(dataset_name)}_{sanitize(dataset_config)}"
        )
        tokenizer_path = tokenizer_dir if tokenizer_dir else os.path.join(
            local_data_dir, sanitize(model_name) + "_tokenizer"
        )
        model_path = model_dir

        # Load dataset
        raw_dataset = load_from_disk(dataset_path)
        test_dataset = raw_dataset["test"]
    else:
        print("Loading from Hugging Face hub...")
        tokenizer_path = tokenizer_dir if tokenizer_dir else model_dir
        model_path = model_dir

        # Load dataset
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # model = FastOPT.from_pretrained(model_path, depth=3, strategy="FF" )
    model = OPTForCausalLM.from_pretrained(model_path)
    # model = OPTForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
    model.to(device)

    # Prepare input
    if max_samples is not None:
        test_text = "\n\n".join(test_dataset["text"][:max_samples])
    else:
        test_text = "\n\n".join(test_dataset["text"])

    encodings = tokenizer(test_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    stride = 512
    max_length = model.config.max_position_embeddings

    decoder_layers = model.model.decoder.layers
    # tree_load_layers = [
    #     torch.zeros(
    #     (model.model.decoder.layers[0].fc2.parallel_size,
    #      model.model.decoder.layers[0].fc2.n_nodes),
    #     dtype=torch.float32,
    #     ) for layer in decoder_layers
    # ]
    nlls = []
    prev_end_loc = 0

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
            for i, layer in enumerate(decoder_layers):
                if hasattr(layer.fc2, "last_decision_map") and False:
                    tree_load_layers[i] += layer.fc2.last_decision_map.cpu() / max_possible_steps
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nPerplexity: {ppl.item():.2f}")
    print("\nTree Load:")
    # print(tree_load_layers[0])

    # Log to wandb
    if wandb_project and False:
        wandb.init(project=wandb_project, group=wandb_group, dir=wandb_logdir, config={
            "model_dir": model_dir,
            "local_data_dir": local_data_dir,
            "dataset": dataset_name,
            "dataset_config": dataset_config
        })

        wandb.log({"perplexity": ppl.item()})
        for i, tree_load in enumerate(tree_load_layers):
            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            heatmap = ax.imshow(tree_load.numpy(), aspect='auto', interpolation='nearest')
            plt.colorbar(heatmap, ax=ax)
            plt.title(f"Tree Load Heatmap - Layer {i}")
            plt.xlabel("Nodes")
            plt.ylabel("Parallel Units")
            wandb.log({f"tree_load_heatmap_layer_{i}": wandb.Image(fig)})
            plt.close(fig)

            # Histogram
            wandb.log({f"tree_load_distribution_layer_{i}": wandb.Histogram(tree_load.flatten().numpy())})


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned FastOPT model (offline/online).")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Optional tokenizer directory.")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name.")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group name.")
    parser.add_argument("--wandb_logdir", type=str, default=None, help="WandB log directory name.")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to use from test set.")
    parser.add_argument("--local_data_dir", type=str, default=None, help="Path to local dataset/tokenizer/model.")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="Model name used for local path resolution.")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name (e.g., wikitext).")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config name.")

    args = parser.parse_args()

    if args.wandb_logdir is None:
        args.wandb_logdir = os.environ.get('WANDB_LOGDIR', None)

    evaluate(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_logdir=args.wandb_logdir,
        max_samples=args.max_samples,
        local_data_dir=args.local_data_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    main()
