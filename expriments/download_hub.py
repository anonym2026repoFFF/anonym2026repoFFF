import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def sanitize(name):
    return name.replace("/", "_").replace("-", "_")

def download_assets(model_name, dataset_name, dataset_config, local_data_dir, cache_dir):
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Make HF respect this cache dir even if other libs look at env vars
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))

    # Compose subdirectory names using sanitized keys
    model_dir = os.path.join(local_data_dir, sanitize(model_name))
    tokenizer_dir = os.path.join(local_data_dir, sanitize(model_name) + "_tokenizer")
    dataset_dir = os.path.join(local_data_dir, f"{sanitize(dataset_name)}_{sanitize(dataset_config)}")

    # 1. Download & save dataset
    print(f"Downloading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config , cache_dir=os.environ["HF_DATASETS_CACHE"])
    dataset.save_to_disk(dataset_dir)

    # 2. Download & save tokenizer
    print(f"Downloading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=os.environ["HF_TRANSFORMERS_CACHE"])
    tokenizer.save_pretrained(tokenizer_dir)

    # 3. Download & save model
    print(f"Downloading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.environ["HF_TRANSFORMERS_CACHE"])
    model.save_pretrained(model_dir)

    print("\nDownload complete. Assets saved to:")
    print(f"   - Model:     {model_dir}")
    print(f"   - Tokenizer: {tokenizer_dir}")
    print(f"   - Dataset:   {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model, tokenizer, and dataset for offline training.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., facebook/opt-125m)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., wikitext)")
    parser.add_argument("--dataset_config", type=str, required=True, help="Dataset config (e.g., wikitext-2-raw-v1)")
    parser.add_argument("--local_data_dir", type=str, default="local_data", help="Directory to save assets")
    parser.add_argument("--cache_dir", type=str, default="/p/scratch/eelsaisdc/fastopt/temp", help="Cache directory with plenty of space")

    args = parser.parse_args()

    download_assets(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        local_data_dir=args.local_data_dir,
        cache_dir=args.cache_dir
    )
