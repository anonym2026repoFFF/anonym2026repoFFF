import os
import argparse
import yaml
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import random


import random
from datasets import Dataset

def shard_is_usable(path, sample_rows=128, min_valid_ratio=0.9):
    """
    Quick health check:
    - has 'text' column
    - among the first `sample_rows`, at least `min_valid_ratio` look like non-empty strings
    """
    try:
        ds = Dataset.from_file(path)
    except Exception:
        return False, "failed_to_open"

    if "text" not in ds.features:
        return False, "no_text_column"

    # Fast probe on a small slice
    n = min(sample_rows, len(ds))
    if n == 0:
        return False, "empty_shard"

    try:
        probe = ds.select(range(n))  # cheap head()
    except Exception:
        return False, "select_failed"

    ok = 0
    for ex in probe:
        t = ex.get("text", None)
        if isinstance(t, str) and t.strip():
            ok += 1

    ratio = ok / n
    if ratio < min_valid_ratio:
        return False, f"low_valid_ratio_{ratio:.2f}"
    return True, "ok"


def select_valid_shards(all_shards, num_shards, seed=42, allow_using_all_when_zero=True):
    """
    Returns `selected_shards, bad_shards_info`.
    If `num_shards == 0` and allow_using_all_when_zero, uses all *valid* shards.
    Raises if not enough valid shards to meet `num_shards`.
    """
    rng = random.Random(seed)
    shuffled = all_shards[:]
    rng.shuffle(shuffled)

    valid = []
    bad = []   # list of (path, reason)

    # Single pass over all shards is enough because we shuffled
    for p in shuffled:
        if num_shards > 0 and len(valid) >= num_shards:
            break
        ok, reason = shard_is_usable(p)
        if ok:
            valid.append(p)
        else:
            bad.append((p, reason))

    if num_shards == 0 and allow_using_all_when_zero:
        if not valid:
            raise ValueError("No valid shards found at all.")
        return valid, bad

    if len(valid) < num_shards:
        msg = (
            f"Requested {num_shards} shards but only found {len(valid)} usable.\n"
            f"Bad shards ({len(bad)}):\n" +
            "\n".join(f" - {path} [{reason}]" for path, reason in bad[:20]) +
            ("\n ... (truncated)" if len(bad) > 20 else "")
        )
        raise ValueError(msg)

    return valid[:num_shards], bad


def sanitize(name):
    return name.replace("/", "_").replace("-", "_")

def preprocess_batch(examples, tokenizer, block_size):
    tok = tokenizer(
        examples["text"],
        return_special_tokens_mask=True,
        truncation=False,
    )

    concatenated = {k: sum(tok[k], []) for k in tok}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size

    result = {
        k: [
            concatenated[k][i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k in concatenated
    }
    result["labels"] = result["input_ids"].copy()
    result["num_tokens"] = [len(seq) for seq in result["input_ids"]]

    return result

def preprocess_batch(examples, tokenizer, block_size):
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "Tokenizer must have eos_token_id"

    # Tokenize each text separately (no truncation)
    tok = tokenizer(
        examples["text"],
        return_special_tokens_mask=True,
        truncation=False,
        add_special_tokens=False,  # keep raw pieces; we'll insert EOS manually
    )

    # Interleave EOS between docs to prevent cross-doc leakage
    input_ids = []
    special_masks = []
    for ids, sm in zip(tok["input_ids"], tok["special_tokens_mask"]):
        input_ids.extend(ids)
        special_masks.extend(sm)
        # append EOS as a separator
        input_ids.append(eos_id)
        special_masks.append(1)

    # Now chunk the long stream into fixed blocks
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    special_masks = special_masks[:total_length]

    def chunk(lst):
        return [lst[i:i+block_size] for i in range(0, total_length, block_size)]

    result = {
        "input_ids": chunk(input_ids),
        "attention_mask": chunk([1] * total_length),  # all ones (no padding)
        "special_tokens_mask": chunk(special_masks),
    }
    result["labels"] = [seq[:] for seq in result["input_ids"]]
    result["num_tokens"] = [len(seq) for seq in result["input_ids"]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    dataset_name = config["dataset"]["name"]
    dataset_config = config["dataset"]["sub_dataset"]
    big_dataset = bool(config.get("dataset", {}).get("big_dataset", False))
    num_shards = int(config.get("dataset", {}).get("num_shards", 10))
    use_fast_tokenizer = config.get("use_fast_tokenizer", True)
    block_size = int(config.get("block_size", 512))
    local_data_dir = config.get("local_data_dir", "").strip()
    save_path = config.get("save_data_dir", "").strip()
    save_path = os.path.join(save_path, f"tokenized_data_nshards_{num_shards}_blocksize_{block_size}")

    os.makedirs(save_path, exist_ok=True)

    # Load tokenizer
    tokenizer_dir = os.path.join(local_data_dir, sanitize(model_name) + "_tokenizer") if local_data_dir else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast_tokenizer)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if local_data_dir:
        print("Loading from local paths...")
        if big_dataset:
            dataset_dir = dataset_name
        else:
            dataset_dir = os.path.join(local_data_dir, f"{sanitize(dataset_name)}_{sanitize(dataset_config)}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Local dataset not found at: {dataset_dir}")
        
        for path in [dataset_dir, tokenizer_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Offline path not found: {path}")
            
        if big_dataset:
            # print("Big dataset mode: loading random shards...")
            # shards_dir = os.path.join(dataset_dir, "train")
            # all_shards = [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith(".arrow")]

            # if not all_shards:
            #     raise FileNotFoundError(f"No .arrow shards in {shards_dir}")

            # if len(all_shards) < num_shards:
            #     raise ValueError(f"Requested {num_shards} shards, but only {len(all_shards)} available.")
            
            # if num_shards == 0:
            #     selected_shards = all_shards
            #     print(f"Using all available shards. Total: {len(all_shards)} shards.")
            # else:
            #     selected_shards = random.sample(all_shards, num_shards)
            # train_dataset = concatenate_datasets([Dataset.from_file(p) for p in tqdm(selected_shards, desc="Loading shards")])
            # eval_dataset = load_from_disk(os.path.join(dataset_dir, "validation"))

            shards_dir = os.path.join(dataset_dir, "train")
            all_shards = [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith(".arrow")]

            if not all_shards:
                raise FileNotFoundError(f"No .arrow shards in {shards_dir}")

            print(f"Big dataset mode: {len(all_shards)} shards detected.")
            if num_shards == 0:
                print("num_shards=0 → use all *usable* shards.")
            else:
                print(f"Sampling {num_shards} usable shards (resampling if a pick is bad).")

            selected_shards, bad_info = select_valid_shards(all_shards, num_shards)

            if bad_info:
                print("Some shards were skipped during sampling because they failed health checks:")
                for p, reason in bad_info[:10]:
                    print(f" - {p} [{reason}]")
                if len(bad_info) > 10:
                    print(f" ... and {len(bad_info) - 10} more")

            # Now load ONLY the validated shards
            train_dataset = concatenate_datasets(
                [Dataset.from_file(p) for p in tqdm(selected_shards, desc="Loading validated shards")]
            )
            eval_dataset = load_from_disk(os.path.join(dataset_dir, "validation"))
        else:
            raw_datasets = load_from_disk(dataset_dir)
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
    else:
        if big_dataset:
            raise ValueError("Big dataset mode is not supported with HuggingFace Hub.")
        raw_datasets = load_dataset(dataset_name, dataset_config)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]

    # Tokenize
    print("Tokenizing training dataset...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_batch(x, tokenizer, block_size),
        batched=True,
        batch_size=1000,
        num_proc=48,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )

    print("Tokenizing evaluation dataset...")
    eval_tokenized = eval_dataset.map(
        lambda x: preprocess_batch(x, tokenizer, block_size),
        batched=True,
        batch_size=1000,
        num_proc=48,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
    )

    # Save tokenized datasets
    print("Saving tokenized datasets to disk...")
    train_tokenized.save_to_disk(os.path.join(save_path, "train"))
    eval_tokenized.save_to_disk(os.path.join(save_path, "validation"))

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    main()