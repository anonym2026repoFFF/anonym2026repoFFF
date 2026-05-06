# Project Overview

This repository contains the official anonymous implementation of our research project on efficient transformer-based language model training and inference. The project focuses on improving inference and generation strategies for autoregressive language models while maintaining compatibility with standard Hugging Face workflows.

The framework supports:

- Training transformer language models
- Dataset preprocessing and tokenization
- Evaluation on language modeling benchmarks
- Flexible inference strategies
- Custom configuration-based experimentation
- Checkpoint saving and loading
- Reproducible experiments

The implementation is designed to be modular and configurable for research experimentation.

---

# Abstract

At typical context lengths, the feed-forward MLP block accounts for a large share of a transformer's compute budget, motivating sparse alternatives to dense MLP blocks.
We study sparse, tree-structured feed-forward layers as drop-in replacements for MLP blocks in deep transformer architectures, enabling conditional computation via hard hierarchical routing without a separate router network.
%or auxiliary balancing losses.
We demonstrate for the first time that this form of tree-structured conditional sparsity can be applied for  autoregressive language modeling and downstream question answering, including zero- and few-shot settings, and its scalability beyond 1B parameters.
Despite activating fewer than 5\% of the feed-forward block's units per token, our models match dense baselines under controlled training and fine-tuning protocols.
We further analyze training dynamics and identify an emergent auto-pruning effect: the interaction of hard routing with asymmetric nonlinearities progressively deactivates unused paths, yielding partial conversion of dynamic routing into static structural sparsity.
We show that simple architectural choices can modulate this behavior and recover balanced trees without auxiliary losses.
Overall, our work demonstrates that tree-structured feed-forward layers provide a scalable and controllable mechanism for sparsifying large transformer models.

---

# Features

- Hugging Face Transformers integration
- Support for OPT-family language models
- Config-driven experimentation
- Fixed-length token grouping pipeline
- Training checkpoint support
- Evaluation utilities
- Custom inference strategies
- Reproducible research pipeline

---

# Repository Structure

```text
.
├── train.py
├── preprocess.py
├── eval_2.py
├── inference.py
├── download_hub.py
├── configs/
├── checkpoints/
├── cache/
├── outputs/
└── README.md
```

---

# Requirements

Minimal required packages:

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

Recommended installation:

```bash
pip install torch transformers datasets accelerate sentencepiece
```

---

# Download Model and Dataset

To download the base OPT-1.3B model and WikiText dataset:

```bash
python download_hub.py \
    --model_name facebook/opt-1.3b \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --local_data_dir /YOURFOLDER/fastopt/cache/
```

This command downloads:

- The pretrained OPT-1.3B model
- The WikiText-2 dataset
- Local cache files for training and evaluation

---

# Data Preprocessing

To preprocess the dataset, tokenize samples, and group sequences into fixed lengths:

```bash
python preprocess.py --config 1.3B_OPT_CONFIG_FILE.yaml
```

The configuration file can be modified based on the provided base configuration files in the main project directory.

Typical configurable parameters include:

- Sequence length
- Batch size
- Dataset location
- Tokenizer settings
- Training hyperparameters
- Output directories

---

# Training

To start training:

```bash
python train.py --config 1.3B_OPT_CONFIG_FILE.yaml
```

Training checkpoints and logs will be automatically saved during training.

---

# Evaluation

To evaluate a trained checkpoint:

```bash
python eval_2.py \
    --config config_125_FFF_sp.yaml \
    --checkpoint YOUR_CHECKPOINT_FOLDER/opt-finetuned-wikitext_20250829_105807/checkpoint-123
```

Evaluation supports loading intermediate or final checkpoints for benchmarking and validation.

---

# Inference

To run inference using a trained checkpoint:

```bash
python inference.py \
    --model_path ./MODEL_CHECKPOINT \
    --prompt "Once upon a time" \
    --depth 3 \
    --strategy FF
```

Example configurable inference options:

- Prompt text
- Search depth
- Generation strategy
- Decoding parameters
- Checkpoint path

---

# Configuration Files

Experiments are controlled through YAML configuration files.

Example options include:

```yaml
model_name: facebook/opt-1.3b
max_seq_length: 1024
batch_size: 4
learning_rate: 1e-5
num_train_epochs: 3
```

Users can duplicate and modify base configuration files to reproduce or extend experiments.

---

# Reproducibility

To reproduce experiments:

1. Download the pretrained model and dataset
2. Preprocess the dataset
3. Train using the provided configuration
4. Evaluate checkpoints
5. Run inference using trained checkpoints

Recommended workflow:

```bash
# Step 1: Download
python download_hub.py \
    --model_name facebook/opt-1.3b \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --local_data_dir /YOURFOLDER/fastopt/cache/

# Step 2: Preprocess
python preprocess.py --config YOUR_CONFIG.yaml

# Step 3: Train
python train.py --config YOUR_CONFIG.yaml

# Step 4: Evaluate
python eval_2.py \
    --config YOUR_CONFIG.yaml \
    --checkpoint YOUR_CHECKPOINT

# Step 5: Inference
python inference.py \
    --model_path YOUR_CHECKPOINT \
    --prompt "Hello"
```

---

# Notes

- The codebase is intended for research purposes.
- GPU acceleration is strongly recommended for training and evaluation.
- Tested primarily on NVIDIA CUDA-enabled devices.
- Results may vary depending on hardware and software versions.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{anonymous2026,
  title={Anonymous Research Submission},
  author={Anonymous Authors},
  journal={Anonymous Submission},
  year={2026}
}
```

---

# License

This repository is released for research and academic use only.

---

# Acknowledgements

This project builds upon:

- Hugging Face Transformers
- PyTorch
- Hugging Face Datasets
- Facebook OPT models
