import argparse
import torch
from transformers import AutoTokenizer
from new_fff import FFF
from fastOPT import FastOPT  # Ensure fastOPT.py is on your PYTHONPATH

def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained LLM.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text to generate from",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=30,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth argument for FastOPT (same as used in training)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="new_FFF",
        choices=["new_FFF", "FFF"],
        help="Strategy used in FastOPT (must match training config)",
    )
    args = parser.parse_args()

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load custom model
    model = FastOPT.from_pretrained(
        args.model_path,
        depth=args.depth,
        strategy=args.strategy,
    )
    model = model.to(device)
    model.eval()

    # Encode input
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Generated Text ===")
    print(output_text)

if __name__ == "__main__":
    main()
