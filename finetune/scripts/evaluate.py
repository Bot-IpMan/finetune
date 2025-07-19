#!/usr/bin/env python3
"""Utility script to evaluate a causal language model on a JSONL dataset."""
import argparse
import json
from pathlib import Path
from typing import Iterable, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def _load_dataset(file_path: str):
    """Load dataset from a JSONL file."""
    ds = load_dataset("json", data_files=file_path)["train"]
    return ds


def _format_example(example: Dict[str, str]) -> str:
    """Format a single training example into a prompt used during training."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text and input_text.strip():
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
    else:
        prompt = f"Instruction: {instruction}\nOutput: {output_text}"
    return prompt


def evaluate_model(model_path: str, data_file: str, max_length: int = 512) -> float:
    """Compute perplexity of ``model_path`` on ``data_file``."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    dataset = _load_dataset(data_file)

    total_loss = 0.0
    total_tokens = 0

    for ex in tqdm(dataset, desc="Evaluating", unit="example"):
        prompt = _format_example(ex)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss = out.loss

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

    if total_tokens == 0:
        return float("inf")

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("model", help="Path or name of the model to evaluate")
    parser.add_argument("data", help="Path to JSONL evaluation dataset")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum length for tokenization")
    args = parser.parse_args(argv)

    perplexity = evaluate_model(args.model, args.data, max_length=args.max_length)
    print(json.dumps({"model": args.model, "perplexity": perplexity}, indent=2))


if __name__ == "__main__":
    main()
