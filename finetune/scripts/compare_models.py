#!/usr/bin/env python3
"""Simple script to compare two language models using perplexity."""
import argparse
import json
from typing import Iterable

from evaluate import evaluate_model


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two models on the same dataset")
    parser.add_argument("model_a", help="First model path or name")
    parser.add_argument("model_b", help="Second model path or name")
    parser.add_argument("data", help="Path to JSONL evaluation dataset")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum length for tokenization")
    args = parser.parse_args(argv)

    ppl_a = evaluate_model(args.model_a, args.data, max_length=args.max_length)
    ppl_b = evaluate_model(args.model_b, args.data, max_length=args.max_length)

    results = {
        args.model_a: ppl_a,
        args.model_b: ppl_b,
        "better_model": args.model_a if ppl_a < ppl_b else args.model_b,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
