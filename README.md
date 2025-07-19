# Finetune

This project provides scripts and configuration files for finetuning language models.

## Dataset format

Prepared datasets are stored in `finetune/data/processed`. Training data is saved to `train_dataset.jsonl` and evaluation data to `eval_dataset.jsonl`.
Each file uses the JSON Lines format where every line contains an object with the keys:

- `instruction` – text describing the task
- `input` – optional input for the model
- `output` – expected response

Example entry:

```json
{"instruction": "Напиши функцію для обчислення факторіалу", "input": "", "output": "def factorial(n): ..."}
```
