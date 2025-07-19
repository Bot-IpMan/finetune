# Finetune

This repository demonstrates how to fine‑tune the [Qwen2.5‑Coder](https://github.com/QwenLM/Qwen) model using LoRA. All training steps are executed in a Docker container for a reproducible environment.

## Quick start

1. **Build the Docker image**
   ```bash
   cd finetune
   docker-compose build
   ```
2. **Start the container**
   ```bash
   docker-compose up -d
   ```
3. **Prepare the data**
   ```bash
   docker-compose exec finetune python /workspace/scripts/prepare_data.py
   ```
4. **Run training**
   ```bash
   docker-compose exec finetune python /workspace/scripts/train.py --config /workspace/configs/training_config.yaml
   ```
5. **Test the model**
   ```bash
   docker-compose exec finetune python /workspace/scripts/test_model.py
   ```

All these steps can also be executed with a single command:
```bash
./finetune/run_training.sh
```

The `finetune` directory contains the Dockerfile, configuration files and helper scripts for the project.
