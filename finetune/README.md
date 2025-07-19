# Finetune Project

This directory contains the Docker environment and scripts used to fine-tune the Qwen2.5-Coder model.

## Структура проекту
- `Dockerfile` – базовий образ з CUDA та залежностями.
- `docker-compose.yml` – конфігурація сервісу `finetune`.
- `configs/` – YAML файли з параметрами тренування.
- `data/` – сирі та оброблені дані.
- `scripts/` – підготовка даних, тренування та тестування.
- `run_training.sh` – скрипт, що виконує всі етапи послідовно.

## Покроковий запуск
1. Побудуйте Docker образ
   ```bash
   docker-compose build
   ```
2. Запустіть контейнер
   ```bash
   docker-compose up -d
   ```
3. Підготуйте дані
   ```bash
   docker-compose exec finetune python /workspace/scripts/prepare_data.py
   ```
4. Запустіть тренування
   ```bash
   docker-compose exec finetune python /workspace/scripts/train.py --config /workspace/configs/training_config.yaml
   ```
5. Протестуйте модель
   ```bash
   docker-compose exec finetune python /workspace/scripts/test_model.py
   ```

Замiсть кроків 3–5 можна виконати
```bash
./run_training.sh
```
