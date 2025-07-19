# Finetune Project

## Структура проекту

Детально описана у `scripts/init_finetune_project.sh`.

## Швидкий старт (GPU)
```bash
docker-compose build
docker-compose up -d
```

Скрипт `run_training.sh` збирає образ, готує дані та запускає тренування:
```bash
bash run_training.sh
```

## Запуск без GPU
Для машин без відеокарти передбачено окремий Dockerfile та compose-файл.
Зберіть та запустіть контейнер так:
```bash
docker-compose -f docker-compose.cpu.yml build
docker-compose -f docker-compose.cpu.yml up -d
bash run_training.sh --cpu
```

