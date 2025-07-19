#!/bin/bash

echo "🚀 Запуск файн-тюнінгу Qwen2.5-Coder..."

# Перевірка наявності GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ GPU не знайдено. Переконайтеся, що NVIDIA драйвери встановлені."
    exit 1
fi

echo "✅ GPU доступне"

# Побудова контейнера
echo "🏗️ Побудова Docker образу..."
docker-compose build

# Запуск контейнера
echo "🐳 Запуск контейнера..."
docker-compose up -d

# Очікування запуску
sleep 5

# Підготовка даних
echo "📊 Підготовка даних..."
docker-compose exec finetune python /workspace/scripts/prepare_data.py

# Перевірка підготовлених даних
echo "🔍 Перевірка даних..."
if docker-compose exec finetune test -f /workspace/data/processed/train_dataset.jsonl; then
    echo "✅ Тренувальні дані готові"
else
    echo "❌ Помилка підготовки даних"
    exit 1
fi

# Запуск тренування
echo "🎯 Початок тренування..."
docker-compose exec finetune python /workspace/scripts/train.py --config /workspace/configs/training_config.yaml

echo "🎉 Тренування завершено!"
echo "📁 Результати збережено в ./models/finetuned/"

# Запуск тестування
echo "🧪 Запуск тестування..."
docker-compose exec finetune python /workspace/scripts/test_model.py

echo "✅ Процес завершено успішно!"