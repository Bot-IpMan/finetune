#!/bin/bash

echo "🚀 Запуск файн-тюнінгу Qwen2.5-Coder..."

# Вибір docker-compose файлу
COMPOSE="docker-compose"
if [[ "$1" == "--cpu" ]]; then
    COMPOSE="docker-compose -f docker-compose.cpu.yml"
    echo "ℹ️ Запуск у режимі CPU"
else
    # Перевірка наявності GPU
    if ! nvidia-smi &> /dev/null; then
        echo "❌ GPU не знайдено. Переконайтеся, що NVIDIA драйвери встановлені."
        exit 1
    fi
    echo "✅ GPU доступне"
fi

# Побудова контейнера
echo "🏗️ Побудова Docker образу..."
$COMPOSE build

# Запуск контейнера
echo "🐳 Запуск контейнера..."
$COMPOSE up -d

# Очікування запуску
sleep 5

# Підготовка даних
echo "📊 Підготовка даних..."
$COMPOSE exec finetune python /workspace/scripts/prepare_data.py

# Перевірка підготовлених даних
echo "🔍 Перевірка даних..."
if $COMPOSE exec finetune test -f /workspace/data/processed/train_dataset.jsonl; then
    echo "✅ Тренувальні дані готові"
else
    echo "❌ Помилка підготовки даних"
    exit 1
fi

# Запуск тренування
echo "🎯 Початок тренування..."
$COMPOSE exec finetune python /workspace/scripts/train.py --config /workspace/configs/training_config.yaml

echo "🎉 Тренування завершено!"
echo "📁 Результати збережено в ./models/finetuned/"

# Запуск тестування
echo "🧪 Запуск тестування..."
$COMPOSE exec finetune python /workspace/scripts/test_model.py

echo "✅ Процес завершено успішно!"
