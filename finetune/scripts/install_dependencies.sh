#!/bin/bash

echo "🔧 Встановлення залежностей для файн-тюнінгу моделі..."

# Оновлюємо pip
echo "📦 Оновлюю pip..."
python3 -m pip install --upgrade pip

# Основні залежності
echo "📦 Встановлюю основні залежності..."
pip install scipy numpy

# Machine Learning бібліотеки
echo "🤖 Встановлюю ML бібліотеки..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hugging Face екосистема
echo "🤗 Встановлюю Hugging Face бібліотеки..."
pip install transformers datasets tokenizers

# PEFT для LoRA
echo "🔧 Встановлюю PEFT..."
pip install peft accelerate

# Видаляємо проблемний bitsandbytes якщо він є
echo "🗑️ Видаляю bitsandbytes (може викликати проблеми на CPU)..."
pip uninstall bitsandbytes -y 2>/dev/null || true

# Додаткові утиліти
echo "🛠️ Встановлюю додаткові утиліти..."
pip install evaluate tensorboard wandb

echo "✅ Всі залежності встановлено!"
echo ""
echo "🚀 Тепер можете запустити тренування:"
echo "python3 train.py"