# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Встановлення базових залежностей
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Створення символічного посилання для python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Оновлення pip
RUN python -m pip install --upgrade pip

# Встановлення PyTorch з CUDA підтримкою
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Копіювання файлу залежностей
WORKDIR /workspace
COPY requirements/requirements.txt .

# Встановлення Python залежностей
RUN pip install --no-cache-dir -r requirements.txt

# Створення необхідних директорій
RUN mkdir -p /workspace/data/processed \
             /workspace/models \
             /workspace/logs/tensorboard \
             /workspace/scripts \
             /workspace/configs \
             /workspace/.cache

# Встановлення змінних оточення
ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache
ENV HF_HOME=/workspace/.cache
ENV CUDA_VISIBLE_DEVICES=0

# Копіювання скриптів та конфігурацій
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/

# Встановлення прав доступу
RUN chmod +x /workspace/scripts/*.py

# Створення користувача (опціонально)
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /workspace
USER developer

# Точка входу
CMD ["bash"]