#!/usr/bin/env python3
"""
Скрипт для файн-тюнінгу Qwen2.5-Coder моделі
"""

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
import numpy as np

# Налаштування логування
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Аргументи для моделі"""
    model_name: str = field(default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    torch_dtype: str = field(default="float16")
    device_map: str = field(default="auto")
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=True)

@dataclass
class LoraArguments:
    """Аргументи для LoRA"""
    r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    bias: str = field(default="none")
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class DataArguments:
    """Аргументи для даних"""
    train_file: str = field(default="/workspace/data/processed/train_dataset.jsonl")
    eval_file: str = field(default="/workspace/data/processed/eval_dataset.jsonl")
    max_length: int = field(default=512)
    instruction_template: str = field(
        default="Instruction: {instruction}\nInput: {input}\nOutput: {output}"
    )

class CodeDataset:
    """Клас для роботи з датасетом коду"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_data(self, file_path: str) -> Dataset:
        """Завантажує дані з JSONL файлу"""
        try:
            dataset = load_dataset('json', data_files=file_path)['train']
            logger.info(f"Завантажено {len(dataset)} зразків з {file_path}")
            return dataset
        except Exception as e:
            logger.error(f"Помилка завантаження даних: {e}")
            raise
    
    def preprocess_function(self, examples):
        """Попередня обробка даних"""
        inputs = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples.get('input', [''] * len(examples['instruction']))[i]
            output = examples['output'][i]
            
            # Формуємо промпт
            if input_text and input_text.strip():
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                prompt = f"Instruction: {instruction}\nOutput: {output}"
            
            inputs.append(prompt)
        
        # Токенізація
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Встановлюємо labels = input_ids для causal LM
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

class ModelTrainer:
    """Клас для тренування моделі"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Завантажує конфігурацію"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_tokenizer(self):
        """Налаштовує токенайзер"""
        logger.info("Завантаження токенайзера...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Додаємо pad_token якщо його немає
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Токенайзер завантажено. Vocab size: {len(self.tokenizer)}")
        
    def setup_model(self):
        """Налаштовує модель"""
        logger.info("Завантаження базової моделі...")
        
        # Налаштування завантаження
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, self.config['model']['torch_dtype']),
            "device_map": self.config['model']['device_map'],
            "low_cpu_mem_usage": True,
        }
        
        if self.config['model']['load_in_4bit']:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = bnb_config
        
        # Завантажуємо модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            **model_kwargs
        )
        
        # Налаштовуємо LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Resize token embeddings якщо потрібно
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Модель налаштована з LoRA адаптерами")
    
    def prepare_datasets(self):
        """Підготовка датасетів"""
        logger.info("Підготовка датасетів...")
        
        dataset_processor = CodeDataset(self.tokenizer, self.config['data']['max_length'])
        
        # Завантажуємо дані
        train_dataset = dataset_processor.load_data(self.config['data']['train_file'])
        eval_dataset = dataset_processor.load_data(self.config['data']['eval_file'])
        
        # Обробляємо дані
        train_dataset = train_dataset.map(
            dataset_processor.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            dataset_processor.preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} зразків")
        logger.info(f"Eval dataset: {len(eval_dataset)} зразків")
        
        return train_dataset, eval_dataset
    
    def setup_trainer(self, train_dataset, eval_dataset):
        """Налаштування тренера"""
        logger.info("Налаштування тренера...")
        
        # Параметри тренування
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            optim=self.config['training']['optim'],
            fp16=self.config['training']['fp16'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            report_to=self.config['training']['report_to'],
            logging_dir=self.config['logging']['tensorboard_dir']
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, не masked LM
        )
        
        # Ініціалізуємо тренер
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Тренер налаштований")
    
    def train(self):
        """Запускає тренування"""
        logger.info("🚀 Початок тренування...")
        
        # Створюємо директорії
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['tensorboard_dir'], exist_ok=True)
        
        # Налаштування
        self.setup_tokenizer()
        self.setup_model()
        train_dataset, eval_dataset = self.prepare_datasets()
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Тренування
        result = self.trainer.train()
        
        # Збереження моделі
        logger.info("💾 Збереження моделі...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])
        
        # Збереження метрик
        with open(f"{self.config['training']['output_dir']}/training_results.json", 'w') as f:
            json.dump(result.metrics, f, indent=2)
        
        logger.info("✅ Тренування завершено!")
        return result

def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Файн-тюнінг Qwen2.5-Coder")
    parser.add_argument(
        "--config",
        default="/workspace/configs/training_config.yaml",
        help="Шлях до конфігураційного файлу"
    )
    
    args = parser.parse_args()
    
    # Запускаємо тренування
    trainer = ModelTrainer(args.config)
    result = trainer.train()
    
    print(f"🎉 Тренування завершено! Результати збережено в {trainer.config['training']['output_dir']}")

if __name__ == "__main__":
    main()