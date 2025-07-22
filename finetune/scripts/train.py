#!/usr/bin/env python3
"""
Виправлений скрипт для файн-тюнінгу Qwen2.5-Coder моделі
Виправлені проблеми з scipy та bitsandbytes
"""

import os
import json
import torch
import logging
import tempfile
import shutil
from pathlib import Path

# Встановлюємо змінні оточення ПЕРЕД імпортом transformers
temp_base = tempfile.mkdtemp(prefix='qwen_train_')
os.environ['HF_HOME'] = temp_base
os.environ['TRANSFORMERS_CACHE'] = temp_base
os.environ['HF_DATASETS_CACHE'] = temp_base
os.environ['HUGGINGFACE_HUB_CACHE'] = temp_base

# Вимикаємо bitsandbytes для CPU
os.environ['DISABLE_BNB'] = '1'
os.environ['BNB_DISABLE_APEX'] = '1'

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Налаштування логування
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeTrainer:
    def __init__(self):
        self.temp_dir = temp_base
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        logger.info(f"Використовую тимчасовий каталог: {self.temp_dir}")
        
        self.config = {
            'model_name': 'Qwen/Qwen2.5-Coder-1.5B',  # Менша модель для CPU
            'max_length': 128,  # Ще більше скорочено для CPU
            'batch_size': 1,
            'learning_rate': 1e-4,
            'epochs': 1,
            'lora_r': 2,  # Мінімальні параметри для CPU
            'lora_alpha': 4,
        }
        
    def create_dummy_dataset(self):
        """Створює мінімальний тестовий датасет"""
        data = [
            {"text": "def hello():\n    print('Hello World')"},
            {"text": "def add(a, b):\n    return a + b"},
            {"text": "class Calculator:\n    def multiply(self, x, y):\n        return x * y"},
            {"text": "import numpy as np\n\ndef process_array(arr):\n    return np.mean(arr)"},
        ]
        return Dataset.from_list(data)
    
    def setup_tokenizer(self):
        """Налаштування токенайзера"""
        logger.info("Завантаження токенайзера...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                trust_remote_code=True,
                cache_dir=self.temp_dir,
                local_files_only=False,
                force_download=False,
                use_fast=True  # Використовуємо швидкий токенайзер
            )
            logger.info(f"Успішно завантажено токенайзер для {self.config['model_name']}")
            
        except Exception as e:
            logger.error(f"Помилка завантаження токенайзера {self.config['model_name']}: {e}")
            # Fallback до CodeT5 або GPT2
            fallback_models = ["Salesforce/codet5-base", "gpt2"]
            
            for fallback in fallback_models:
                try:
                    logger.info(f"Пробую fallback токенайзер: {fallback}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback,
                        cache_dir=self.temp_dir,
                        use_fast=True
                    )
                    self.config['model_name'] = fallback
                    logger.info(f"Успішно завантажено fallback токенайзер: {fallback}")
                    break
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback} також не працює: {fallback_error}")
                    continue
            else:
                raise RuntimeError("Не вдалося завантажити жоден токенайзер")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Встановлено pad_token як eos_token")
            
    def setup_model(self):
        """Налаштування моделі"""
        logger.info("Завантаження моделі...")
        
        try:
            # Спробуємо завантажити оригінальну модель БЕЗ quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.float32,  # Використовуємо float32 для CPU
                device_map=None,  # Не використовуємо device_map для CPU
                trust_remote_code=True,
                cache_dir=self.temp_dir,
                low_cpu_mem_usage=True,
                # Вимикаємо всі quantization опції
                load_in_8bit=False,
                load_in_4bit=False,
                quantization_config=None
            )
            logger.info(f"Успішно завантажено модель: {self.config['model_name']}")
            
        except Exception as e:
            logger.error(f"Помилка завантаження {self.config['model_name']}: {e}")
            # Fallback до меншої доступної моделі
            fallback_models = ["gpt2", "distilgpt2"]
            
            for fallback in fallback_models:
                try:
                    logger.info(f"Пробую fallback модель: {fallback}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback,
                        cache_dir=self.temp_dir,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.config['model_name'] = fallback
                    logger.info(f"Успішно завантажено fallback модель: {fallback}")
                    break
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback} також не працює: {fallback_error}")
                    continue
            else:
                raise RuntimeError("Не вдалося завантажити жодну модель")
        
        # Визначаємо target_modules на основі архітектури моделі
        model_type = self.model.config.model_type.lower()
        logger.info(f"Тип моделі: {model_type}")
        
        if "gpt" in model_type:
            target_modules = ["c_attn", "c_proj"]
        elif "qwen" in model_type:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "codet5" in model_type:
            target_modules = ["q", "v", "k", "o"]
        else:
            # Загальні назви для transformer моделей
            target_modules = ["q_proj", "v_proj"]
            logger.warning(f"Невідомий тип моделі {model_type}, використовую загальні target_modules")
        
        # Налаштування LoRA БЕЗ quantization
        try:
            lora_config = LoraConfig(
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # Вимикаємо всі quantization опції
                use_rslora=False,
                init_lora_weights=True,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Модель успішно налаштована з LoRA")
            
        except Exception as e:
            logger.error(f"Помилка налаштування LoRA: {e}")
            # Якщо LoRA не працює, працюємо без неї
            logger.warning("Працюю без LoRA адаптера")
            # Заморожуємо параметри моделі для економії пам'яті
            for param in self.model.parameters():
                param.requires_grad = False
            # Розморожуємо тільки останні шари
            for param in list(self.model.parameters())[-4:]:
                param.requires_grad = True
        
    def tokenize_data(self, examples):
        """Токенізація даних"""
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config['max_length'],
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    def train(self):
        """Запуск тренування"""
        logger.info("🚀 Початок тренування...")
        
        # Створюємо вихідну директорію
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Налаштування
            self.setup_tokenizer()
            self.setup_model()
            
            # Підготовка даних
            dataset = self.create_dummy_dataset()
            tokenized_dataset = dataset.map(
                self.tokenize_data,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=1  # Один процес для стабільності
            )
            
            logger.info(f"Розмір датасету: {len(tokenized_dataset)}")
            
            # Мінімальні параметри тренування для CPU
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.config['epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                learning_rate=self.config['learning_rate'],
                warmup_steps=1,
                logging_steps=1,
                save_steps=50,
                save_total_limit=1,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                report_to=[],  # Без звітності
                logging_dir=None,
                disable_tqdm=False,
                # CPU оптимізації
                dataloader_pin_memory=False,
                gradient_checkpointing=False,  # Вимикаємо для стабільності
                fp16=False,  # Вимикаємо FP16 для CPU
                bf16=False,  # Вимикаємо BF16 для CPU
                optim="adamw_torch",  # Використовуємо стандартний optimizer
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=None,  # Вимикаємо для простоти
            )
            
            # Тренер
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Тренування
            logger.info("Початок тренування...")
            result = trainer.train()
            
            # Збереження
            logger.info("💾 Збереження моделі...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Збереження результатів
            with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
                json.dump(result.metrics, f, indent=2)
            
            # Збереження конфігурації
            with open(os.path.join(self.output_dir, 'training_config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"✅ Тренування завершено! Модель збережена в {self.output_dir}")
            logger.info(f"📊 Втрати: {result.training_loss:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Помилка під час тренування: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Очищення пам'яті
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def install_missing_packages():
    """Встановлює відсутні пакети"""
    import subprocess
    import sys
    
    required_packages = [
        'scipy',  # Основна проблема
        'datasets',
        'peft',
        'accelerate'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} вже встановлено")
        except ImportError:
            logger.info(f"📦 Встановлюю {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ {package} успішно встановлено")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Помилка встановлення {package}: {e}")

def main():
    """Головна функція"""
    try:
        # Спочатку встановлюємо відсутні пакети
        logger.info("🔍 Перевіряю та встановлюю необхідні пакети...")
        install_missing_packages()
        
        # Показуємо інформацію про середовище
        logger.info(f"Python: {torch.__version__}")
        logger.info(f"Transformers: {transformers.__version__}")
        logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
        logger.info(f"Пристрій: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        trainer = CodeTrainer()
        result = trainer.train()
        
        print("🎉 Тренування успішно завершено!")
        print(f"📁 Результати збережено в: {trainer.output_dir}")
        
    except KeyboardInterrupt:
        print("❌ Тренування перервано користувачем")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")
        import traceback
        traceback.print_exc()
        
        # Пропонуємо рішення
        print("\n🔧 Можливі рішення:")
        print("1. Встановіть scipy: pip install scipy")
        print("2. Встановіть всі залежності: pip install scipy datasets peft accelerate")
        print("3. Якщо проблеми з bitsandbytes: pip uninstall bitsandbytes")
        print("4. Використовуйте віртуальне середовище з Python 3.8+")

if __name__ == "__main__":
    main()