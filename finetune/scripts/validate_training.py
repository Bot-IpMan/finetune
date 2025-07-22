#!/usr/bin/env python3
"""
Скрипт для перевірки успішності навчання Qwen2.5-Coder моделі
"""

import os
import json
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingValidator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.results = {}
        
    def check_files_exist(self):
        """Перевіряє наявність необхідних файлів"""
        logger.info("🔍 Перевіряю наявність файлів моделі...")
        
        required_files = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'tokenizer_config.json',
            'tokenizer.json',
            'results.json',
            'training_config.json'
        ]
        
        missing_files = []
        existing_files = []
        
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if os.path.exists(file_path):
                existing_files.append(file)
                logger.info(f"✅ {file} знайдено")
            else:
                missing_files.append(file)
                logger.warning(f"⚠️ {file} відсутній")
        
        self.results['files_check'] = {
            'existing_files': existing_files,
            'missing_files': missing_files,
            'all_files_present': len(missing_files) == 0
        }
        
        return len(missing_files) == 0
    
    def check_training_results(self):
        """Перевіряє результати навчання"""
        logger.info("📊 Аналізую результати навчання...")
        
        results_file = os.path.join(self.model_path, 'results.json')
        config_file = os.path.join(self.model_path, 'training_config.json')
        
        if not os.path.exists(results_file):
            logger.error("❌ Файл results.json не знайдено")
            return False
            
        try:
            with open(results_file, 'r') as f:
                training_results = json.load(f)
                
            with open(config_file, 'r') as f:
                training_config = json.load(f)
                
            # Перевіряємо ключові метрики
            train_loss = training_results.get('train_loss', float('inf'))
            train_steps = training_results.get('train_steps_per_second', 0)
            
            logger.info(f"📈 Втрати навчання: {train_loss:.4f}")
            logger.info(f"⚡ Кроків за секунду: {train_steps:.2f}")
            
            # Критерії успішності
            success_criteria = {
                'loss_reasonable': train_loss < 10.0,  # Loss не повинен бути занадто високим
                'loss_not_nan': not np.isnan(train_loss) and not np.isinf(train_loss),
                'steps_completed': train_steps > 0
            }
            
            self.results['training_metrics'] = {
                'train_loss': train_loss,
                'train_steps_per_second': train_steps,
                'success_criteria': success_criteria,
                'training_successful': all(success_criteria.values())
            }
            
            # Виводимо результати
            if all(success_criteria.values()):
                logger.info("✅ Навчання пройшло успішно!")
            else:
                logger.warning("⚠️ Можливі проблеми з навчанням:")
                for criterion, passed in success_criteria.items():
                    if not passed:
                        logger.warning(f"   - {criterion}: FAILED")
            
            return all(success_criteria.values())
            
        except Exception as e:
            logger.error(f"❌ Помилка читання результатів: {e}")
            return False
    
    def test_model_loading(self):
        """Тестує завантаження навченої моделі"""
        logger.info("🔄 Тестую завантаження навченої моделі...")
        
        try:
            # Завантажуємо токенайзер
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("✅ Токенайзер завантажено успішно")
            
            # Завантажуємо базову модель
            config_file = os.path.join(self.model_path, 'training_config.json')
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            base_model_name = config.get('model_name', 'gpt2')
            
            # Завантажуємо модель з адаптером
            try:
                # Спробуємо завантажити як PEFT модель
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(base_model, self.model_path)
                logger.info("✅ PEFT модель завантажено успішно")
                peft_model = True
                
            except Exception as peft_error:
                logger.warning(f"PEFT модель не завантажилась: {peft_error}")
                # Спробуємо завантажити як звичайну модель
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                logger.info("✅ Звичайна модель завантажено успішно")
                peft_model = False
            
            self.results['model_loading'] = {
                'tokenizer_loaded': True,
                'model_loaded': True,
                'is_peft_model': peft_model,
                'base_model': base_model_name
            }
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Помилка завантаження моделі: {e}")
            self.results['model_loading'] = {
                'tokenizer_loaded': False,
                'model_loaded': False,
                'error': str(e)
            }
            return None, None
    
    def test_model_inference(self, model, tokenizer):
        """Тестує генерацію тексту навченою моделлю"""
        logger.info("🧪 Тестую генерацію тексту...")
        
        test_prompts = [
            "def hello():",
            "class Calculator:",
            "import numpy as np\n\ndef process_data(",
            "# This function calculates"
        ]
        
        generation_results = []
        
        try:
            model.eval()
            
            for i, prompt in enumerate(test_prompts):
                logger.info(f"Тест {i+1}: '{prompt}'")
                
                # Токенізуємо вхід
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                
                # Генеруємо текст
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Декодуємо результат
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_text = generated[len(prompt):].strip()
                
                result = {
                    'prompt': prompt,
                    'generated': new_text,
                    'full_output': generated,
                    'success': len(new_text) > 0
                }
                
                generation_results.append(result)
                logger.info(f"✅ Згенеровано: '{new_text[:50]}...'")
            
            successful_generations = sum(1 for r in generation_results if r['success'])
            
            self.results['inference_test'] = {
                'total_tests': len(test_prompts),
                'successful_generations': successful_generations,
                'success_rate': successful_generations / len(test_prompts),
                'results': generation_results,
                'inference_working': successful_generations > 0
            }
            
            logger.info(f"📊 Успішних генерацій: {successful_generations}/{len(test_prompts)}")
            
            return successful_generations > 0
            
        except Exception as e:
            logger.error(f"❌ Помилка тестування генерації: {e}")
            self.results['inference_test'] = {
                'error': str(e),
                'inference_working': False
            }
            return False
    
    def check_model_size(self):
        """Перевіряє розмір моделі та адаптерів"""
        logger.info("📏 Перевіряю розмір файлів моделі...")
        
        file_sizes = {}
        total_size = 0
        
        for file in os.listdir(self.model_path):
            file_path = os.path.join(self.model_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                file_sizes[file] = size
                total_size += size
                
                size_mb = size / (1024 * 1024)
                logger.info(f"📄 {file}: {size_mb:.2f} MB")
        
        total_mb = total_size / (1024 * 1024)
        logger.info(f"📊 Загальний розмір: {total_mb:.2f} MB")
        
        self.results['model_size'] = {
            'file_sizes': file_sizes,
            'total_size_bytes': total_size,
            'total_size_mb': total_mb,
            'reasonable_size': total_mb > 0.1  # Принаймні 100KB
        }
        
        return total_mb > 0.1
    
    def run_full_validation(self):
        """Запускає повну валідацію"""
        logger.info("🚀 Початок повної валідації навченої моделі...")
        
        validation_steps = [
            ("Перевірка файлів", self.check_files_exist),
            ("Аналіз результатів навчання", self.check_training_results),
            ("Перевірка розміру моделі", self.check_model_size),
        ]
        
        success_count = 0
        
        for step_name, step_func in validation_steps:
            logger.info(f"\n--- {step_name} ---")
            try:
                success = step_func()
                if success:
                    success_count += 1
                    logger.info(f"✅ {step_name}: ПРОЙДЕНО")
                else:
                    logger.warning(f"⚠️ {step_name}: ПРОБЛЕМИ")
            except Exception as e:
                logger.error(f"❌ {step_name}: ПОМИЛКА - {e}")
        
        # Тестуємо завантаження та інференс
        logger.info(f"\n--- Тестування моделі ---")
        model, tokenizer = self.test_model_loading()
        
        if model is not None and tokenizer is not None:
            success_count += 1
            inference_success = self.test_model_inference(model, tokenizer)
            if inference_success:
                success_count += 1
        
        # Підсумки
        total_steps = len(validation_steps) + 2  # +2 для завантаження та інференсу
        success_rate = success_count / total_steps
        
        self.results['overall'] = {
            'total_validation_steps': total_steps,
            'successful_steps': success_count,
            'success_rate': success_rate,
            'validation_passed': success_rate >= 0.8  # 80% успіху
        }
        
        logger.info(f"\n🎯 ПІДСУМКИ ВАЛІДАЦІЇ:")
        logger.info(f"   Успішних кроків: {success_count}/{total_steps}")
        logger.info(f"   Рівень успіху: {success_rate*100:.1f}%")
        
        if success_rate >= 0.8:
            logger.info("🎉 НАВЧАННЯ ПРОЙШЛО УСПІШНО!")
        else:
            logger.warning("⚠️ ВИЯВЛЕНО ПРОБЛЕМИ З НАВЧАННЯМ")
        
        # Зберігаємо результати валідації
        validation_results_path = os.path.join(self.model_path, 'validation_results.json')
        with open(validation_results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"📁 Результати валідації збережено в: {validation_results_path}")
        
        return success_rate >= 0.8

def main():
    """Головна функція"""
    import sys
    
    if len(sys.argv) != 2:
        print("Використання: python validate_training.py <шлях_до_навченої_моделі>")
        print("Приклад: python validate_training.py ./output")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Директорія {model_path} не існує")
        sys.exit(1)
    
    validator = TrainingValidator(model_path)
    success = validator.run_full_validation()
    
    if success:
        print("\n✅ Навчання пройшло успішно! Модель готова до використання.")
        sys.exit(0)
    else:
        print("\n❌ Виявлено проблеми з навчанням. Перевірте логи вище.")
        sys.exit(1)

if __name__ == "__main__":
    main()