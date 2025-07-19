#!/usr/bin/env python3
"""
Скрипт для тестування файн-тюненої моделі
"""

import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ModelTester:
    """Клас для тестування файн-тюненої моделі"""
    
    def __init__(self, model_path: str = "/workspace/models/finetuned"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.base_model = None
        
    def load_model(self):
        """Завантажує файн-тюнену модель"""
        print("🚀 Завантаження файн-тюненої моделі...")
        
        try:
            # Завантажуємо токенайзер
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Встановлюємо pad_token якщо потрібно
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Завантажуємо базову модель
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Завантажуємо LoRA адаптер
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            print("✅ Модель завантажена успішно!")
            return True
            
        except Exception as e:
            print(f"❌ Помилка завантаження моделі: {e}")
            return False
    
    def load_base_model(self):
        """Завантажує базову модель для порівняння"""
        print("🔄 Завантаження базової моделі для порівняння...")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Базова модель завантажена!")
            return True
        except Exception as e:
            print(f"❌ Помилка завантаження базової моделі: {e}")
            return False
    
    def generate_code(self, instruction: str, input_text: str = "", model=None, max_length: int = 256):
        """Генерує код за інструкцією"""
        if model is None:
            model = self.model
            
        # Формуємо промпт
        if input_text and input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"
        
        # Токенізація
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Генерація
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        generation_time = time.time() - start_time
        
        # Декодування
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Витягуємо тільки згенерований код
        if "Output:" in generated_text:
            output = generated_text.split("Output:")[-1].strip()
        else:
            output = generated_text.replace(prompt, "").strip()
        
        return {
            "output": output,
            "generation_time": generation_time,
            "prompt_length": len(inputs["input_ids"][0]),
            "output_length": len(outputs[0]) - len(inputs["input_ids"][0])
        }
    
    def run_test_suite(self):
        """Запускає набір тестів"""
        print("\n🧪 Запуск тестового набору...")
        print("=" * 60)
        
        test_cases = [
            {
                "name": "Базова функція додавання",
                "instruction": "Напиши функцію для додавання двох чисел",
                "input": "Python",
                "expected_keywords": ["def", "return", "+"]
            },
            {
                "name": "Функція множення",
                "instruction": "Створи функцію для множення двох чисел",
                "input": "Python",
                "expected_keywords": ["def", "return", "*"]
            },
            {
                "name": "Функція факторіалу",
                "instruction": "Напиши функцію для обчислення факторіалу",
                "input": "",
                "expected_keywords": ["def", "factorial", "return"]
            },
            {
                "name": "Клас банківський рахунок",
                "instruction": "Створи клас для роботи з банківським рахунком",
                "input": "Python",
                "expected_keywords": ["class", "def", "self"]
            },
            {
                "name": "Гідропонний моніторинг",
                "instruction": "Напиши функцію для розрахунку pH рівня для гідропоніки",
                "input": "Python",
                "expected_keywords": ["def", "ph", "return"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔹 Тест {i}: {test_case['name']}")
            print(f"   Інструкція: {test_case['instruction']}")
            print(f"   Вхід: {test_case['input']}")
            
            # Тестуємо файн-тюнену модель
            finetuned_result = self.generate_code(
                test_case['instruction'], 
                test_case['input']
            )
            
            # Оцінюємо якість
            quality_score = self.evaluate_output(
                finetuned_result['output'], 
                test_case['expected_keywords']
            )
            
            print(f"   ⏱️  Час генерації: {finetuned_result['generation_time']:.2f}s")
            print(f"   📏 Довжина виходу: {finetuned_result['output_length']} токенів")
            print(f"   ⭐ Оцінка якості: {quality_score}/10")
            print(f"   📝 Результат:\n{finetuned_result['output'][:200]}...")
            
            # Порівняння з базовою моделлю (якщо доступна)
            if self.base_model is not None:
                base_result = self.generate_code(
                    test_case['instruction'], 
                    test_case['input'],
                    model=self.base_model
                )
                base_quality = self.evaluate_output(
                    base_result['output'], 
                    test_case['expected_keywords']
                )
                
                improvement = quality_score - base_quality
                print(f"   📊 Порівняння з базовою: {improvement:+.1f} балів")
            
            results.append({
                "test_name": test_case['name'],
                "instruction": test_case['instruction'],
                "input": test_case['input'],
                "output": finetuned_result['output'],
                "quality_score": quality_score,
                "generation_time": finetuned_result['generation_time'],
                "output_length": finetuned_result['output_length']
            })
            
            print("-" * 40)
        
        return results
    
    def evaluate_output(self, output: str, expected_keywords: List[str]) -> float:
        """Оцінює якість згенерованого коду"""
        score = 0.0
        output_lower = output.lower()
        
        # Перевірка наявності очікуваних ключових слів (40% оцінки)
        keyword_score = sum(1 for keyword in expected_keywords if keyword.lower() in output_lower)
        score += (keyword_score / len(expected_keywords)) * 4.0
        
        # Перевірка синтаксичної структури (30% оцінки)
        if "def " in output_lower:
            score += 1.0
        if "return" in output_lower:
            score += 1.0
        if output.count("(") == output.count(")"):  # Збалансовані дужки
            score += 1.0
        
        # Перевірка наявності коментарів або docstring (20% оцінки)
        if '"""' in output or "'''" in output or "#" in output:
            score += 2.0
        
        # Перевірка довжини та структури (10% оцінки)
        if 50 <= len(output) <= 500:  # Оптимальна довжина
            score += 1.0
        
        return min(score, 10.0)  # Максимум 10 балів
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Генерує звіт про результати тестування"""
        print("\n📊 ЗВІТ ПРО ТЕСТУВАННЯ")
        print("=" * 60)
        
        # Загальна статистика
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        avg_length = sum(r['output_length'] for r in results) / len(results)
        
        print(f"📈 Середня оцінка якості: {avg_quality:.1f}/10")
        print(f"⏱️  Середній час генерації: {avg_time:.2f}s")
        print(f"📏 Середня довжина відповіді: {avg_length:.0f} токенів")
        
        # Розподіл оцінок
        excellent = sum(1 for r in results if r['quality_score'] >= 8)
        good = sum(1 for r in results if 6 <= r['quality_score'] < 8)
        fair = sum(1 for r in results if 4 <= r['quality_score'] < 6)
        poor = sum(1 for r in results if r['quality_score'] < 4)
        
        print(f"\n🏆 Розподіл результатів:")
        print(f"   Відмінно (8-10): {excellent}/{len(results)} тестів")
        print(f"   Добре (6-8): {good}/{len(results)} тестів")
        print(f"   Задовільно (4-6): {fair}/{len(results)} тестів")
        print(f"   Погано (0-4): {poor}/{len(results)} тестів")
        
        # Збереження детального звіту
        report_path = Path("/workspace/models/finetuned/test_results.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "avg_quality": avg_quality,
                    "avg_time": avg_time,
                    "avg_length": avg_length,
                    "total_tests": len(results)
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Детальний звіт збережено: {report_path}")
        
        return avg_quality

def main():
    """Головна функція"""
    tester = ModelTester()
    
    # Завантаження моделі
    if not tester.load_model():
        print("❌ Не вдалося завантажити модель. Переконайтеся, що тренування завершено.")
        return
    
    # Завантаження базової моделі для порівняння (опціонально)
    print("\n🤔 Завантажити базову модель для порівняння? (може зайняти додатковий час)")
    load_base = input("Введіть 'y' щоб завантажити (або натисніть Enter для пропуску): ").strip().lower()
    if load_base == 'y':
        tester.load_base_model()
    else:
        print("ℹ️  Базову модель не завантажено")

    results = tester.run_test_suite()
    tester.generate_report(results)

if __name__ == "__main__":
    main()
