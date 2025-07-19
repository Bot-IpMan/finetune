# prepare_data.py
#!/usr/bin/env python3
"""
Скрипт для підготовки тренувальних даних
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import markdown
from bs4 import BeautifulSoup
import re

class DataProcessor:
    """Клас для обробки та підготовки даних"""
    
    def __init__(self, data_dir: str = "/workspace/data", output_dir: str = "/workspace/data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_markdown(self, file_path: Path) -> str:
        """Витягує текст з markdown файлу"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Конвертуємо markdown в HTML, потім в чистий текст
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def generate_code_samples(self) -> List[Dict[str, str]]:
        """Генерує зразки коду для тренування"""
        samples = []
        
        # Базові зразки програмування
        basic_samples = [
            {
                "instruction": "Напиши функцію для додавання двох чисел",
                "input": "Python",
                "output": """def add_numbers(a, b):
    \"\"\"Додає два числа та повертає результат\"\"\"
    return a + b

# Приклад використання:
result = add_numbers(5, 3)
print(f"5 + 3 = {result}")"""
            },
            {
                "instruction": "Створи функцію для множення двох чисел",
                "input": "Python",
                "output": """def multiply(a, b):
    \"\"\"Множить два числа\"\"\"
    return a * b

# Приклад використання:
result = multiply(4, 6)
print(f"4 * 6 = {result}")"""
            },
            {
                "instruction": "Напиши функцію для обчислення факторіалу",
                "input": "",
                "output": """def factorial(n):
    \"\"\"Обчислює факторіал числа n\"\"\"
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Альтернативна ітеративна версія:
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result"""
            },
            {
                "instruction": "Створи клас для роботи з банківським рахунком",
                "input": "Python",
                "output": """class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self.balance = initial_balance
        self.transaction_history = []
    
    def deposit(self, amount):
        \"\"\"Поповнення рахунку\"\"\"
        if amount > 0:
            self.balance += amount
            self.transaction_history.append(f"Поповнення: +{amount}")
            return True
        return False
    
    def withdraw(self, amount):
        \"\"\"Зняття коштів\"\"\"
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            self.transaction_history.append(f"Зняття: -{amount}")
            return True
        return False
    
    def get_balance(self):
        return self.balance"""
            },
            {
                "instruction": "Напиши функцію для сортування списку",
                "input": "bubble sort",
                "output": """def bubble_sort(arr):
    \"\"\"Сортування бульбашкою\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Приклад використання:
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers.copy())
print(f"Відсортований список: {sorted_numbers}")"""
            },
            {
                "instruction": "Створи функцію для перевірки паліндрому",
                "input": "Python",
                "output": """def is_palindrome(text):
    \"\"\"Перевіряє чи є рядок паліндромом\"\"\"
    # Видаляємо пробіли та приводимо до нижнього регістру
    cleaned = ''.join(text.split()).lower()
    return cleaned == cleaned[::-1]

# Альтернативна версія з рекурсією:
def is_palindrome_recursive(s, start=0, end=None):
    if end is None:
        end = len(s) - 1
    
    if start >= end:
        return True
    
    if s[start] != s[end]:
        return False
    
    return is_palindrome_recursive(s, start + 1, end - 1)"""
            }
        ]
        
        samples.extend(basic_samples)
        
        # Зразки для гідропоніки (з ваших MD файлів)
        hydroponic_samples = [
            {
                "instruction": "Напиши функцію для розрахунку pH рівня для гідропонного вирощування",
                "input": "Python",
                "output": """def calculate_ph_level(current_ph, target_ph=6.0):
    \"\"\"Розраховує необхідну корекцію pH для гідропоніки\"\"\"
    if current_ph < target_ph:
        adjustment = "Додати pH UP (лужний розчин)"
        difference = target_ph - current_ph
    elif current_ph > target_ph:
        adjustment = "Додати pH DOWN (кислотний розчин)"
        difference = current_ph - target_ph
    else:
        adjustment = "pH в нормі"
        difference = 0
    
    return {
        "current_ph": current_ph,
        "target_ph": target_ph,
        "difference": difference,
        "adjustment": adjustment
    }"""
            },
            {
                "instruction": "Створи клас для моніторингу гідропонної системи",
                "input": "Python",
                "output": """class HydroponicMonitor:
    def __init__(self, system_name):
        self.system_name = system_name
        self.ph_level = 6.0
        self.ec_level = 1.2  # Electrical conductivity
        self.water_temperature = 20.0
        self.nutrient_level = 100.0
    
    def update_readings(self, ph=None, ec=None, temp=None, nutrients=None):
        \"\"\"Оновлює показники системи\"\"\"
        if ph is not None:
            self.ph_level = ph
        if ec is not None:
            self.ec_level = ec
        if temp is not None:
            self.water_temperature = temp
        if nutrients is not None:
            self.nutrient_level = nutrients
    
    def check_system_health(self):
        \"\"\"Перевіряє стан системи\"\"\"
        issues = []
        
        if self.ph_level < 5.5 or self.ph_level > 6.5:
            issues.append(f"pH поза нормою: {self.ph_level}")
        
        if self.ec_level < 0.8 or self.ec_level > 2.0:
            issues.append(f"EC поза нормою: {self.ec_level}")
        
        if self.water_temperature < 18 or self.water_temperature > 24:
            issues.append(f"Температура поза нормою: {self.water_temperature}°C")
        
        if self.nutrient_level < 20:
            issues.append(f"Низький рівень поживних речовин: {self.nutrient_level}%")
        
        return issues if issues else ["Система в нормі"]"""
            }
        ]
        
        samples.extend(hydroponic_samples)
        return samples
    
    def create_datasets(self, train_ratio: float = 0.8):
        """Створює тренувальний та валідаційний датасети"""
        print("🔄 Генерація зразків даних...")
        
        # Генеруємо зразки
        samples = self.generate_code_samples()
        
        # Перемішуємо дані
        random.shuffle(samples)
        
        # Розділяємо на тренувальні та валідаційні
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]
        
        print(f"📊 Створено {len(train_samples)} тренувальних зразків")
        print(f"📊 Створено {len(eval_samples)} валідаційних зразків")
        
        # Збереження тренувальних даних
        train_file = self.output_dir / "train_dataset.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Збереження валідаційних даних
        eval_file = self.output_dir / "eval_dataset.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"💾 Дані збережено в {self.output_dir}")
        
        # Створюємо файл з статистикою
        stats = {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "train_ratio": train_ratio,
            "categories": self.get_sample_categories(samples)
        }
        
        with open(self.output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return train_file, eval_file, stats
    
    def get_sample_categories(self, samples: List[Dict]) -> Dict[str, int]:
        """Аналізує категорії зразків"""
        categories = {}
        for sample in samples:
            instruction = sample['instruction'].lower()
            
            if 'функція' in instruction or 'function' in instruction:
                categories['functions'] = categories.get('functions', 0) + 1
            elif 'клас' in instruction or 'class' in instruction:
                categories['classes'] = categories.get('classes', 0) + 1
            elif 'гідропон' in instruction or 'hydroponic' in instruction:
                categories['hydroponic'] = categories.get('hydroponic', 0) + 1
            else:
                categories['other'] = categories.get('other', 0) + 1
        
        return categories
    
    def validate_datasets(self):
        """Валідує створені датасети"""
        print("✅ Валідація датасетів...")
        
        train_file = self.output_dir / "train_dataset.jsonl"
        eval_file = self.output_dir / "eval_dataset.jsonl"
        
        # Перевіряємо тренувальні дані
        with open(train_file, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
        
        # Перевіряємо валідаційні дані
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_lines = f.readlines()
        
        print(f"📝 Тренувальний файл: {len(train_lines)} рядків")
        print(f"📝 Валідаційний файл: {len(eval_lines)} рядків")
        
        # Перевіряємо формат першого зразка
        try:
            sample = json.loads(train_lines[0])
            required_keys = ['instruction', 'input', 'output']
            
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Відсутній ключ: {key}")
            
            print("✅ Формат даних валідний")
            
            # Показуємо приклад
            print("\n📄 Приклад тренувального зразка:")
            print(f"Instruction: {sample['instruction']}")
            print(f"Input: {sample['input']}")
            print(f"Output: {sample['output']}")
        except Exception as e:
            print(f"❌ Помилка валідації даних: {e}")
            return False

        return True
