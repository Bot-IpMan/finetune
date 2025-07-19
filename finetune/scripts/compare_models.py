#!/usr/bin/env python3
"""
Скрипт для порівняння різних мовних моделей для генерації коду.
"""

import os
import json
import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Метрики для моделі"""
    name: str
    avg_quality: float = 0.0
    avg_generation_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    syntax_accuracy: float = 0.0
    code_completeness: float = 0.0
    memory_usage_mb: float = 0.0

class ModelComparator:
    """Клас для порівняння моделей"""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.results = []

    def add_model(self, name: str, model_path: str, is_peft: bool = False):
        """Додає модель для порівняння"""
        logger.info(f"🔄 Завантаження моделі: {name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if is_peft:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

            self.models[name] = model
            self.tokenizers[name] = tokenizer

            logger.info(f"✅ Модель {name} завантажена")
            return True

        except Exception as e:
            logger.error(f"❌ Помилка завантаження моделі {name}: {e}")
            return False

    def generate_response(self, model_name: str, instruction: str, input_text: str = "") -> Dict[str, Any]:
        """Генерує відповідь від моделі"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        if input_text and input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(model.device)

        start_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        generation_time = time.time() - start_time
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Output:" in generated_text:
            output = generated_text.split("Output:")[-1].strip()
        else:
            output = generated_text.replace(prompt, "").strip()

        output_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        return {
            "output": output,
            "generation_time": generation_time,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second
        }

    def evaluate_code_quality(self, code: str) -> Dict[str, float]:
        """Оцінює якість коду"""
        scores = {
            'syntax_valid': 0.0,
            'has_function': 0.0,
            'has_docstring': 0.0,
            'has_comments': 0.0,
            'proper_structure': 0.0,
            'code_completeness': 0.0
        }

        try:
            compile(code, '<string>', 'exec')
            scores['syntax_valid'] = 1.0
        except:
            scores['syntax_valid'] = 0.0

        if 'def ' in code:
            scores['has_function'] = 1.0

        if '"""' in code or "'''" in code:
            scores['has_docstring'] = 1.0

        if '#' in code:
            scores['has_comments'] = 1.0

        if code.count('(') == code.count(')') and code.count('{') == code.count('}'):
            scores['proper_structure'] = 1.0

        if len(code.strip()) > 30 and len(code.split('\n')) > 2:
            scores['code_completeness'] = 1.0

        return scores

    def run_benchmark(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Запускає бенчмарк для всіх моделей"""
        logger.info("🏃 Запуск бенчмарку...")

        all_results = {}

        for model_name in self.models.keys():
            logger.info(f"🎯 Тестування моделі: {model_name}")

            model_results = []
            generation_times = []
            tokens_per_second = []
            quality_scores = []

            for i, test_case in enumerate(test_cases):
                result = self.generate_response(
                    model_name,
                    test_case['instruction'],
                    test_case.get('input', '')
                )

                quality = self.evaluate_code_quality(result['output'])
                avg_quality = np.mean(list(quality.values()))

                model_results.append({
                    'test_case': i,
                    'instruction': test_case['instruction'],
                    'generated_code': result['output'],
                    'generation_time': result['generation_time'],
                    'tokens_per_second': result['tokens_per_second'],
                    'quality_scores': quality,
                    'avg_quality': avg_quality
                })

                generation_times.append(result['generation_time'])
                tokens_per_second.append(result['tokens_per_second'])
                quality_scores.append(avg_quality)

                if (i + 1) % 5 == 0 or (i + 1) == len(test_cases):
                    logger.info(f"  Завершено {i + 1}/{len(test_cases)} тестів")

            metrics = ModelMetrics(
                name=model_name,
                avg_quality=np.mean(quality_scores),
                avg_generation_time=np.mean(generation_times),
                avg_tokens_per_second=np.mean(tokens_per_second),
                syntax_accuracy=np.mean([r['quality_scores']['syntax_valid'] for r in model_results]),
                code_completeness=np.mean([r['quality_scores']['code_completeness'] for r in model_results]),
                memory_usage_mb=self.get_model_memory_usage(model_name)
            )

            all_results[model_name] = {
                'metrics': metrics,
                'detailed_results': model_results
            }

            logger.info(f"✅ Модель {model_name} - Середня якість: {metrics.avg_quality:.3f}")

        return all_results

    def get_model_memory_usage(self, model_name: str) -> float:
        """Оцінює використання пам'яті моделлю"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            return memory_allocated
        return 0.0

    def generate_comparison_table(self, results: Dict[str, Any]) -> str:
        """Генерує таблицю порівняння"""
        table_data = []

        for model_name, result in results.items():
            metrics = result['metrics']
            table_data.append([
                metrics.name,
                f"{metrics.avg_quality:.3f}",
                f"{metrics.avg_generation_time:.2f}s",
                f"{metrics.avg_tokens_per_second:.1f}",
                f"{metrics.syntax_accuracy:.3f}",
                f"{metrics.code_completeness:.3f}",
                f"{metrics.memory_usage_mb:.1f}MB"
            ])

        headers = [
            "Модель", "Якість", "Час ген.", "Ток/сек",
            "Синтаксис", "Повнота", "Пам'ять"
        ]

        return tabulate(table_data, headers=headers, tablefmt="grid")

    def create_comparison_plots(self, results: Dict[str, Any], output_dir: Path):
        """Створює графіки порівняння"""
        logger.info("📊 Створення графіків...")

        model_names = list(results.keys())
        metrics_data = {
            'Quality': [results[name]['metrics'].avg_quality for name in model_names],
            'Speed (tok/s)': [results[name]['metrics'].avg_tokens_per_second for name in model_names],
            'Syntax Accuracy': [results[name]['metrics'].syntax_accuracy for name in model_names],
            'Completeness': [results[name]['metrics'].code_completeness for name in model_names]
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Порівняння моделей', fontsize=16)

        axes[0, 0].bar(model_names, metrics_data['Quality'], color='skyblue')
        axes[0, 0].set_title('Середня якість коду')
        axes[0, 0].set_ylabel('Оцінка')
        axes[0, 0].set_ylim(0, 1)

        axes[0, 1].bar(model_names, metrics_data['Speed (tok/s)'], color='lightgreen')
        axes[0, 1].set_title('Швидкість генерації')
        axes[0, 1].set_ylabel('Токенів/секунда')

        axes[1, 0].bar(model_names, metrics_data['Syntax Accuracy'], color='orange')
        axes[1, 0].set_title('Синтаксична точність')
        axes[1, 0].set_ylabel('Точність')
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].bar(model_names, metrics_data['Completeness'], color='pink')
        axes[1, 1].set_title('Повнота коду')
        axes[1, 1].set_ylabel('Оцінка повноти')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model_comparison.png"
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"✅ Графіки порівняння моделей збережено до {output_path}")

def load_test_cases(json_path: Path) -> List[Dict[str, str]]:
    """Завантажує тестові кейси з JSON"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # ==== Налаштування ====
    MODELS = [
        # ('Назва', 'шлях_до_моделі', is_peft)
        ('Qwen-1.5B', 'Qwen/Qwen2.5-Coder-1.5B-Instruct', False),
        # ('FineTuned', '/шлях/до/peft-моделі', True), # додати свою модель тут
    ]
    TEST_CASES_PATH = Path("test_cases.json")  # Шлях до json з тест-кейсами
    OUTPUT_DIR = Path("./results")

    comparator = ModelComparator()

    # Додаємо моделі
    for name, path, is_peft in MODELS:
        comparator.add_model(name, path, is_peft)

    # Завантажуємо тест-кейси
    test_cases = load_test_cases(TEST_CASES_PATH)
    logger.info(f"📝 Завантажено {len(test_cases)} тестових кейсів")

    # Запуск бенчмарку
    results = comparator.run_benchmark(test_cases)

    # Вивід таблиці порівняння
    print("\n" + comparator.generate_comparison_table(results))

    # Збереження детальних результатів
    detailed_path = OUTPUT_DIR / "detailed_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"🗂 Детальні результати збережено до {detailed_path}")

    # Візуалізація
    comparator.create_comparison_plots(results, OUTPUT_DIR)
    logger.info("🏁 Завершено!")

if __name__ == "__main__":
    main()
# compare_models.py
