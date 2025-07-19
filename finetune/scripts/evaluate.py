#!/usr/bin/env python3
"""
Скрипт для детального оцінювання файн-тюненої моделі
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset, load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Клас для збереження метрик оцінювання"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    perplexity: float = 0.0
    bleu_score: float = 0.0
    code_quality_score: float = 0.0

class ModelEvaluator:
    """Клас для оцінювання моделі"""
    
    def __init__(self, model_path: str = "/workspace/models/finetuned"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.results = []
        
    def load_models(self):
        """Завантажує файн-тюнену та базову моделі"""
        logger.info("🔄 Завантаження моделей...")
        
        # Завантаження токенайзера
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Завантаження файн-тюненої моделі
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Завантаження базової моделі для порівняння
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("✅ Моделі завантажено успішно")
    
    def calculate_perplexity(self, model, dataset: Dataset) -> float:
        """Обчислює perplexity моделі на датасеті"""
        logger.info("📊 Обчислення perplexity...")
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for example in dataset:
                # Токенізація
                inputs = self.tokenizer(
                    example['text'] if 'text' in example else str(example),
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                # Обчислення loss
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_code_quality(self, generated_code: str) -> Dict[str, float]:
        """Оцінює якість згенерованого коду"""
        scores = {
            'syntax_score': 0.0,
            'structure_score': 0.0,
            'completeness_score': 0.0,
            'readability_score': 0.0
        }
        
        # Перевірка синтаксису
        try:
            compile(generated_code, '<string>', 'exec')
            scores['syntax_score'] = 1.0
        except SyntaxError:
            scores['syntax_score'] = 0.0
        
        # Перевірка структури
        if 'def ' in generated_code:
            scores['structure_score'] += 0.3
        if 'class ' in generated_code:
            scores['structure_score'] += 0.3
        if 'return' in generated_code:
            scores['structure_score'] += 0.2
        if generated_code.count('(') == generated_code.count(')'):
            scores['structure_score'] += 0.2
        
        # Перевірка повноти
        if len(generated_code.strip()) > 20:
            scores['completeness_score'] += 0.3
        if len(generated_code.split('\n')) > 3:
            scores['completeness_score'] += 0.3
        if any(keyword in generated_code for keyword in ['def', 'class', 'import']):
            scores['completeness_score'] += 0.4
        
        # Перевірка читабельності
        if '"""' in generated_code or "'''" in generated_code:
            scores['readability_score'] += 0.4
        if '#' in generated_code:
            scores['readability_score'] += 0.3
        if len([line for line in generated_code.split('\n') if line.strip()]) > 0:
            scores['readability_score'] += 0.3
        
        return scores
    
    def comprehensive_evaluation(self, eval_dataset_path: str):
        """Проводить комплексне оцінювання"""
        logger.info("🎯 Початок комплексного оцінювання...")
        
        # Завантаження тестового датасету
        eval_dataset = load_dataset('json', data_files=eval_dataset_path)['train']
        
        finetuned_results = []
        base_results = []
        
        for i, example in enumerate(eval_dataset):
            if i >= 50:  # Обмежуємо кількість для швидшого тестування
                break
                
            instruction = example['instruction']
            input_text = example.get('input', '')
            expected_output = example['output']
            
            # Генерація файн-тюненої моделі
            finetuned_output = self.generate_response(self.model, instruction, input_text)
            finetuned_quality = self.evaluate_code_quality(finetuned_output)
            
            # Генерація базової моделі
            base_output = self.generate_response(self.base_model, instruction, input_text)
            base_quality = self.evaluate_code_quality(base_output)
            
            finetuned_results.append({
                'instruction': instruction,
                'expected': expected_output,
                'generated': finetuned_output,
                'quality_scores': finetuned_quality
            })
            
            base_results.append({
                'instruction': instruction,
                'expected': expected_output,
                'generated': base_output,
                'quality_scores': base_quality
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Оброблено {i + 1} прикладів...")
        
        return finetuned_results, base_results
    
    def generate_response(self, model, instruction: str, input_text: str = "") -> str:
        """Генерує відповідь моделі"""
        if input_text and input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Витягуємо тільки згенерований код
        if "Output:" in generated_text:
            output = generated_text.split("Output:")[-1].strip()
        else:
            output = generated_text.replace(prompt, "").strip()
        
        return output
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> EvaluationMetrics:
        """Обчислює агреговані метрики"""
        metrics = EvaluationMetrics()
        
        if not results:
            return metrics
        
        # Середні показники якості коду
        syntax_scores = [r['quality_scores']['syntax_score'] for r in results]
        structure_scores = [r['quality_scores']['structure_score'] for r in results]
        completeness_scores = [r['quality_scores']['completeness_score'] for r in results]
        readability_scores = [r['quality_scores']['readability_score'] for r in results]
        
        metrics.accuracy = np.mean(syntax_scores)
        metrics.precision = np.mean(structure_scores)
        metrics.recall = np.mean(completeness_scores)
        metrics.f1_score = np.mean(readability_scores)
        
        # Загальна оцінка якості коду
        total_quality = []
        for r in results:
            quality = np.mean(list(r['quality_scores'].values()))
            total_quality.append(quality)
        
        metrics.code_quality_score = np.mean(total_quality)
        
        return metrics
    
    def generate_comparison_report(self, finetuned_results: List[Dict], base_results: List[Dict]):
        """Генерує порівняльний звіт"""
        logger.info("📋 Генерація порівняльного звіту...")
        
        finetuned_metrics = self.calculate_aggregate_metrics(finetuned_results)
        base_metrics = self.calculate_aggregate_metrics(base_results)
        
        # Створюємо звіт
        report = {
            'summary': {
                'total_examples_evaluated': len(finetuned_results),
                'finetuned_model_performance': {
                    'syntax_accuracy': finetuned_metrics.accuracy,
                    'structure_score': finetuned_metrics.precision,
                    'completeness_score': finetuned_metrics.recall,
                    'readability_score': finetuned_metrics.f1_score,
                    'overall_code_quality': finetuned_metrics.code_quality_score
                },
                'base_model_performance': {
                    'syntax_accuracy': base_metrics.accuracy,
                    'structure_score': base_metrics.precision,
                    'completeness_score': base_metrics.recall,
                    'readability_score': base_metrics.f1_score,
                    'overall_code_quality': base_metrics.code_quality_score
                },
                'improvements': {
                    'syntax_improvement': finetuned_metrics.accuracy - base_metrics.accuracy,
                    'structure_improvement': finetuned_metrics.precision - base_metrics.precision,
                    'completeness_improvement': finetuned_metrics.recall - base_metrics.recall,
                    'readability_improvement': finetuned_metrics.f1_score - base_metrics.f1_score,
                    'overall_improvement': finetuned_metrics.code_quality_score - base_metrics.code_quality_score
                }
            },
            'detailed_results': {
                'finetuned_examples': finetuned_results[:10],  # Перші 10 для прикладу
                'base_examples': base_results[:10]
            }
        }
        
        # Збереження звіту
        report_path = self.model_path / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Звіт збережено: {report_path}")
        
        # Виведення резюме
        self.print_summary(report['summary'])
        
        return report
    
    def print_summary(self, summary: Dict):
        """Виводить резюме оцінювання"""
        print("\n" + "="*60)
        print("📊 РЕЗЮМЕ ОЦІНЮВАННЯ МОДЕЛІ")
        print("="*60)
        
        print(f"🔍 Оцінено прикладів: {summary['total_examples_evaluated']}")
        print("\n🎯 ФАЙН-ТЮНЕНА МОДЕЛЬ:")
        ft_perf = summary['finetuned_model_performance']
        print(f"  Синтаксична точність: {ft_perf['syntax_accuracy']:.3f}")
        print(f"  Структурна оцінка: {ft_perf['structure_score']:.3f}")
        print(f"  Повнота коду: {ft_perf['completeness_score']:.3f}")
        print(f"  Читабельність: {ft_perf['readability_score']:.3f}")
        print(f"  🏆 Загальна якість: {ft_perf['overall_code_quality']:.3f}")
        
        print("\n📚 БАЗОВА МОДЕЛЬ:")
        base_perf = summary['base_model_performance']
        print(f"  Синтаксична точність: {base_perf['syntax_accuracy']:.3f}")
        print(f"  Структурна оцінка: {base_perf['structure_score']:.3f}")
        print(f"  Повнота коду: {base_perf['completeness_score']:.3f}")
        print(f"  Читабельність: {base_perf['readability_score']:.3f}")
        print(f"  🏆 Загальна якість: {base_perf['overall_code_quality']:.3f}")
        
        print("\n📈 ПОКРАЩЕННЯ:")
        improvements = summary['improvements']
        print(f"  Синтаксис: {improvements['syntax_improvement']:+.3f}")
        print(f"  Структура: {improvements['structure_improvement']:+.3f}")
        print(f"  Повнота: {improvements['completeness_improvement']:+.3f}")
        print(f"  Читабельність: {improvements['readability_improvement']:+.3f}")
        print(f"  🎉 Загальне покращення: {improvements['overall_improvement']:+.3f}")
        
        if improvements['overall_improvement'] > 0:
            print("\n✅ Файн-тюнінг УСПІШНИЙ! Модель покращилась.")
        else:
            print("\n⚠️ Потрібне додаткове налаштування. Модель не показала значних покращень.")

def main():
    """Головна функція"""
    evaluator = ModelEvaluator()
    
    try:
        # Завантаження моделей
        evaluator.load_models()
        
        # Комплексне оцінювання
        finetuned_results, base_results = evaluator.comprehensive_evaluation(
            "/workspace/data/processed/eval_dataset.jsonl"
        )
        
        # Генерація звіту
        report = evaluator.generate_comparison_report(finetuned_results, base_results)
        
        print("\n🎉 Оцінювання завершено успішно!")
        
    except Exception as e:
        logger.error(f"❌ Помилка під час оцінювання: {e}")
        raise

if __name__ == "__main__":
    main()
