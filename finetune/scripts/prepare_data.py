#!/usr/bin/env python3
"""
Скрипт для підготовки тренувальних даних з MD файлів про гідропонне вирощування гороху
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import markdown
from bs4 import BeautifulSoup


class HydroponicDataProcessor:
    """Клас для обробки даних про гідропонне вирощування гороху"""

    def __init__(self, data_dir: str = "/workspace/data", output_dir: str = "/workspace/data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.md_files = [
            'Growing Peas Hydroponically A Comprehensive Guide.md',
            'Growing Peas Without Soil_.md',
            'Growing_Green_Peas.md',
            'HYDROPONIC GREEN PEA PRODUCTION MASTER GUIDE.md',
            'Hydroponic Green Pea Cultivation Guide.md',
            'green_peas_growing_guide.md'
        ]

    def extract_text_from_markdown(self, file_path: Path) -> str:
        """Витягує чистий текст з markdown файлу"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        except Exception as exc:
            print(f"⚠️  Помилка читання {file_path}: {exc}")
            return ""

    def extract_sections(self, text: str) -> Dict[str, List[str]]:
        """Розбиває текст на тематичні секції"""
        sections = {
            'equipment': [],
            'nutrients': [],
            'growing_process': [],
            'troubleshooting': [],
            'harvesting': [],
            'general': []
        }

        equipment_keywords = ['equipment', 'system', 'pump', 'reservoir', 'light', 'container', 'hydroponic system']
        nutrient_keywords = ['nutrient', 'ph', 'ec', 'fertilizer', 'solution', 'concentration', 'ppm']
        process_keywords = ['germination', 'seedling', 'transplant', 'growth stage', 'flowering', 'pollination']
        troubleshooting_keywords = ['problem', 'disease', 'pest', 'deficiency', 'yellowing', 'wilting']
        harvesting_keywords = ['harvest', 'picking', 'mature', 'ready', 'yield']

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for para in paragraphs:
            lower = para.lower()
            if any(k in lower for k in equipment_keywords):
                sections['equipment'].append(para)
            elif any(k in lower for k in nutrient_keywords):
                sections['nutrients'].append(para)
            elif any(k in lower for k in process_keywords):
                sections['growing_process'].append(para)
            elif any(k in lower for k in troubleshooting_keywords):
                sections['troubleshooting'].append(para)
            elif any(k in lower for k in harvesting_keywords):
                sections['harvesting'].append(para)
            else:
                sections['general'].append(para)
        return sections

    def generate_training_samples(self) -> List[Dict[str, str]]:
        """Повертає список тренувальних зразків"""
        print("🔄 Обробка MD файлів та генерація зразків...")
        samples: List[Dict[str, str]] = []
        all_content = []
        for md_file in self.md_files:
            path = self.data_dir / md_file
            if path.exists():
                print(f"📖 Обробка файлу: {md_file}")
                text = self.extract_text_from_markdown(path)
                if text:
                    all_content.append((md_file, text))
                    print(f"   ✅ Витягнуто {len(text)} символів")
            else:
                print(f"⚠️  Файл не знайдено: {md_file}")

        for filename, content in all_content:
            sections = self.extract_sections(content)
            file_samples = self.create_samples_from_sections(sections, filename)
            samples.extend(file_samples)
            print(f"   📝 Створено {len(file_samples)} зразків з {filename}")

        samples.extend(self.generate_base_code_samples())
        print(f"📊 Загалом створено {len(samples)} тренувальних зразків")
        return samples

    def create_samples_from_sections(self, sections: Dict[str, List[str]], filename: str) -> List[Dict[str, str]]:
        """Створює зразки з секцій тексту"""
        samples: List[Dict[str, str]] = []
        for equipment_text in sections['equipment'][:3]:
            samples.append({
                "instruction": "Напиши програму для моніторингу гідропонного обладнання для вирощування гороху",
                "input": "Python",
                "output": self.generate_equipment_monitoring_code(equipment_text)
            })
        for nutrient_text in sections['nutrients'][:3]:
            samples.append({
                "instruction": "Створи функцію для розрахунку поживного розчину для гороху",
                "input": "Python",
                "output": self.generate_nutrient_calculation_code(nutrient_text)
            })
        for process_text in sections['growing_process'][:3]:
            samples.append({
                "instruction": "Напиши клас для відстеження стадій росту гороху в гідропоніці",
                "input": "Python",
                "output": self.generate_growth_tracking_code(process_text)
            })
        for trouble_text in sections['troubleshooting'][:2]:
            samples.append({
                "instruction": "Створи систему діагностики проблем при вирощуванні гороху",
                "input": "Python",
                "output": self.generate_troubleshooting_code(trouble_text)
            })
        for harvest_text in sections['harvesting'][:2]:
            samples.append({
                "instruction": "Напиши функцію для розрахунку оптимального часу збору врожаю гороху",
                "input": "Python",
                "output": self.generate_harvest_timing_code(harvest_text)
            })
        return samples

    def generate_equipment_monitoring_code(self, context_text: str) -> str:
        """Генерує код для моніторингу обладнання"""
        return '''class HydroponicEquipmentMonitor:
    """Система моніторингу гідропонного обладнання для гороху"""

    def __init__(self):
        self.water_pump_status = True
        self.air_pump_status = True
        self.ph_sensor_value = 6.0
        self.ec_sensor_value = 1.2
        self.water_level = 80.0  # у відсотках
        self.temperature = 20.0  # °C

    def check_water_pump(self):
        """Перевіряє стан водяного насосу"""
        if not self.water_pump_status:
            return {"status": "ERROR", "message": "Водяний насос не працює"}
        return {"status": "OK", "message": "Водяний насос працює нормально"}

    def check_nutrient_levels(self):
        """Перевіряє рівень поживних речовин"""
        if self.ec_level < 0.8:
            return {"status": "WARNING", "message": "Низький рівень поживних речовин"}
        elif self.ec_level > 2.0:
            return {"status": "WARNING", "message": "Високий рівень поживних речовин"}
        return {"status": "OK", "message": "Рівень поживних речовин в нормі"}

    def monitor_ph(self):
        """Моніторить рівень pH"""
        optimal_ph = 6.0
        if abs(self.ph_sensor_value - optimal_ph) > 0.5:
            return {"status": "WARNING", "message": f"pH {self.ph_sensor_value} потребує корекції"}
        return {"status": "OK", "message": "pH в оптимальному діапазоні"}

    def get_system_status(self):
        """Повертає загальний стан системи"""
        checks = [
            self.check_water_pump(),
            self.check_nutrient_levels(),
            self.monitor_ph()
        ]
        errors = [c for c in checks if c['status'] == 'ERROR']
        warnings = [c for c in checks if c['status'] == 'WARNING']
        if errors:
            return {"overall_status": "CRITICAL", "issues": errors}
        elif warnings:
            return {"overall_status": "WARNING", "issues": warnings}
        else:
            return {"overall_status": "OK", "message": "Всі системи працюють нормально"}'''

    def generate_nutrient_calculation_code(self, context_text: str) -> str:
        """Генерує код для розрахунку поживних речовин"""
        return '''class PeaNutrientCalculator:
    """Розрахунок поживного розчину для гороху"""

    def __init__(self):
        self.optimal_nutrients = {
            'nitrogen': 150,
            'phosphorus': 50,
            'potassium': 200,
            'calcium': 180,
            'magnesium': 50,
            'sulfur': 70
        }
        self.optimal_ph = 6.0
        self.optimal_ec = 1.2

    def calculate_nutrient_solution(self, water_volume_liters: float, growth_stage: str):
        """Розраховує концентрацію поживних речовин"""
        stage_multipliers = {
            'seedling': 0.5,
            'vegetative': 1.0,
            'flowering': 1.2,
            'pod_filling': 1.3
        }
        multiplier = stage_multipliers.get(growth_stage, 1.0)
        solution = {}
        for nutrient, base_ppm in self.optimal_nutrients.items():
            adjusted_ppm = base_ppm * multiplier
            mg_per_liter = adjusted_ppm
            total_mg = mg_per_liter * water_volume_liters
            solution[nutrient] = {
                'ppm': adjusted_ppm,
                'mg_per_liter': mg_per_liter,
                'total_mg': total_mg
            }
        return solution

    def adjust_ph(self, current_ph: float, target_ph: float = None):
        """Розраховує корекцію pH"""
        if target_ph is None:
            target_ph = self.optimal_ph
        ph_difference = current_ph - target_ph
        if ph_difference > 0.1:
            return {
                'action': 'add_ph_down',
                'amount': f'{abs(ph_difference) * 2:.1f} ml на 10L води',
                'reason': 'pH занадто високий'
            }
        elif ph_difference < -0.1:
            return {
                'action': 'add_ph_up',
                'amount': f'{abs(ph_difference) * 1.5:.1f} ml на 10L води',
                'reason': 'pH занадто низький'
            }
        else:
            return {'action': 'no_adjustment', 'reason': 'pH в оптимальному діапазоні'}

    def monitor_ec_levels(self, current_ec: float):
        """Моніторить рівень електропровідності"""
        if current_ec < self.optimal_ec - 0.2:
            return {'status': 'low', 'recommendation': 'Додати поживні речовини'}
        elif current_ec > self.optimal_ec + 0.3:
            return {'status': 'high', 'recommendation': 'Розбавити водою'}
        else:
            return {'status': 'optimal', 'recommendation': 'Рівень поживних речовин оптимальний'}'''

    def generate_growth_tracking_code(self, context_text: str) -> str:
        """Генерує код для відстеження росту"""
        return '''class PeaGrowthTracker:
    """Система відстеження росту гороху в гідропоніці"""

    def __init__(self, variety: str = "sugar_snap"):
        self.variety = variety
        self.germination_date = None
        self.transplant_date = None
        self.current_stage = "seed"
        self.days_since_germination = 0
        self.growth_stages = {
            'seed': {'duration': 0, 'description': 'Насіння'},
            'germination': {'duration': 7, 'description': 'Проростання'},
            'seedling': {'duration': 14, 'description': 'Сіянець'},
            'vegetative': {'duration': 30, 'description': 'Вегетативний ріст'},
            'flowering': {'duration': 21, 'description': 'Цвітіння'},
            'pod_development': {'duration': 14, 'description': 'Формування стручків'},
            'harvest_ready': {'duration': 7, 'description': 'Готовий до збору'}
        }

    def start_germination(self):
        """Починає процес проростання"""
        from datetime import datetime
        self.germination_date = datetime.now()
        self.current_stage = "germination"
        return f"Проростання почато {self.germination_date.strftime('%Y-%m-%d')}"

    def update_growth_stage(self):
        """Оновлює стадію росту"""
        if not self.germination_date:
            return "Проростання ще не почато"
        from datetime import datetime
        self.days_since_germination = (datetime.now() - self.germination_date).days
        cumulative_days = 0
        for stage, info in self.growth_stages.items():
            cumulative_days += info['duration']
            if self.days_since_germination <= cumulative_days:
                self.current_stage = stage
                break
        return f"Поточна стадія: {self.growth_stages[self.current_stage]['description']}"

    def get_care_recommendations(self):
        """Повертає рекомендації по догляду"""
        recommendations = {
            'germination': [
                'Температура 18-20°C',
                'Висока вологість',
                'Слабке освітлення'
            ],
            'seedling': [
                'Збільшити освітлення до 12-14 годин',
                'pH 6.0-6.2',
                'EC 0.8-1.0'
            ],
            'vegetative': [
                'Повне освітлення 14-16 годин',
                'pH 6.0-6.5',
                'EC 1.2-1.4',
                "Підтримка опор для в'юнких сортів"
            ],
            'flowering': [
                'Стабільне освітлення',
                'Уникати стресових факторів',
                'Контролювати вологість для запилення'
            ],
            'pod_development': [
                'Збільшити калій у розчині',
                'Регулярний полив',
                'Перевіряти на шкідників'
            ],
            'harvest_ready': [
                'Збирати щодня',
                'Ранкові години для збору',
                'Перевіряти стручки на зрілість'
            ]
        }
        return recommendations.get(self.current_stage, ['Загальний догляд'])

    def predict_harvest_date(self):
        """Прогнозує дату збору врожаю"""
        if not self.germination_date:
            return "Спочатку почніть проростання"
        total_days = sum(stage['duration'] for stage in self.growth_stages.values())
        from datetime import timedelta
        harvest_date = self.germination_date + timedelta(days=total_days)
        return f"Очікувана дата початку збору: {harvest_date.strftime('%Y-%m-%d')}"'''

    def generate_troubleshooting_code(self, context_text: str) -> str:
        """Генерує код для діагностики проблем"""
        return '''class PeaTroubleshootingSystem:
    """Система діагностики проблем при вирощуванні гороху"""

    def __init__(self):
        self.common_issues = {
            'yellowing_leaves': {
                'causes': ['nitrogen_deficiency', 'overwatering', 'root_rot'],
                'solutions': ['increase_nitrogen', 'reduce_watering', 'check_roots']
            },
            'stunted_growth': {
                'causes': ['nutrient_deficiency', 'ph_imbalance', 'insufficient_light'],
                'solutions': ['check_nutrient_levels', 'adjust_ph', 'increase_lighting']
            },
            'poor_flowering': {
                'causes': ['too_much_nitrogen', 'temperature_stress', 'insufficient_pollination'],
                'solutions': ['reduce_nitrogen', 'control_temperature', 'manual_pollination']
            }
        }

    def diagnose_leaf_problems(self, symptoms: list):
        """Діагностує проблеми з листям"""
        diagnosis = []
        if 'yellowing' in symptoms:
            if 'lower_leaves' in symptoms:
                diagnosis.append({
                    'problem': 'Дефіцит азоту',
                    'solution': 'Збільшити концентрацію азоту в розчині',
                    'urgency': 'medium'
                })
            elif 'all_leaves' in symptoms:
                diagnosis.append({
                    'problem': 'Перезволоження або корінцева гниль',
                    'solution': 'Перевірити корені та зменшити полив',
                    'urgency': 'high'
                })
        if 'brown_spots' in symptoms:
            diagnosis.append({
                'problem': 'Грибкове захворювання',
                'solution': 'Покращити вентиляцію та обробити фунгіцидом',
                'urgency': 'high'
            })
        if 'curling' in symptoms:
            diagnosis.append({
                'problem': 'Стрес від температури або вологості',
                'solution': 'Налаштувати клімат-контроль',
                'urgency': 'medium'
            })
        return diagnosis

    def check_nutrient_deficiency(self, ph: float, ec: float, visual_symptoms: list):
        """Перевіряє дефіцит поживних речовин"""
        deficiencies = []
        if ph < 5.5:
            deficiencies.append({
                'nutrient': 'Загальна доступність',
                'reason': 'pH занадто кислий',
                'solution': 'Підвищити pH до 6.0-6.5'
            })
        elif ph > 7.0:
            deficiencies.append({
                'nutrient': 'Залізо та мікроелементи',
                'reason': 'pH занадто лужний',
                'solution': 'Знизити pH до 6.0-6.5'
            })
        if ec < 0.8:
            deficiencies.append({
                'nutrient': 'Загальні поживні речовини',
                'reason': 'Низький рівень солей',
                'solution': 'Збільшити концентрацію добрив'
            })
        if 'interveinal_chlorosis' in visual_symptoms:
            deficiencies.append({
                'nutrient': 'Залізо',
                'reason': 'Хлороз між жилками листя',
                'solution': 'Додати хелатне залізо'
            })
        if 'purple_stems' in visual_symptoms:
            deficiencies.append({
                'nutrient': 'Фосфор',
                'reason': 'Фіолетове забарвлення стебел',
                'solution': 'Збільшити рівень фосфору'
            })
        return deficiencies

    def emergency_protocol(self, issue_type: str):
        """Протокол екстреної допомоги"""
        protocols = {
            'system_failure': [
                '1. Перевірити живлення',
                '2. Забезпечити ручний полив',
                '3. Перевірити насоси',
                "4. Зв'язатися з технічною підтримкою"
            ],
            'nutrient_lockout': [
                '1. Промити систему чистою водою',
                '2. Перевірити pH',
                '3. Приготувати свіжий розчин',
                '4. Поступово збільшувати концентрацію'
            ],
            'disease_outbreak': [
                '1. Ізолювати уражені рослини',
                '2. Покращити вентиляцію',
                '3. Зменшити вологість',
                '4. Застосувати відповідні препарати'
            ]
        }
        return protocols.get(issue_type, ['Зв\'язатися зі спеціалістом'])'''

    def generate_harvest_timing_code(self, context_text: str) -> str:
        """Генерує код для визначення часу збору врожаю"""
        return '''class PeaHarvestOptimizer:
    """Система оптимізації збору врожаю гороху"""

    def __init__(self):
        self.variety_characteristics = {
            'sugar_snap': {
                'days_to_harvest': 60,
                'optimal_size': 'plump_pods',
                'harvest_window': 5
            },
            'snow_pea': {
                'days_to_harvest': 55,
                'optimal_size': 'flat_pods',
                'harvest_window': 3
            },
            'shelling_pea': {
                'days_to_harvest': 70,
                'optimal_size': 'full_pods',
                'harvest_window': 7
            }
        }

    def assess_pod_readiness(self, variety: str, visual_indicators: list):
        """Оцінює готовність стручків до збору"""
        characteristics = self.variety_characteristics.get(variety, {})
        readiness_score = 0
        feedback = []
        if variety == 'sugar_snap':
            if 'plump_pods' in visual_indicators:
                readiness_score += 40
                feedback.append("Стручки добре наповнені")
            if 'bright_green' in visual_indicators:
                readiness_score += 30
                feedback.append("Яскраво-зелений колір")
            if 'crisp_texture' in visual_indicators:
                readiness_score += 30
                feedback.append("Хрусткі стручки")
        elif variety == 'snow_pea':
            if 'flat_pods' in visual_indicators:
                readiness_score += 50
                feedback.append("Плоскі стручки без горбочків")
            if 'tender_pods' in visual_indicators:
                readiness_score += 50
                feedback.append("Ніжні стручки")
        elif variety == 'shelling_pea':
            if 'full_pods' in visual_indicators:
                readiness_score += 40
                feedback.append("Повністю наповнені стручки")
            if 'round_peas' in visual_indicators:
                readiness_score += 40
                feedback.append("Круглі горошини видно крізь стручок")
            if 'glossy_pods' in visual_indicators:
                readiness_score += 20
                feedback.append("Глянсові стручки")
        if readiness_score >= 80:
            recommendation = "ЗБИРАТИ ЗАРАЗ - оптимальна зрілість"
        elif readiness_score >= 60:
            recommendation = "Готові до збору протягом 1-2 днів"
        elif readiness_score >= 40:
            recommendation = "Почекати ще 2-3 дні"
        else:
            recommendation = "Занадто рано для збору"
        return {
            'readiness_score': readiness_score,
            'recommendation': recommendation,
            'feedback': feedback
        }

    def calculate_harvest_schedule(self, planting_date, variety: str):
        """Розраховує графік збору врожаю"""
        from datetime import datetime, timedelta
        if isinstance(planting_date, str):
            planting_date = datetime.strptime(planting_date, "%Y-%m-%d")
        characteristics = self.variety_characteristics.get(variety, {})
        days_to_harvest = characteristics.get('days_to_harvest', 65)
        harvest_window = characteristics.get('harvest_window', 5)
        harvest_start = planting_date + timedelta(days=days_to_harvest)
        harvest_end = harvest_start + timedelta(days=harvest_window)
        return {
            'variety': variety,
            'harvest_start': harvest_start.strftime("%Y-%m-%d"),
            'harvest_end': harvest_end.strftime("%Y-%m-%d"),
            'window_days': harvest_window
        }
'''

    def generate_base_code_samples(self) -> List[Dict[str, str]]:
        """Повертає базові приклади коду"""
        return [
            {
                "instruction": "Напиши функцію для додавання двох чисел",
                "input": "Python",
                "output": """def add_numbers(a, b):\n    return a + b"""
            },
            {
                "instruction": "Створи функцію для множення двох чисел",
                "input": "Python",
                "output": """def multiply(a, b):\n    return a * b"""
            },
            {
                "instruction": "Напиши функцію для обчислення факторіалу",
                "input": "",
                "output": """def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"""
            },
            {
                "instruction": "Створи клас для роботи з банківським рахунком",
                "input": "Python",
                "output": """class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    def deposit(self, amount):\n        self.balance += amount\n    def withdraw(self, amount):\n        if amount <= self.balance:\n            self.balance -= amount"""
            },
            {
                "instruction": "Напиши функцію для сортування списку",
                "input": "bubble sort",
                "output": """def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"""
            },
            {
                "instruction": "Створи функцію для перевірки паліндрому",
                "input": "Python",
                "output": """def is_palindrome(text):\n    cleaned = ''.join(text.split()).lower()\n    return cleaned == cleaned[::-1]"""
            }
        ]

    def create_datasets(self, train_ratio: float = 0.8):
        """Генерує та зберігає датасети"""
        samples = self.generate_training_samples()
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]
        train_file = self.output_dir / "train_dataset.jsonl"
        eval_file = self.output_dir / "eval_dataset.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        with open(eval_file, 'w', encoding='utf-8') as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        stats = {
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'eval_samples': len(eval_samples),
            'train_ratio': train_ratio,
            'categories': self.get_sample_categories(samples)
        }
        with open(self.output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"💾 Дані збережено в {self.output_dir}")
        return train_file, eval_file, stats

    def get_sample_categories(self, samples: List[Dict[str, str]]) -> Dict[str, int]:
        """Підраховує кількість зразків за категоріями"""
        categories: Dict[str, int] = {}
        for sample in samples:
            instruction = sample['instruction'].lower()
            if 'функцію' in instruction or 'function' in instruction:
                categories['functions'] = categories.get('functions', 0) + 1
            elif 'клас' in instruction or 'class' in instruction:
                categories['classes'] = categories.get('classes', 0) + 1
            elif 'гідропон' in instruction or 'hydroponic' in instruction:
                categories['hydroponic'] = categories.get('hydroponic', 0) + 1
            else:
                categories['other'] = categories.get('other', 0) + 1
        return categories

    def validate_datasets(self):
        """Перевіряє збережені датасети"""
        train_file = self.output_dir / "train_dataset.jsonl"
        eval_file = self.output_dir / "eval_dataset.jsonl"
        with open(train_file, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_lines = f.readlines()
        print(f"📝 Тренувальний файл: {len(train_lines)} рядків")
        print(f"📝 Валідаційний файл: {len(eval_lines)} рядків")
        try:
            sample = json.loads(train_lines[0])
            for key in ['instruction', 'input', 'output']:
                if key not in sample:
                    raise ValueError(f"Відсутній ключ: {key}")
            print("✅ Формат даних валідний")
        except Exception as exc:
            print(f"❌ Помилка перевірки даних: {exc}")


if __name__ == "__main__":
    processor = HydroponicDataProcessor()
    processor.create_datasets()
    processor.validate_datasets()
