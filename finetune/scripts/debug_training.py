#!/usr/bin/env python3
"""
Діагностика проблем навчання
"""

import os
import json
import sys

def check_file_contents(directory):
    """Перевіряє вміст файлів у директорії"""
    print(f"📁 Перевіряю директорію: {directory}")
    
    if not os.path.exists(directory):
        print(f"❌ Директорія {directory} не існує")
        return
    
    files = os.listdir(directory)
    print(f"📄 Знайдено файлів: {len(files)}")
    
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} байт")
            
            # Перевіряємо вміст JSON файлів
            if file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            print(f"    ⚠️ Файл порожній!")
                        else:
                            print(f"    📝 Перші 100 символів: {content[:100]}")
                            # Спробуємо парсити JSON
                            try:
                                data = json.loads(content)
                                print(f"    ✅ JSON валідний, ключів: {len(data)}")
                            except json.JSONDecodeError as e:
                                print(f"    ❌ JSON некоректний: {e}")
                except Exception as e:
                    print(f"    ❌ Помилка читання файлу: {e}")

def check_training_logs():
    """Шукає та аналізує логи навчання"""
    print("\n🔍 Шукаю логи навчання...")
    
    # Можливі місця логів
    log_locations = [
        "./training.log",
        "./output/training.log",
        "./qwen_train_*/training.log",
    ]
    
    for pattern in log_locations:
        import glob
        matching_files = glob.glob(pattern)
        for log_file in matching_files:
            print(f"📋 Знайдено лог: {log_file}")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"  Рядків у лозі: {len(lines)}")
                    
                    # Показуємо останні 10 рядків
                    print("  Останні записи:")
                    for line in lines[-10:]:
                        print(f"    {line.strip()}")
                        
            except Exception as e:
                print(f"  ❌ Помилка читання логу: {e}")

def check_system_resources():
    """Перевіряє системні ресурси"""
    print("\n💻 Перевіряю системні ресурси...")
    
    try:
        import psutil
        
        # Пам'ять
        memory = psutil.virtual_memory()
        print(f"🧠 Пам'ять: {memory.available / (1024**3):.1f}GB доступно з {memory.total / (1024**3):.1f}GB")
        
        # Диск
        disk = psutil.disk_usage('.')
        print(f"💾 Диск: {disk.free / (1024**3):.1f}GB доступно з {disk.total / (1024**3):.1f}GB")
        
        if memory.available < 2 * 1024**3:  # Менше 2GB
            print("  ⚠️ Мало пам'яті - може бути причиною проблем")
            
        if disk.free < 1024**3:  # Менше 1GB
            print("  ⚠️ Мало місця на диску")
            
    except ImportError:
        print("  ℹ️ psutil не встановлено, не можу перевірити ресурси")
    except Exception as e:
        print(f"  ❌ Помилка перевірки ресурсів: {e}")

def check_dependencies():
    """Перевіряє залежності"""
    print("\n📦 Перевіряю залежності...")
    
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'scipy',
        'accelerate'
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = "невідома версія"
            print(f"  ✅ {package}: {version}")
        except ImportError:
            print(f"  ❌ {package}: НЕ ВСТАНОВЛЕНО")
        except Exception as e:
            print(f"  ⚠️ {package}: помилка - {e}")

def suggest_fixes():
    """Пропонує рішення проблем"""
    print("\n🔧 МОЖЛИВІ РІШЕННЯ:")
    print("\n1. 📄 Пусті файли моделі:")
    print("   - Навчання було перервано або зависло")
    print("   - Перевірте логи на помилки")
    print("   - Запустіть навчання знову з меншими параметрами")
    
    print("\n2. 💾 Проблеми з диском:")
    print("   - Перевірте права доступу до директорії output/")
    print("   - Можливо файлова система read-only")
    print("   - Спробуйте запустити в іншій директорії")
    
    print("\n3. 🧠 Проблеми з пам'яттю:")
    print("   - Зменшіть batch_size до 1")
    print("   - Зменшіть max_length до 64")
    print("   - Використайте менший LoRA rank (r=1)")
    
    print("\n4. 📦 Проблеми з залежностями:")
    print("   - Встановіть відсутні пакети")
    print("   - Оновіть transformers: pip install --upgrade transformers")
    print("   - Спробуйте: pip install scipy datasets peft accelerate")
    
    print("\n5. 🏃‍♂️ Перезапуск навчання:")
    print("   - Видаліть директорію output/: rm -rf output/")
    print("   - Запустіть скрипт знову з debug інформацією")
    print("   - Додайте більше логування")

def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "./output"
    
    print("🔍 ДІАГНОСТИКА ПРОБЛЕМ НАВЧАННЯ")
    print("=" * 50)
    
    check_file_contents(directory)
    check_training_logs()
    check_system_resources()
    check_dependencies()
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("🎯 НАСТУПНІ КРОКИ:")
    print("1. Видаліть пошкоджену директорію: rm -rf output/")
    print("2. Перевірте логи навчання")
    print("3. Запустіть навчання з меншими параметрами")
    print("4. Моніторте процес навчання")

if __name__ == "__main__":
    main()