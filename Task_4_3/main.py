import re
from datetime import datetime, timedelta
from collections import Counter
import os

print("=" * 60)
print("📊 АНАЛИЗАТОР ЛОГОВ")
print("=" * 60)

def analyze_logs_simple(log_file="app.log", hours=24):
    """Простой анализ логов"""
    
    print(f"📖 Анализирую файл: {log_file}")
    print(f"⏰ Период: последние {hours} часов")
    
    cutoff = datetime.now() - timedelta(hours=hours)
    errors = []
    warnings = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем ошибки и предупреждения
                if re.search(r'ERROR|FAILED|CRITICAL', line, re.IGNORECASE):
                    errors.append(line)
                elif re.search(r'WARNING|ALERT', line, re.IGNORECASE):
                    warnings.append(line)
    
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return
    
    # Генерируем отчет
    generate_simple_report(errors, warnings, hours)

def create_test_logs(filename):
    """Создание тестовых логов"""
    logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO Приложение запущено",
        f"{(datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')} ERROR Ошибка базы данных: соединение разорвано",
        f"{(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')} WARNING Высокое использование памяти: 85%",
        f"{(datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')} ERROR Таймаут запроса API",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO Пользователь успешно аутентифицирован",
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(log + '\n')
    
    print(f"✅ Создан файл: {filename}")

def generate_simple_report(errors, warnings, hours):
    """Генерация простого отчета"""
    
    print("\n" + "=" * 60)
    print("📋 ОТЧЕТ АНАЛИЗА ЛОГОВ")
    print("=" * 60)
    
    total_errors = len(errors)
    total_warnings = len(warnings)
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"• Ошибок (ERROR): {total_errors}")
    print(f"• Предупреждений (WARNING): {total_warnings}")
    print(f"• Всего проблем: {total_errors + total_warnings}")
    
    if total_errors + total_warnings == 0:
        print(f"\n✅ За последние {hours} часов ошибок не обнаружено!")
        return
    
    print(f"\n🚨 ПОСЛЕДНИЕ ОШИБКИ:")
    for error in errors[:5]:  # Показываем первые 5 ошибок
        print(f"  • {error[:100]}...")
    
    print(f"\n⚠️  ПОСЛЕДНИЕ ПРЕДУПРЕЖДЕНИЯ:")
    for warning in warnings[:3]:  # Показываем первые 3 предупреждения
        print(f"  • {warning[:100]}...")
    
    # Анализ частых ошибок
    print(f"\n🔍 АНАЛИЗ ОШИБОК:")
    
    # Группируем ошибки по типам
    error_types = Counter()
    for error in errors:
        if "баз" in error.lower() or "database" in error.lower():
            error_types["Database"] += 1
        elif "timeout" in error.lower() or "таймаут" in error.lower():
            error_types["Timeout"] += 1
        elif "memory" in error.lower() or "памят" in error.lower():
            error_types["Memory"] += 1
        elif "соединен" in error.lower() or "connection" in error.lower():
            error_types["Connection"] += 1
        else:
            error_types["Other"] += 1
    
    for error_type, count in error_types.most_common():
        if count > 0:
            print(f"  • {error_type}: {count} случаев")
    
    # Простые рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if error_types.get("Database", 0) > 0:
        print("  • Проверьте соединение с базой данных")
    
    if error_types.get("Timeout", 0) > 0:
        print("  • Увеличьте таймауты или оптимизируйте запросы")
    
    if error_types.get("Memory", 0) > 0:
        print("  • Проверьте использование памяти приложением")
    
    if total_errors > 5:
        print("  • Критический уровень ошибок! Требуется немедленное вмешательство")
    elif total_errors > 0:
        print("  • Есть ошибки, требующие внимания")
    else:
        print("  • Система работает стабильно")

def main():
    """Основная функция"""
    
    log_file = "App.log"
    hours = 24
    
    analyze_logs_simple(log_file, hours)
    
    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 60)

if __name__ == "__main__":
    main()