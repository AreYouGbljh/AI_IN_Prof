import requests
import csv
import time

def analyze_sentiment(api_key, review):
    """Анализ тональности одного отзыва"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Определи тональность этого отзыва. Ответь одним словом: "позитивный", "негативный" или "нейтральный".
    
Отзыв: {review}"""
    
    data = {
        "model": "stepfun/step-3.5-flash:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        sentiment = result['choices'][0]['message']['content'].strip().lower()
        
        # Нормализация ответа
        if "позитив" in sentiment or "положит" in sentiment:
            return "позитивный"
        elif "негатив" in sentiment or "отрицат" in sentiment:
            return "негативный"
        else:
            return "нейтральный"
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return "ошибка"

def main():
    api_key = "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" 
    
    reviews = [
        "Отличный курс! Все понятно объясняют, материал полезный.",
        "Мне не понравилось. Преподаватель говорил слишком быстро.",
        "Было сложно, но интересно. Много нового узнал.",
        "Супер! Рекомендую всем.",
        "Слишком дорого для такого содержания.",
        "Нормальный курс, но есть куда расти.",
    ]
    
    results = []
    
    print("Анализирую отзывы...")
    for review in reviews:
        print(f"Отзыв: {review}")
        sentiment = analyze_sentiment(api_key, review)
        results.append([review, sentiment])
        print(f"Тональность: {sentiment}\n")
        time.sleep(0.5)  # Пауза между запросами
    
    # Сохранение в CSV
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['текст_отзыва', 'тональность'])
        writer.writerows(results)
    
    print("Результаты сохранены в results.csv")

if __name__ == "__main__":
    # Для запуска простой версии:
    # main_simple()
    
    # Для запуска полной версии:
    main()