import os
import re
import requests
import json
from typing import List, Optional, Dict

class MarkdownTranslator:
    def __init__(self, api_key: str, target_lang: str = "english", model: str = "stepfun/step-3.5-flash:free"):
        self.api_key = api_key
        self.target_lang = target_lang
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
    def split_markdown(self, content: str, max_tokens: int = 3000) -> List[str]:
        parts = []
        
        # Удаляем лишние пустые строки в начале
        content = content.lstrip('\n')
        
        # Разделяем по основным заголовкам (##, ### и т.д.)
        # Используем более сложное регулярное выражение для корректного разделения
        pattern = r'(?=\n#{1,6}\s+.+\n)'
        sections = re.split(pattern, '\n' + content)
        
        # Убираем пустые строки в начале каждой секции
        sections = [section.strip() for section in sections if section.strip()]
        
        current_part = ""
        for section in sections:
            estimated_tokens = self._estimate_tokens(current_part + section)
            
            if estimated_tokens > max_tokens and current_part:
                # Если добавление новой секции превысит лимит, сохраняем текущую часть
                parts.append(current_part.strip())
                current_part = section
            else:
                if current_part:
                    current_part += "\n\n" + section
                else:
                    current_part = section
        
        # Добавляем последнюю часть, если она не пустая
        if current_part.strip():
            parts.append(current_part.strip())
        
        # Если текст не содержит заголовков или очень большой, разбиваем по абзацам
        if len(parts) == 0 or (len(parts) == 1 and self._estimate_tokens(parts[0]) > max_tokens):
            return self._split_by_paragraphs(content, max_tokens)
        
        return parts
    
    def _split_by_paragraphs(self, content: str, max_tokens: int) -> List[str]:
        """
        Резервный метод: разбивает текст по абзацам
        
        Args:
            content: Исходный текст
            max_tokens: Максимальное количество токенов
            
        Returns:
            Список частей текста
        """
        # Разбиваем на абзацы (две и более пустых строки)
        paragraphs = re.split(r'\n\s*\n', content)
        parts = []
        current_part = ""
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            estimated_tokens = self._estimate_tokens(current_part + paragraph)
            
            if estimated_tokens > max_tokens and current_part:
                parts.append(current_part.strip())
                current_part = paragraph
            else:
                if current_part:
                    current_part += "\n\n" + paragraph
                else:
                    current_part = paragraph
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    def _estimate_tokens(self, text: str) -> int:
        words = len(text.split())
        tokens = words * 1.3
        return int(tokens)
    
    def translate_text(self, text: str) -> str:
        prompt = f"""Переведи следующий Markdown текст на {self.target_lang}. 

{text}"""
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Ты профессиональный технический переводчик. Твоя задача - точно переводить документацию, сохраняя все форматирование Markdown."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4096, 
        }
        
        try:
            print(f"Отправка запроса к OpenRouter API (модель: {self.model})...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=120  # Увеличиваем таймаут для больших текстов
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Извлекаем переведенный текст
            translated_text = result['choices'][0]['message']['content']
            
            # Очищаем возможные артефакты перевода
            translated_text = self._clean_translation(translated_text, text)
            
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Статус код: {e.response.status_code}")
                print(f"Ответ API: {e.response.text[:500]}")
            return text
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Ошибка в формате ответа API: {e}")
            print(f"Полученный ответ: {response.text[:500] if 'response' in locals() else 'Нет ответа'}")
            return text
    
    def _clean_translation(self, translated: str, original: str) -> str:
        # Удаляем возможные префиксы типа "Перевод:"
        lines = translated.split('\n')
        if len(lines) > 0 and any(keyword in lines[0].lower() for keyword in ['перевод:', 'translation:', 'ответ:']):
            lines = lines[1:]
        
        # Восстанавливаем исходные блоки кода если они были повреждены
        code_blocks = re.findall(r'```[\s\S]*?```', original)
        if code_blocks:
            translated_code_blocks = re.findall(r'```[\s\S]*?```', translated)
            for i, (orig, trans) in enumerate(zip(code_blocks, translated_code_blocks)):
                if orig != trans and '```' in orig and '```' in trans:
                    # Заменяем переведенный блок кода на оригинальный
                    translated = translated.replace(trans, orig)
        
        return '\n'.join(lines).strip()
    
    def translate_file(self, input_file: str, output_file: Optional[str] = None):
        if not os.path.exists(input_file):
            print(f"Файл {input_file} не найден!")
            return
        
        # Читаем исходный файл
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Размер файла: {len(content)} символов, ~{self._estimate_tokens(content)} токенов")
        
        # Разбиваем на части
        parts = self.split_markdown(content)
        print(f"Файл разбит на {len(parts)} логических частей")
        
        # Переводим каждую часть
        translated_parts = []
        for i, part in enumerate(parts, 1):
            print(f"\n{'='*50}")
            print(f"Перевод части {i}/{len(parts)}")
            print(f"Размер части: {len(part)} символов, ~{self._estimate_tokens(part)} токенов")
            print(f"Первые 100 символов: {part[:100]}...")
            
            translated = self.translate_text(part)
            translated_parts.append(translated)
            
            print(f"✓ Часть {i} успешно переведена")
        
        # Объединяем переведенные части
        translated_content = "\n\n".join(translated_parts)
        
        # Определяем имя выходного файла
        if output_file is None:
            name, ext = os.path.splitext(input_file)
            lang_code = self.target_lang[:2].lower()
            output_file = f"{name}_{lang_code}{ext}"
        
        # Сохраняем результат
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        print(f"\n{'='*50}")
        print(f"✅ Перевод успешно сохранен в файл: {output_file}")
        print(f"Итоговый размер: {len(translated_content)} символов")
        
    def translate_with_progress(self, input_file: str, output_file: Optional[str] = None):
        if not os.path.exists(input_file):
            print(f"Файл {input_file} не найден!")
            return
        
        # Читаем исходный файл
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Размер файла: {len(content)} символов, ~{self._estimate_tokens(content)} токенов")
        
        # Разбиваем на части
        parts = self.split_markdown(content)
        print(f"Файл разбит на {len(parts)} логических частей")
        
        # Проверяем, есть ли сохраненный прогресс
        progress_file = "translation_progress.json"
        translated_parts = []
        start_index = 0
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    translated_parts = progress.get('translated_parts', [])
                    start_index = len(translated_parts)
                    print(f"Найдено сохранение прогресса. Продолжаем с части {start_index + 1}")
            except json.JSONDecodeError:
                print("Ошибка чтения файла прогресса. Начинаем заново.")
        
        # Переводим оставшиеся части
        for i in range(start_index, len(parts)):
            print(f"\n{'='*50}")
            print(f"Перевод части {i + 1}/{len(parts)}")
            print(f"Размер части: {len(parts[i])} символов, ~{self._estimate_tokens(parts[i])} токенов")
            
            translated = self.translate_text(parts[i])
            translated_parts.append(translated)
            
            # Сохраняем прогресс
            progress = {
                'input_file': input_file,
                'target_language': self.target_lang,
                'model': self.model,
                'translated_parts': translated_parts,
                'current_index': i + 1,
                'total_parts': len(parts),
                'timestamp': os.path.getmtime(input_file) if os.path.exists(input_file) else None
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Часть {i + 1} переведена, прогресс сохранён")
        
        # Удаляем файл прогресса после успешного завершения
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Файл прогресса удалён")
        
        # Объединяем переведенные части
        translated_content = "\n\n".join(translated_parts)
        
        # Определяем имя выходного файла
        if output_file is None:
            name, ext = os.path.splitext(input_file)
            lang_code = self.target_lang[:2].lower()
            output_file = f"{name}_{lang_code}{ext}"
        
        # Сохраняем результат
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        print(f"\n{'='*50}")
        print(f"✅ Перевод успешно завершён и сохранён в файл: {output_file}")
        print(f"Итоговый размер: {len(translated_content)} символов")


def get_available_models(api_key: str) -> List[Dict]:
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        models = response.json().get('data', [])
        
        print("Доступные модели для перевода:")
        print("-" * 50)
        for model in models:
            if 'gpt' in model['id'].lower() or 'claude' in model['id'].lower():
                print(f"ID: {model['id']}")
                print(f"  Контекст: {model.get('context_length', 'N/A')} токенов")
                print(f"  Провайдер: {model.get('pricing', {}).get('prompt', 'N/A')} $/1M токенов")
                print()
        
        return models
        
    except Exception as e:
        print(f"Ошибка при получении списка моделей: {e}")
        return []


def main():
    # Настройки
    API_KEY = "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" 
    INPUT_FILE = "README.md"
    TARGET_LANGUAGE = "english"
    MODEL = "stepfun/step-3.5-flash:free"
    
    print("\n" + "="*50)
    print("Запуск переводчика")
    print("="*50)
    
    # Создаем переводчик
    translator = MarkdownTranslator(
        api_key=API_KEY, 
        target_lang=TARGET_LANGUAGE,
        model=MODEL
    )
    
    # Выбираем метод перевода
    use_progress_saving = True  # True для сохранения прогресса, False для простого метода
    
    if use_progress_saving:
        translator.translate_with_progress(INPUT_FILE)
    else:
        translator.translate_file(INPUT_FILE)


if __name__ == "__main__":
    # Проверяем наличие API ключа
    if "your_openrouter_api_key_here" in main.__code__.co_consts:
        print("⚠️  ВНИМАНИЕ: Вы используете демонстрационный API ключ!")
        print("Пожалуйста, замените 'your_openrouter_api_key_here' на ваш реальный ключ OpenRouter")
        print("Получить ключ можно на: https://openrouter.ai/keys")
        
        answer = input("Продолжить с демонстрационным ключом? (y/n): ")
        if answer.lower() != 'y':
            exit()
    
    main()