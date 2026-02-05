import requests
import json

class OpenRouterChat:
    def __init__(self, api_key, model="stepfun/step-3.5-flash:free"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.conversation_history = []
        
    def add_message(self, role, content):
        """Добавить сообщение в историю диалога"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_message):
        """Отправить запрос и получить ответ от модели"""
        # Добавляем сообщение пользователя в историю
        self.add_message("user", user_message)
        
        # Формируем запрос с историей диалога
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "temperature": 0.6
        }
        
        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            
            # Парсим ответ
            result = response.json()
            assistant_message = result['choices'][0]['message']['content']
            
            # Добавляем ответ ассистента в историю
            self.add_message("assistant", assistant_message)
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Ошибка при запросе к API: {e}"
        except KeyError as e:
            return f"Ошибка парсинга ответа API: {e}"
    
    def print_conversation_history(self):
        """Вывести всю историю диалога"""
        print("\n" + "="*50)
        print("ИСТОРИЯ ДИАЛОГА:")
        for msg in self.conversation_history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            print(f"{role}: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"{role}: {msg['content']}")
        print("="*50 + "\n")

def main():
    # Получаем API-ключ
    api_key = "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"
    
    # Инициализируем чат
    chat = OpenRouterChat(api_key)
    
    print("="*50)
    print("Чат с помощником (для выхода введите 'выход' или 'exit')")
    print("="*50)
    
    # Можно добавить системное сообщение для задания поведения ассистента
    system_message = input("Введите системное сообщение (или нажмите Enter для пропуска): ").strip()
    if system_message:
        chat.add_message("system", system_message)
        print(f"Системное сообщение установлено: {system_message}")
    
    while True:
        try:
            # Получаем вопрос пользователя
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                print("Завершение сеанса...")
                break
            
            if not user_input:
                print("Сообщение не может быть пустым!")
                continue
            
            # Получаем ответ от ассистента
            print("Ассистент: ", end="", flush=True)
            response = chat.get_response(user_input)
            print(response)
            
            # Спросить, хочет ли пользователь увидеть историю
            if len(chat.conversation_history) % 3 == 0:  # Каждые 3 сообщения
                show_history = input("\nПоказать историю диалога? (да/нет): ").strip().lower()
                if show_history in ['да', 'yes', 'y']:
                    chat.print_conversation_history()
                    
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")

if __name__ == "__main__":
    main()