#!/usr/bin/env python3
"""
Telegram-бот с характером на OpenRouter.ai
Рабочая версия без ошибок с event loop
"""

import asyncio
import logging
import sys
import os
from typing import Optional

# Настройка asyncio для Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Импорт библиотек
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters
    )
    import aiohttp
    import json
    from datetime import datetime
except ImportError as e:
    print(f"❌ Ошибка импорта библиотек: {e}")
    print("📦 Установите зависимости: pip install python-telegram-bot aiohttp")
    sys.exit(1)

print("=" * 60)
print("🤖 TELEGRAM БОТ С ХАРАКТЕРОМ")
print("=" * 60)

# ==================== КОНФИГУРАЦИЯ ====================

def setup_config():
    """Настройка конфигурации бота"""
    
    print("\n🔧 НАСТРОЙКА КОНФИГУРАЦИИ")
    print("-" * 40)
    
    # Пробуем получить из переменных окружения
    telegram_token = "8110547264:AAFtNoRF5icAqBaOpgVPNLHDMmrrSvNdPVI"
    openrouter_key = "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"
    
    return telegram_token, openrouter_key

TELEGRAM_TOKEN, OPENROUTER_KEY = setup_config()

# Настройки бота
BOT_CONFIG = {
    "name": "Русик",
    "personality": "саркастичный, умный, немного ленивый",
    "greeting": "Привет! Я бот с характером. Задавай вопросы или используй /help для списка команд!",
    "demo_mode": OPENROUTER_KEY == "demo"
}

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== OPENROUTER КЛИЕНТ ====================

class OpenRouterClient:
    """Простой клиент для OpenRouter API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.demo_mode = api_key == "demo"
        
    async def get_response(self, message: str) -> str:
        """Получить ответ от OpenRouter"""
        
        if self.demo_mode:
            # Демо-ответы
            responses = [
                f"🤖 *В демо-режиме:* Получил твое сообщение: '{message[:50]}...'",
                "Для реальных ответов нужен API ключ от OpenRouter.ai",
                "Демо: обычно здесь был бы умный ответ от ИИ",
                "🔧 *Настройка:* Получи ключ на https://openrouter.ai/keys"
            ]
            return responses[len(message) % len(responses)]
        
        try:
            # Реальный запрос к OpenRouter
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Telegram Bot"
            }
            
            payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"Ты {BOT_CONFIG['name']}. {BOT_CONFIG['personality']}. "
                                  f"Отвечай кратко, с юмором и сарказмом."
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": 500
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        return f"❌ Ошибка API ({response.status}): {error_text[:100]}"
                        
        except asyncio.TimeoutError:
            return "⏱️ Таймаут запроса. Попробуй снова."
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return f"⚠️ Ошибка соединения: {str(e)[:100]}"

# Создаем клиент
openrouter_client = OpenRouterClient(OPENROUTER_KEY)

# ==================== ОСНОВНЫЕ КОМАНДЫ ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    
    welcome_text = f"""
{BOT_CONFIG['greeting']}

👤 *Информация о тебе:*
ID: `{user.id}`
Имя: {user.first_name or 'Не указано'}
Username: @{user.username or 'Не указан'}

🤖 *Обо мне:*
Имя: {BOT_CONFIG['name']}
Характер: {BOT_CONFIG['personality']}
Режим: {'ДЕМО 🔧' if BOT_CONFIG['demo_mode'] else 'РАБОЧИЙ ✅'}

💡 *Просто напиши мне сообщение!*
"""
    
    # Создаем клавиатуру
    keyboard = [
        [InlineKeyboardButton("❓ Помощь", callback_data="help")],
        [InlineKeyboardButton("ℹ️ О боте", callback_data="about")],
        [InlineKeyboardButton("🔄 Статус", callback_data="status")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
    
    logger.info(f"User {user.id} started bot")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    
    help_text = """
📚 *Помощь по боту*

*Основные команды:*
/start - Начать общение
/help - Эта справка
/about - Информация о боте
/image [описание] - Сгенерировать изображение

*Как использовать:*
1. Просто напиши сообщение - я отвечу
2. Используй команды для специальных действий
3. Будь вежлив и конкретен в вопросах

*Режим работы:* """ + ("ДЕМО (нужен API ключ)" if BOT_CONFIG['demo_mode'] else "РАБОЧИЙ")
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /about"""
    
    about_text = f"""
🤖 *О боте {BOT_CONFIG['name']}*

*Технологии:*
• Python 3.8+
• python-telegram-bot
• OpenRouter.ai API
• aiohttp

*Возможности:*
✅ Общение через ИИ
✅ Демо-режим для тестирования
✅ Устойчивость к ошибкам
✅ Простая настройка

*Для разработчиков:*
Код открыт для изучения и модификации.
Добавь свои фичи и улучшения!

*Текущий режим:* {'ДЕМО' if BOT_CONFIG['demo_mode'] else 'РАБОЧИЙ'}
"""
    
    await update.message.reply_text(about_text, parse_mode="Markdown")

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /image"""
    
    if not context.args:
        await update.message.reply_text(
            "🎨 *Генерация изображений*\n\n"
            "Используй: `/image [описание]`\n\n"
            "*Пример:*\n"
            "`/image кот в космосе`\n"
            "`/image закат над горами`\n"
            "`/image робот пьет кофе`",
            parse_mode="Markdown"
        )
        return
    
    prompt = " ".join(context.args)
    
    if BOT_CONFIG['demo_mode']:
        await update.message.reply_text(
            f"🖼 *Демо-режим:*\n\n"
            f"Запрос на генерацию: '{prompt}'\n\n"
            f"В рабочем режиме здесь была бы картинка!\n"
            f"Получи API ключ на https://openrouter.ai",
            parse_mode="Markdown"
        )
    else:
        # Здесь можно добавить реальную генерацию изображений
        await update.message.reply_text(
            f"🎨 Генерация изображения: '{promfit}'\n\n"
            f"*В разработке...*\n"
            f"Скоро будет добавлена генерация через Stability AI",
            parse_mode="Markdown"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user = update.effective_user
    user_message = update.message.text
    
    logger.info(f"Message from {user.id}: {user_message[:50]}...")
    
    # Показываем статус "печатает"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # Получаем ответ
    bot_response = await openrouter_client.get_response(user_message)
    
    # Отправляем ответ
    response_text = f"*{BOT_CONFIG['name']}:* {bot_response}"
    await update.message.reply_text(response_text, parse_mode="Markdown")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий кнопок"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "help":
        await help_command(update, context)
    elif query.data == "about":
        await about_command(update, context)
    elif query.data == "status":
        await update.callback_query.message.reply_text(
            f"🟢 *Статус бота:* РАБОТАЕТ\n\n"
            f"Имя: {BOT_CONFIG['name']}\n"
            f"Режим: {'ДЕМО' if BOT_CONFIG['demo_mode'] else 'РАБОЧИЙ'}\n"
            f"API: {'Не настроен' if BOT_CONFIG['demo_mode'] else 'Настроен'}\n\n"
            f"Всё работает отлично! 👍",
            parse_mode="Markdown"
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    
    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Произошла ошибка. Попробуй позже или используй /start",
            parse_mode="Markdown"
        )
    except:
        pass

# ==================== ЗАПУСК БОТА ====================

def main():
    """Основная функция запуска"""
    
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК БОТА")
    print("=" * 60)
    
    # Создаем и настраиваем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("image", image_command))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    
    application.add_error_handler(error_handler)
    
    print(f"🤖 Имя бота: {BOT_CONFIG['name']}")
    print(f"🔧 Режим: {'ДЕМО' if BOT_CONFIG['demo_mode'] else 'РАБОЧИЙ'}")
    print(f"📱 Токен: {TELEGRAM_TOKEN[:10]}...{TELEGRAM_TOKEN[-10:]}")
    print(f"🌐 API ключ: {'Не настроен' if BOT_CONFIG['demo_mode'] else 'Настроен'}")
    
    print("\n" + "=" * 60)
    print("✅ БОТ ЗАПУЩЕН!")
    print("=" * 60)
    print("📱 Найди своего бота в Telegram")
    print("💬 Напиши /start для начала общения")
    print("⏹ Нажми Ctrl+C для остановки")
    print("=" * 60 + "\n")
    
    # Запускаем бота
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при запуске: {e}")
        logger.error(f"Failed to start bot: {e}")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    # Просто запускаем main() - без asyncio.run()
    main()