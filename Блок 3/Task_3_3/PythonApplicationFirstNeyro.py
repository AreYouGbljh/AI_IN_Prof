import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем случайные seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ДЕТЕКТОР СПАМА НА TF-IDF + НЕЙРОСЕТЬ")
print("=" * 60)

# 1. ЗАГРУЗКА И ИЗУЧЕНИЕ ДАННЫХ
print("\n1. ЗАГРУЖАЕМ И ИЗУЧАЕМ ДАННЫЕ")

# Создаем датасет вручную (альтернатива загрузке)
# Источник: UCI SMS Spam Collection Dataset
print("Создаем датасет SMS сообщений...")

# Примеры реальных SMS сообщений
spam_examples = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net 16+",
    "Congratulations! You've been selected for a free iPhone! Reply YES to claim your prize. Delivery in 24h.",
    "Last chance to claim your prize! Call now 0800 123 4567 before it's too late.",
    "You have won a £1000 cash prize! To claim, text CLAIM to 88888. Std rates apply.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May. Text FA to 87121 to receive entry question.",
    "Your mobile number has won a £2000 prize. To claim, call 09058094561.",
    "You are a winner! You have been specially selected for this offer. Reply WIN to 444.",
    "Christmas sale! Buy 1 get 1 free on all gifts. Visit www.xmasshop.com",
    "Earn £1000 per week from home! No experience needed. Start today: www.easy-money.com"
]

ham_examples = [
    "Sorry, I'll call later",
    "Don't worry, I'm on my way",
    "See you at the meeting tomorrow at 10am",
    "Can you pick up some milk on your way home?",
    "Just finished work, heading home now",
    "Thanks for your help yesterday",
    "Meeting postponed to 3pm",
    "Don't forget to call your mom",
    "What time are we meeting for lunch?",
    "The package has been delivered"
]

# Добавляем больше примеров для разнообразия
more_spam = [
    "Limited time offer! 50% off all products. Use code SAVE50",
    "Your account has been compromised. Click here to secure: http://fake-bank.com",
    "You have 1 new voicemail. Call 09001234567 to listen (premium rate)",
    "Claim your free gift card now! Text GIFT to 55555",
    "Your computer may be infected. Download antivirus from: www.virus-protect-fake.com",
    "You have been pre-approved for a $5000 loan! No credit check required.",
    "Hot singles in your area want to meet you! Click here now.",
    "Your PayPal account needs verification. Login immediately to avoid suspension.",
    "Your Netflix subscription has expired. Update payment info at: netflix-fake-update.com",
    "Your flight has been cancelled. Call 1-800-FAKE-AIR for rebooking."
]

more_ham = [
    "Are we still on for dinner tonight?",
    "Can you send me the report when you get a chance?",
    "Running 15 minutes late, sorry",
    "Happy birthday! Hope you have a great day",
    "The meeting room has been changed to room 205",
    "Let me know if you need anything else",
    "Just saw your message, will respond properly later",
    "What's the plan for the weekend?",
    "Thanks for the update",
    "Call me when you're free"
]

# Объединяем все примеры
spam_messages = spam_examples + more_spam
ham_messages = ham_examples + more_ham

# Создаем DataFrame
data = []
for msg in spam_messages:
    data.append({"text": msg, "label": "spam"})
for msg in ham_messages:
    data.append({"text": msg, "label": "ham"})

df = pd.DataFrame(data)

print(f"Размер датасета: {len(df)} сообщений")
print(f"Спам: {len(spam_messages)} сообщений")
print(f"Не спам: {len(ham_messages)} сообщений")

# 2. АНАЛИЗ ДАННЫХ
print("\n2. АНАЛИЗИРУЕМ ДАННЫЕ")

# Посмотрим на первые несколько сообщений
print("\nПримеры сообщений:")
print("-" * 50)
for i in range(3):
    print(f"{df['label'][i].upper()}: {df['text'][i][:80]}...")
print("-" * 50)

# Распределение классов
plt.figure(figsize=(8, 5))
class_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # зеленый для ham, красный для spam
bars = plt.bar(class_counts.index, class_counts.values, color=colors)
plt.title('Распределение классов в датасете', fontsize=14)
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество сообщений', fontsize=12)
#plt.show()

# Добавляем значения на столбцы
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom')

plt.tight_layout()

# Длина сообщений
df['message_length'] = df['text'].apply(len)
print(f"\nСредняя длина сообщения: {df['message_length'].mean():.1f} символов")
print(f"Минимальная длина: {df['message_length'].min()} символов")
print(f"Максимальная длина: {df['message_length'].max()} символов")

# Длина сообщений по классам
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df[df['label'] == 'ham']['message_length'].hist(bins=30, color='green', alpha=0.7)
plt.title('Длина НЕ спам сообщений')
plt.xlabel('Длина (символы)')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
df[df['label'] == 'spam']['message_length'].hist(bins=30, color='red', alpha=0.7)
plt.title('Длина спам сообщений')
plt.xlabel('Длина (символы)')
plt.ylabel('Частота')

plt.tight_layout()
#plt.show()

# 3. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ТЕКСТА
print("\n3. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ТЕКСТА")

def simple_text_preprocess(text):
    """Простая предобработка текста"""
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Убираем лишние пробелы
    text = " ".join(text.split())
    
    return text

# Применяем предобработку
df['clean_text'] = df['text'].apply(simple_text_preprocess)

print("Примеры до и после обработки:")
print("-" * 50)
print(f"ДО: {df['text'][0][:100]}...")
print(f"ПОСЛЕ: {df['clean_text'][0][:100]}...")
print("-" * 50)

# 4. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
print("\n4. РАЗДЕЛЕНИЕ ДАННЫХ")

X = df['clean_text']  # тексты сообщений
y = df['label']       # метки (spam/ham)

# Преобразуем метки в числа (0=ham, 1=spam)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Кодировка классов: {label_encoder.classes_}")
print(f"Пример: 'ham' -> {label_encoder.transform(['ham'])[0]}")
print(f"Пример: 'spam' -> {label_encoder.transform(['spam'])[0]}")

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nОбучающая выборка: {len(X_train)} сообщений")
print(f"Тестовая выборка: {len(X_test)} сообщений")
print(f"Процент спама в обучающей выборке: {(y_train == 1).mean() * 100:.1f}%")
print(f"Процент спама в тестовой выборке: {(y_test == 1).mean() * 100:.1f}%")

# 5. ВЕКТОРИЗАЦИЯ TF-IDF
print("\n5. ПРЕОБРАЗОВАНИЕ ТЕКСТА В ЧИСЛА (TF-IDF)")

print("TF-IDF (Term Frequency-Inverse Document Frequency) - это метод, который:")
print("1. Считает, как часто слово встречается в документе (TF)")
print("2. Учитывает, насколько слово редкое во всех документах (IDF)")
print("3. Объединяет оба значения для получения важности слова")

# Создаем TF-IDF вектор
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,      
    min_df=2,              
    max_df=0.9,            
    ngram_range=(1, 2)      
)

# Обучаем на тренировочных данных и преобразуем их
print("\nОбучаем TF-IDF вектор...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Преобразуем тестовые данные (только transform, не fit!)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Размерность признаков после TF-IDF: {X_train_tfidf.shape[1]}")
print(f"Пример признаков (первые 10): {tfidf_vectorizer.get_feature_names_out()[:10]}")

# Преобразуем из sparse matrix в dense для нейросети
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

print(f"\nФорма данных для нейросети:")
print(f"X_train: {X_train_dense.shape}")
print(f"X_test: {X_test_dense.shape}")

# 6. СОЗДАЕМ НЕЙРОННУЮ СЕТЬ
print("\n6. СОЗДАЕМ НЕЙРОННУЮ СЕТЬ")

# Простая нейросеть для бинарной классификации
model = keras.Sequential([
    # Входной слой
    layers.Dense(128, activation='relu', input_shape=(X_train_dense.shape[1],)),
    layers.Dropout(0.3),  # предотвращаем переобучение
    
    # Скрытый слой
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Выходной слой (1 нейрон, сигмоида для бинарной классификации)
    layers.Dense(1, activation='sigmoid')
])

# Компилируем модель
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # для бинарной классификации
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Архитектура модели:")
model.summary()

# 7. ОБУЧАЕМ МОДЕЛЬ
print("\n7. ОБУЧАЕМ МОДЕЛЬ")

print("Начинаем обучение...")
history = model.fit(
    X_train_dense, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("✅ Обучение завершено!")

# 8. ОЦЕНКА МОДЕЛИ
print("\n8. ОЦЕНКА МОДЕЛИ")

# Предсказания на тестовых данных
y_pred_proba = model.predict(X_test_dense)
y_pred = (y_pred_proba > 0.5).astype(int)  # порог 0.5 для бинарной классификации

# Точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Подробный отчет
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# 9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("\n9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

# Графики обучения
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Точность
axes[0, 0].plot(history.history['accuracy'], label='Тренировочная')
axes[0, 0].plot(history.history['val_accuracy'], label='Валидационная')
axes[0, 0].set_title('Точность во время обучения')
axes[0, 0].set_xlabel('Эпоха')
axes[0, 0].set_ylabel('Точность')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Потери
axes[0, 1].plot(history.history['loss'], label='Тренировочные')
axes[0, 1].plot(history.history['val_loss'], label='Валидационные')
axes[0, 1].set_title('Потери во время обучения')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('Потери')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision и Recall
axes[1, 0].plot(history.history['precision'], label='Precision')
axes[1, 0].plot(history.history['recall'], label='Recall')
axes[1, 0].set_title('Precision и Recall на обучении')
axes[1, 0].set_xlabel('Эпоха')
axes[1, 0].set_ylabel('Значение')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Матрица ошибок')
axes[1, 1].set_xlabel('Предсказанные метки')
axes[1, 1].set_ylabel('Истинные метки')
axes[1, 1].set_xticklabels(['ham', 'spam'])
axes[1, 1].set_yticklabels(['ham', 'spam'])

plt.tight_layout()
#plt.show()

# 10. ТЕСТИРУЕМ НА НОВЫХ СООБЩЕНИЯХ
print("\n10. ТЕСТИРОВАНИЕ НА НОВЫХ СООБЩЕНИЯХ")

# Создаем тестовые сообщения
test_messages = [
    "Congratulations! You've won a free iPhone. Reply YES to claim.",
    "Can we meet tomorrow for coffee?",
    "URGENT: Your bank account needs verification. Click here now.",
    "Don't forget to buy milk on your way home.",
    "Earn $1000 weekly from home. No experience required!",
    "The meeting is scheduled for 3 PM today.",
    "Your package will be delivered between 2-4 PM.",
    "You have been selected for a free vacation! Call now."
]

print("Тестируем модель на новых сообщениях:")
print("-" * 60)

for i, message in enumerate(test_messages, 1):
    # Предобработка
    clean_message = simple_text_preprocess(message)
    
    # TF-IDF преобразование
    message_tfidf = tfidf_vectorizer.transform([clean_message]).toarray()
    
    # Предсказание
    prediction_proba = model.predict(message_tfidf)[0][0]
    prediction = "SPAM" if prediction_proba > 0.5 else "HAM"
    
    # Определяем цвет вывода
    color = '\033[91m' if prediction == "SPAM" else '\033[92m'  # красный для спама, зеленый для хама
    
    print(f"{i}. Сообщение: {message[:50]}...")
    print(f"   Предсказание: {color}{prediction}\033[0m (уверенность: {prediction_proba:.2%})")
    print()

# 11. АНАЛИЗ ВАЖНЫХ ПРИЗНАКОВ
print("\n11. АНАЛИЗ ВАЖНЫХ СЛОВ")

# Получаем веса из модели
feature_names = tfidf_vectorizer.get_feature_names_out()
weights = model.layers[0].get_weights()[0]  # веса первого слоя

# Суммируем абсолютные значения весов для каждого признака
feature_importance = np.abs(weights).sum(axis=1)

# Сортируем по важности
important_indices = np.argsort(feature_importance)[-20:]  # топ-20 признаков
important_features = feature_names[important_indices]
importance_values = feature_importance[important_indices]

print("Топ-20 самых важных слов для классификации:")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(important_features)), importance_values)
plt.yticks(range(len(important_features)), important_features)
plt.xlabel('Важность')
plt.title('Самые важные слова для определения спама')
plt.tight_layout()

# 12. СОХРАНЕНИЕ МОДЕЛИ
print("\n12. СОХРАНЕНИЕ МОДЕЛИ И КОМПОНЕНТОВ")

# Сохраняем модель
model.save('spam_detector_model.h5')
print("✅ Модель сохранена как 'spam_detector_model.h5'")

# Сохраняем TF-IDF векторизатор
import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✅ TF-IDF векторизатор сохранен как 'tfidf_vectorizer.pkl'")

# Сохраняем label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder сохранен как 'label_encoder.pkl'")

# 13. ЗАГРУЗКА И ИСПОЛЬЗОВАНИЕ МОДЕЛИ (пример)
print("\n13. КАК ИСПОЛЬЗОВАТЬ МОДЕЛЬ ДЛЯ НОВЫХ СООБЩЕНИЙ")

def predict_spam(message):
    """Функция для предсказания спама"""
    # Предобработка
    clean_msg = simple_text_preprocess(message)
    
    # TF-IDF преобразование
    msg_tfidf = tfidf_vectorizer.transform([clean_msg]).toarray()
    
    # Предсказание
    proba = model.predict(msg_tfidf)[0][0]
    is_spam = proba > 0.5
    
    return {
        'message': message,
        'is_spam': bool(is_spam),
        'probability': float(proba),
        'class': 'SPAM' if is_spam else 'HAM'
    }

# Тестируем функцию
test_msg = "WINNER! Claim your prize now!"
result = predict_spam(test_msg)
print(f"\nТест функции предсказания:")
print(f"Сообщение: {result['message']}")
print(f"Результат: {result['class']}")
print(f"Вероятность: {result['probability']:.2%}")

# 14. ИТОГИ
print("\n" + "=" * 60)
print("ИТОГИ И ВЫВОДЫ")
print("=" * 60)

print(f"\n🎯 РЕЗУЛЬТАТЫ:")
print(f"1. Точность модели: {accuracy * 100:.2f}%")
print(f"2. Обучено на {len(X_train)} сообщениях")
print(f"3. Протестировано на {len(X_test)} сообщениях")

print("\n🔍 КЛЮЧЕВЫЕ СЛОВА ДЛЯ ОПРЕДЕЛЕНИЯ СПАМА:")
for i, word in enumerate(important_features[-5:][::-1], 1):
    print(f"   {i}. {word}")

print("\n" + "=" * 60)
print("ПОЗДРАВЛЯЮ! Вы создали детектор спама!")
print("=" * 60)