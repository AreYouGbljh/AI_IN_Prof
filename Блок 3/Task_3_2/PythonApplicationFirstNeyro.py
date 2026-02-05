import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("=" * 60)
print("СОЗДАНИЕ УЛУЧШЕННЫХ ДАННЫХ ДЛЯ НЕЙРОСЕТИ")
print("=" * 60)

# Устанавливаем seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

# 1. ФУНКЦИИ ДЛЯ СОЗДАНИЯ ЦИФР
print("\nСоздаем четкие паттерны для каждой цифры...")

def create_digit_0():
    """Создает цифру 0 - круг/овал"""
    img = np.zeros((28, 28))
    for r in range(28):
        for c in range(28):
            dist = ((r-14)/1.5)**2 + (c-14)**2  # овал (немного вытянут по вертикали)
            if 25 < dist < 100:  # кольцо
                img[r, c] = 1.0
    return img

def create_digit_1():
    """Создает цифру 1 - вертикальная линия с наклоном"""
    img = np.zeros((28, 28))
    # Вертикальная линия с небольшим наклоном
    for i in range(5, 23):
        img[i, 13:16] = 1.0  # основная линия
    # Небольшой крючок сверху
    img[5:8, 10:13] = 1.0
    return img

def create_digit_2():
    """Создает цифру 2"""
    img = np.zeros((28, 28))
    # Верхняя горизонталь
    img[6:9, 8:20] = 1.0
    # Правая вертикаль вниз
    img[6:15, 18:20] = 1.0
    # Средняя горизонталь
    img[13:16, 8:20] = 1.0
    # Левая вертикаль вниз
    img[13:22, 8:10] = 1.0
    # Нижняя горизонталь
    img[19:22, 8:20] = 1.0
    return img

def create_digit_3():
    """Создает цифру 3"""
    img = np.zeros((28, 28))
    # Верхняя горизонталь
    img[6:9, 8:20] = 1.0
    # Средняя горизонталь
    img[13:16, 8:20] = 1.0
    # Нижняя горизонталь
    img[19:22, 8:20] = 1.0
    # Правая вертикаль (общая для всех частей)
    img[6:22, 18:20] = 1.0
    return img

def create_digit_4():
    """Создает цифру 4"""
    img = np.zeros((28, 28))
    # Левая вертикаль
    img[6:15, 8:10] = 1.0
    # Горизонталь
    img[13:16, 8:20] = 1.0
    # Правая вертикаль
    img[6:22, 18:20] = 1.0
    return img

def create_digit_5():
    """Создает цифру 5"""
    img = np.zeros((28, 28))
    # Верхняя горизонталь
    img[6:9, 8:20] = 1.0
    # Левая вертикаль вниз
    img[6:15, 8:10] = 1.0
    # Средняя горизонталь
    img[13:16, 8:20] = 1.0
    # Правая вертикаль вниз
    img[13:22, 18:20] = 1.0
    # Нижняя горизонталь
    img[19:22, 8:20] = 1.0
    return img

def create_digit_6():
    """Создает цифру 6"""
    img = np.zeros((28, 28))
    # Круг/овал
    for r in range(28):
        for c in range(28):
            dist = ((r-16)/1.2)**2 + (c-14)**2
            if 20 < dist < 80:
                img[r, c] = 1.0
    # Закрываем верхнюю часть
    img[6:10, :] = 0
    # Добавляем вертикальную линию слева
    img[10:20, 8:10] = 1.0
    return img

def create_digit_7():
    """Создает цифру 7"""
    img = np.zeros((28, 28))
    # Верхняя горизонталь
    img[6:9, 8:20] = 1.0
    # Диагональ вниз вправо
    for i in range(9, 23):
        img[i, 10 + (i-9)//2] = 1.0
        img[i, 11 + (i-9)//2] = 1.0
    return img

def create_digit_8():
    """Создает цифру 8 - два круга"""
    img = np.zeros((28, 28))
    # Верхний круг
    for r in range(28):
        for c in range(28):
            dist = ((r-10)/1.2)**2 + (c-14)**2
            if 15 < dist < 40:
                img[r, c] = 1.0
    # Нижний круг
    for r in range(28):
        for c in range(28):
            dist = ((r-18)/1.2)**2 + (c-14)**2
            if 15 < dist < 40:
                img[r, c] = 1.0
    # Соединяем круги
    img[10:18, 13:15] = 1.0
    return img

def create_digit_9():
    """Создает цифру 9"""
    img = np.zeros((28, 28))
    # Круг/овал (как у 6, но перевернутый)
    for r in range(28):
        for c in range(28):
            dist = ((r-12)/1.2)**2 + (c-14)**2
            if 20 < dist < 80:
                img[r, c] = 1.0
    # Закрываем нижнюю часть
    img[16:22, :] = 0
    # Добавляем вертикальную линию справа
    img[8:16, 18:20] = 1.0
    return img

# Словарь функций для создания цифр
digit_functions = {
    0: create_digit_0,
    1: create_digit_1,
    2: create_digit_2,
    3: create_digit_3,
    4: create_digit_4,
    5: create_digit_5,
    6: create_digit_6,
    7: create_digit_7,
    8: create_digit_8,
    9: create_digit_9
}

# 2. СОЗДАЕМ ДАННЫЕ С ВАРИАЦИЯМИ
print("Добавляем вариации и шум для реалистичности...")

def create_variation(base_digit, variation_level=0.3):
    """Создает вариацию базовой цифры"""
    img = base_digit.copy()
    
    # Добавляем случайные вариации
    if np.random.random() > 0.5:
        # Сдвиг
        shift_x, shift_y = np.random.randint(-2, 3, 2)
        img = np.roll(img, shift_x, axis=0)
        img = np.roll(img, shift_y, axis=1)
    
    if np.random.random() > 0.3:
        # Небольшое вращение/искажение
        for i in range(28):
            img[i, :] = np.roll(img[i, :], np.random.randint(-1, 2))
    
    # Добавляем шум
    noise = np.random.randn(28, 28) * variation_level
    img = img + noise
    
    # Обрезаем значения
    img = np.clip(img, 0, 1)
    
    return img

# 3. СОЗДАЕМ ТРЕНИРОВОЧНЫЕ И ТЕСТОВЫЕ ДАННЫЕ
print("Генерируем тренировочные и тестовые данные...")

num_train_per_digit = 600  # по 600 примеров каждой цифры
num_test_per_digit = 100   # по 100 примеров каждой цифры

x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

for digit in range(10):
    base_digit = digit_functions[digit]()
    
    # Тренировочные данные
    for _ in range(num_train_per_digit):
        varied_digit = create_variation(base_digit, variation_level=0.2)
        x_train_list.append(varied_digit)
        y_train_list.append(digit)
    
    # Тестовые данные
    for _ in range(num_test_per_digit):
        varied_digit = create_variation(base_digit, variation_level=0.1)
        x_test_list.append(varied_digit)
        y_test_list.append(digit)

# Преобразуем в numpy массивы
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

# Перемешиваем данные
train_indices = np.random.permutation(len(x_train))
test_indices = np.random.permutation(len(x_test))

x_train = x_train[train_indices]
y_train = y_train[train_indices]
x_test = x_test[test_indices]
y_test = y_test[test_indices]

print(f"\n✅ Данные созданы!")
print(f"Тренировочные данные: {len(x_train)} примеров")
print(f"Тестовые данные: {len(x_test)} примеров")
print(f"Баланс классов: {np.bincount(y_train)}")

# 4. ВИЗУАЛИЗАЦИЯ ДАННЫХ
print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ СОЗДАННЫХ ЦИФР")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit in range(10):
    row = digit // 5
    col = digit % 5
    
    # Находим первый пример этой цифры
    idx = np.where(y_train == digit)[0][0]
    
    axes[row, col].imshow(x_train[idx], cmap='gray')
    axes[row, col].set_title(f"Цифра: {digit}")
    axes[row, col].axis('off')

plt.suptitle("Примеры созданных цифр", fontsize=14)
plt.tight_layout()
plt.show()

# 5. СОЗДАЕМ И ОБУЧАЕМ МОДЕЛЬ
print("\n" + "=" * 60)
print("СОЗДАНИЕ И ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
print("=" * 60)

# Улучшенная архитектура
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),  # больше нейронов
    tf.keras.layers.Dropout(0.3),  # предотвращаем переобучение
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компилируем с оптимизированными параметрами
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Архитектура модели:")
model.summary()

# 6. ОБУЧЕНИЕ С КОНТРОЛЕМ ПЕРЕОБУЧЕНИЯ
print("\nНачинаем обучение...")

# Callback для уменьшения learning rate
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Callback для ранней остановки
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Разделяем тренировочные данные на train/validation
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Обучение: {len(x_train_final)} примеров")
print(f"Валидация: {len(x_val)} примеров")

# Обучаем модель
history = model.fit(
    x_train_final, y_train_final,
    epochs=20,  # больше эпох
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

print("✅ Обучение завершено!")

# 7. ОЦЕНКА МОДЕЛИ
print("\n" + "=" * 60)
print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность на тестовых данных: {test_accuracy * 100:.2f}%")

if test_accuracy > 0.9:
    print("🎉 ОТЛИЧНО! Точность более 90%! 🎉")
elif test_accuracy > 0.8:
    print("👍 ХОРОШО! Точность более 80%!")
else:
    print("Можно улучшить! Попробуйте обучить больше эпох.")

# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Графики обучения
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Точность
axes[0].plot(history.history['accuracy'], label='Тренировочная точность')
axes[0].plot(history.history['val_accuracy'], label='Валидационная точность')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Точность')
axes[0].set_title('Точность во время обучения')
axes[0].legend()
axes[0].grid(True)

# Потери
axes[1].plot(history.history['loss'], label='Тренировочные потери')
axes[1].plot(history.history['val_loss'], label='Валидационные потери')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Потери')
axes[1].set_title('Потери во время обучения')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 9. ТЕСТИРОВАНИЕ НА КОНКРЕТНЫХ ПРИМЕРАХ
print("\nТестируем на случайных примерах...")

num_examples = 10
indices = np.random.choice(len(x_test), num_examples, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, idx in enumerate(indices):
    row = i // 5
    col = i % 5
    
    img = x_test[idx]
    true_label = y_test[idx]
    
    # Предсказание
    img_array = img.reshape(1, 28, 28)
    prediction = model.predict(img_array, verbose=0)[0]
    predicted_label = np.argmax(prediction)
    confidence = prediction[predicted_label]
    
    # Визуализация
    axes[row, col].imshow(img, cmap='gray')
    
    color = 'green' if predicted_label == true_label else 'red'
    axes[row, col].set_title(
        f"Настоящая: {true_label}\nПредсказано: {predicted_label}\nУверенность: {confidence:.1%}",
        color=color,
        fontsize=9
    )
    axes[row, col].axis('off')

plt.suptitle(f"Тестирование модели (Общая точность: {test_accuracy*100:.1f}%)", fontsize=14)
plt.tight_layout()
plt.show()

# 10. МАТРИЦА ОШИБОК
print("\n" + "=" * 60)
print("АНАЛИЗ ОШИБОК МОДЕЛИ")
print("=" * 60)

# Делаем предсказания для всех тестовых данных
y_pred = model.predict(x_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

# Считаем ошибки
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Матрица ошибок (Confusion Matrix)')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.show()

# Отчет по классификации
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred_labels, digits=3))

# 11. СОХРАНЕНИЕ МОДЕЛИ И ИТОГИ
print("\n" + "=" * 60)
print("ИТОГИ")
print("=" * 60)

# Сохраняем модель
model.save('improved_digit_model.h5')
print("✅ Модель сохранена как 'improved_digit_model.h5'")

print(f"\n🎯 ИТОГОВАЯ ТОЧНОСТЬ: {test_accuracy * 100:.2f}%")
print("\n📈 ЧТО БЫЛО УЛУЧШЕНО:")
print("1. Созданы четкие, узнаваемые паттерны для каждой цифры")
print("2. Добавлены вариации (сдвиги, шум) для реалистичности")
print("3. Увеличено количество данных (6000 тренировочных примеров)")
print("4. Улучшена архитектура сети (больше слоев, Dropout)")
print("5. Добавлены callback'и для оптимизации обучения")
print("6. Использована стратификация для баланса классов")

print("\n🚀 ДЛЯ ДАЛЬНЕЙШЕГО УЛУЧШЕНИЯ:")
print("1. Увеличить количество тренировочных данных")
print("2. Добавить больше аугментаций (повороты, масштабирование)")
print("3. Использовать сверточные слои (Conv2D)")
print("4. Настроить гиперпараметры (learning rate, размер батча)")

print("\n" + "=" * 60)
print("ПОЗДРАВЛЯЮ! Вы создали эффективную нейросеть!")
print("=" * 60)