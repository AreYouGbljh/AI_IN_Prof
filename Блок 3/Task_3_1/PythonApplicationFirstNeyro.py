import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml  # для загрузки данных
from sklearn.model_selection import train_test_split  # для разделения данных
from sklearn.ensemble import RandomForestClassifier  # наш классификатор
from sklearn.metrics import accuracy_score  # для оценки точности

# 1. ЗАГРУЖАЕМ ДАННЫЕ
# MNIST - это база картинок рукописных цифр
print("Загружаем базу цифр MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# X - это картинки (70000 штук, каждая 28x28 пикселей = 784 пикселя)
# y - это правильные ответы (какая цифра на картинке)
X, y = mnist.data, mnist.target

print(f"Всего картинок: {X.shape[0]}")
print(f"Каждая картинка: {X.shape[1]} пикселей")

# 2. ПОДГОТАВЛИВАЕМ ДАННЫЕ
# Превращаем ответы в числа (были строки '0', '1', ...)
y = y.astype(int)

# Делим данные: 80% для обучения, 20% для проверки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nКартинок для обучения: {X_train.shape[0]}")
print(f"Картинок для проверки: {X_test.shape[0]}")

# 3. ПОКАЗЫВАЕМ, КАК ВЫГЛЯДЯТ ДАННЫЕ
print("\nСмотрим на несколько примеров:")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    row = i // 5  # номер строки
    col = i % 5   # номер столбца
    
    # Берем i-ю картинку и показываем ее
    img = X_train.iloc[i].values.reshape(28, 28)  # превращаем в квадрат 28x28
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"Цифра: {y_train.iloc[i]}")
    axes[row, col].axis('off')  # убираем оси

plt.tight_layout()
plt.show()

# 4. СОЗДАЕМ И ОБУЧАЕМ МОДЕЛЬ
print("\nСоздаем модель Random Forest...")

# Random Forest - это много маленьких решающих деревьев, которые голосуют
# n_estimators=100 - создаем 100 деревьев
# random_state=42 - чтобы результаты повторялись
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Начинаем обучение модели...")
model.fit(X_train, y_train)  # обучаем на тренировочных данных
print("Обучение завершено!")

# 5. ПРОВЕРЯЕМ, КАК МОДЕЛЬ РАБОТАЕТ
print("\nПроверяем точность модели...")

# Делаем предсказания для тестовых данных
y_pred = model.predict(X_test)

# Считаем точность: сколько из 100% угадано правильно
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.1f}%")

# 6. ПОКАЗЫВАЕМ НЕСКОЛЬКО ПРЕДСКАЗАНИЙ
print("\nСмотрим на несколько предсказаний:")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    row = i // 5
    col = i % 5
    
    # Берем i-ю тестовую картинку
    img = X_test.iloc[i].values.reshape(28, 28)
    
    # Предсказываем, какая это цифра
    prediction = model.predict([X_test.iloc[i]])[0]
    true_label = y_test.iloc[i]  # настоящая цифра
    
    # Раскрашиваем заголовок: зеленый если угадали, красный если нет
    color = 'green' if prediction == true_label else 'red'
    
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"Настоящая: {true_label}\nПредсказано: {prediction}", color=color)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# 7. СМОТРИМ, ГДЕ МОДЕЛЬ ОШИБАЛАСЬ
print("\nИщем ошибки модели...")

# Находим все неправильные предсказания
errors = []
for i in range(len(y_test)):
    if y_pred[i] != y_test.iloc[i]:
        errors.append(i)

if errors:
    print(f"Найдено {len(errors)} ошибок")
    print("Показываем первые 5 ошибок:")
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(min(5, len(errors))):
        idx = errors[i]
        img = X_test.iloc[idx].values.reshape(28, 28)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Настоящая: {y_test.iloc[idx]}\nПредсказано: {y_pred[idx]}", color='red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("Ошибок не найдено (маловероятно, но возможно)")

# 8. ПРОСТОЙ ТЕСТ: ПРЕДСКАЗАНИЕ ДЛЯ ОДНОЙ ЦИФРЫ
print("\nТестируем на одной случайной цифре:")

# Выбираем случайную цифру из тестовой выборки
random_idx = np.random.randint(0, len(X_test))

# Показываем эту цифру
test_image = X_test.iloc[random_idx].values.reshape(28, 28)
plt.imshow(test_image, cmap='gray')
plt.axis('off')

# Предсказываем
true_digit = y_test.iloc[random_idx]
predicted_digit = model.predict([X_test.iloc[random_idx]])[0]

print(f"На картинке цифра: {true_digit}")
print(f"Модель сказала: {predicted_digit}")

if true_digit == predicted_digit:
    print("✅ Модель угадала правильно!")
else:
    print("❌ Модель ошиблась")

plt.title(f"Настоящая: {true_digit}, Предсказано: {predicted_digit}", 
          color='green' if true_digit == predicted_digit else 'red')
plt.show()

# 9. ИТОГИ
print("\n" + "="*50)
print("ИТОГИ:")
print("="*50)
print("1. Мы распознали рукописные цифры БЕЗ нейросетей!")
print("2. Использовали Random Forest (ансамбль деревьев решений)")
print(f"3. Точность: {accuracy * 100:.1f}%")
print("4. Это значит: из 100 цифр модель правильно узнает примерно", int(accuracy * 100))
print("\nГлавная мысль: для многих задач не нужны сложные нейросети.")
print("Часто достаточно простых методов машинного обучения!")