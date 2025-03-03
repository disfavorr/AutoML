import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

# Создание директорий для хранения данных
os.makedirs('lab1/train', exist_ok=True)
os.makedirs('lab1/test', exist_ok=True)

# Функция для добавления шума в данные
def add_noise(data, noise_level=0.1):
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

# Генерация синтетического набора данных
X, y = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=42)

# Добавление аномалий в данные
X[:10] = add_noise(X[:10], noise_level=5)
y[:10] = add_noise(y[:10], noise_level=50)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Сохранение данных в CSV файлы
pd.DataFrame({'feature': X_train.flatten(), 'target': y_train}).to_csv('lab1/train/data.csv', index=False)
pd.DataFrame({'feature': X_test.flatten(), 'target': y_test}).to_csv('lab1/test/data.csv', index=False)