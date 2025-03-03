import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Загрузка данных
train_data = pd.read_csv('lab1/train/data.csv')
test_data = pd.read_csv('lab1/test/data.csv')

# Инициализация стандартизатора
scaler = StandardScaler()

# Обучение стандартизатора на обучающих данных и трансформация
train_data[['feature']] = scaler.fit_transform(train_data[['feature']])

# Трансформация тестовых данных
test_data[['feature']] = scaler.transform(test_data[['feature']])

# Сохранение предобработанных данных
train_data.to_csv('lab1/train/preprocessed_data.csv', index=False)
test_data.to_csv('lab1/test/preprocessed_data.csv', index=False)

# Сохранение обученного стандартизатора
joblib.dump(scaler, 'lab1/scaler.pkl')
