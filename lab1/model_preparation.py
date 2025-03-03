import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка предобработанных данных
train_data = pd.read_csv('lab1/train/preprocessed_data.csv')

# Разделение признаков и целевой переменной
X_train = train_data[['feature']]
y_train = train_data['target']

# Инициализация и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение обученной модели
joblib.dump(model, 'lab1/model.pkl')
