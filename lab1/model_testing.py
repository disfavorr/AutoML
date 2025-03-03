import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка предобработанных данных
test_data = pd.read_csv('lab1/test/preprocessed_data.csv')

# Разделение признаков и целевой переменной
X_test = test_data[['feature']]
y_test = test_data['target']

# Загрузка обученной модели
model = joblib.load('lab1/model.pkl')

# Предсказание на тестовых данных
predictions = model.predict(X_test)

# Вычисление метрики
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')