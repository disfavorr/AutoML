import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs('./train', exist_ok=True)
os.makedirs('./test', exist_ok=True)

data = pd.read_csv('Students_Grading_Dataset.csv')

numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns # Преобразование числовых данных
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median()) # Заполнение пропущенных значений медианными значениями

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('./train/data.csv', index=False)
test_data.to_csv('./test/data.csv', index=False)