import pandas as pd
from sklearn.preprocessing import StandardScaler

# Все столбцы, которые будут являться признаками для модели
feature_cols = [
    "Age",
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Total_Score",
    "Study_Hours_per_Week",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night"
]

def preprocess_data(input_file, output_file, scaler=None, fit_scaler=False):
    data = pd.read_csv(input_file)
    X = data[feature_cols].copy()
    y = data["Grade"].values
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # DataFrame с признаками и целевой переменной, которая будет предсказываться (Grade)
    df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
    df_processed["Grade"] = y
    
    df_processed.to_csv(output_file, index=False)
    return scaler

# Обработка обучающих данных
scaler = preprocess_data('./train/data.csv', './train/preprocessed_data.csv', fit_scaler=True)

# Обработка тестовых данных с использованием уже обученного scaler
preprocess_data('./test/data.csv', './test/preprocessed_data.csv', scaler=scaler, fit_scaler=False)
