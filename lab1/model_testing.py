import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

feature_cols = [
    "Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
    "Quizzes_Avg", "Participation_Score", "Projects_Score",
    "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

test_data = pd.read_csv('./test/preprocessed_data.csv')
X_test = test_data[feature_cols]
y_test = test_data["Total_Score"]

model = joblib.load('/model.pkl')

# Предсказание и оценка модели
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")