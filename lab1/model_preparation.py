import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

feature_cols = [
    "Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
    "Quizzes_Avg", "Participation_Score", "Projects_Score",
    "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

train_data = pd.read_csv('./train/preprocessed_data.csv')
X_train = train_data[feature_cols]
y_train = train_data["Total_Score"]

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, '/model.pkl')
