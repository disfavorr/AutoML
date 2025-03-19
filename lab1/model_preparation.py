import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

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

train_data = pd.read_csv('./train/preprocessed_data.csv')
X_train = train_data[feature_cols]
y_train = train_data["Grade"]

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs('./model', exist_ok=True)

with open("./model/model.pkl", "wb") as f:
    pickle.dump(model, f)
