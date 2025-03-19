import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

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

test_data = pd.read_csv('./test/preprocessed_data.csv')
X_test = test_data[feature_cols]
y_test = test_data["Grade"]

with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction and evaluation using classification metrics
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)