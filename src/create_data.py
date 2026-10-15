
import numpy as np
import pandas as pd
from pathlib import Path

# Make the dataset reproducible
np.random.seed(42)

# Number of students
n = 1000

# Generate student features
study_hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_score = np.random.uniform(30, 100, n)
sleep_hours = np.random.uniform(4, 10, n)

# Generate exam scores using a simulated relationship
noise = np.random.normal(0, 5, n)

exam_score = (
    3.5 * study_hours
    + 0.20 * attendance
    + 0.35 * previous_score
    + 1.5 * sleep_hours
    + noise
)

# Keep scores within the 0–100 range
exam_score = np.clip(exam_score, 0, 100)

# Create a DataFrame
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "exam_score": exam_score
})

# Create the data directory and save the CSV
Path("data").mkdir(exist_ok=True)

df.to_csv("data/student_scores.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nSummary statistics:")
print(df.describe())
