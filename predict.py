
import torch
import torch.nn as nn
import joblib

# 1. Rebuild the same neural network architecture
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 2. Load the saved model parameters
state_dict = torch.load(
    "models/student_model.pth",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# 3. Load the scaler used during training
scaler = joblib.load("models/scaler.joblib")

# 4. Enter student information
study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage (0-100): "))
previous_score = float(input("Previous score (0-100): "))
sleep_hours = float(input("Sleep hours per day: "))

# 5. Prepare input in the same feature order as training
student_data = [[
    study_hours,
    attendance,
    previous_score,
    sleep_hours
]]

# 6. Apply the saved scaler
student_scaled = scaler.transform(student_data)

# 7. Predict without calculating gradients
with torch.no_grad():
    input_tensor = torch.tensor(
        student_scaled,
        dtype=torch.float32
    )
    prediction = model(input_tensor)

score = prediction.item()
score = max(0, min(100, score))

print(f"\nPredicted exam score: {score:.2f}/100")
