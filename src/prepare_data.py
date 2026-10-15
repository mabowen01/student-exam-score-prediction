
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
# Load the dataset
df = pd.read_csv("data/student_scores.csv")

# Separate input features (X) and target (y)
X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]
]

y = df["exam_score"]

# Split off the test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Split the remaining data into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    random_state=42
)

# Scale the input features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# Convert features to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train_scaled, dtype=torch.float32
)
X_val_tensor = torch.tensor(
    X_val_scaled, dtype=torch.float32
)
X_test_tensor = torch.tensor(
    X_test_scaled, dtype=torch.float32
)

# Convert targets to PyTorch tensors
y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

print("\nPyTorch tensor shapes:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_val:", X_val_tensor.shape)
print("y_val:", y_val_tensor.shape)
print("X_test:", X_test_tensor.shape)
print("y_test:", y_test_tensor.shape)
# Print dataset sizes
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

print("\nScaled training shape:", X_train_scaled.shape)
print("Training target shape:", y_train.shape)

print("\nFirst five scaled training rows:")
print(X_train_scaled[:5])
