
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/student_scores.csv")

X = df[
    ["study_hours", "attendance", "previous_score", "sleep_hours"]
].values

y = df["exam_score"].values.reshape(-1, 1)

# ==========================================
# 2. Split data
# 60% training, 20% validation, 20% testing
# ==========================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==========================================
# 3. Scale input features
# Fit the scaler on training data only
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==========================================
# 4. Convert data to PyTorch tensors
# ==========================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 5. Build the neural network
# ==========================================

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

print("\nNeural Network Architecture:")
print(model)

# ==========================================
# 6. Define loss function and optimizer
# ==========================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 2000

train_losses = []
val_losses = []

# ==========================================
# 7. Train the neural network
# ==========================================

for epoch in range(epochs):

    # Training mode
    model.train()

    # Forward propagation
    predictions = model(X_train)

    # Calculate training loss
    loss = loss_fn(predictions, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_fn(val_predictions, y_val)

    # Store losses for visualization
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Print progress every 25 epochs
    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f}"
        )

print("\nTraining completed!")

# ==========================================
# 8. Evaluate on the test dataset
# ==========================================

model.eval()

with torch.no_grad():
    test_predictions = model(X_test)

y_true = y_test.numpy().flatten()
y_pred = test_predictions.numpy().flatten()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)

print("\n========== Test Results ==========")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# ==========================================
# 9. Save trained model and scaler
# ==========================================

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/student_model.pth"
)

joblib.dump(
    scaler,
    "models/scaler.joblib"
)

print("\nModel and scaler saved successfully!")

# ==========================================
# 10. Plot training and validation loss
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, epochs + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save graph to a file
plt.savefig("models/loss_curve.png", dpi=150)

print("Loss graph saved to models/loss_curve.png")

# Display graph
plt.show()

# ==========================================
# 11. Actual vs Predicted Scores
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_true,
    y_pred,
    alpha=0.7,
    label="Predictions"
)

# Perfect prediction reference line
min_score = min(y_true.min(), y_pred.min())
max_score = max(y_true.max(), y_pred.max())

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--",
    label="Perfect Predictions"
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save graph
plt.savefig("models/actual_vs_predicted.png", dpi=150)

print("Actual vs predicted graph saved!")

plt.show()
