import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

AUTHOR_NAME = "Pawan"
ROLL_NO = "2022BCS0061"

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/WineQT.csv")

# Features and target
X = data.drop(["quality", "Id"], axis=1)
y = data["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Save model WITH metadata
model_data = {
    "model": model,
    "author": AUTHOR_NAME,
    "roll_no": ROLL_NO
}
joblib.dump(model_data, "output/model.pkl")

# Save metrics WITH your details
metrics = {
    "author": AUTHOR_NAME,
    "roll_no": ROLL_NO,
    "mse": mse,
    "r2": r2
}

with open("output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save a log file inside artifacts
with open("output/run_log.txt", "w") as f:
    f.write(f"Author: {AUTHOR_NAME}\n")
    f.write(f"Roll No: {ROLL_NO}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"R2 Score: {r2}\n")

print(f"Author: {AUTHOR_NAME}")
print(f"Roll No: {ROLL_NO}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
