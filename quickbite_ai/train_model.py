import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
from langchain_ollama import ChatOllama


np.random.seed(42)

rows = 120
distance_km = np.random.uniform(1, 15, rows)
num_items = np.random.randint(1, 8, rows)
rain_flag = np.random.randint(0, 2, rows)
noise = np.random.normal(0, 3, rows)

delivery_time_min = 18 + distance_km * 4 + num_items * 2.5 + rain_flag * 10 + noise

df = pd.DataFrame({
    "distance_km": distance_km,
    "num_items": num_items,
    "rain_flag": rain_flag,
    "delivery_time_min": delivery_time_min
})

X = df[["distance_km", "num_items", "rain_flag"]]
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
print(f"RMSE: {rmse:.2f}")

joblib.dump(model, "delivery_model.joblib")
size_kb = os.path.getsize("delivery_model.joblib") / 1024
print(f"Model saved as delivery_model.joblib ({size_kb:.2f} KB)")

sample = [[5, 2, 0]]
original_pred = model.predict(sample)[0]
loaded_model = joblib.load("delivery_model.joblib")
loaded_pred = loaded_model.predict(sample)[0]

if abs(original_pred - loaded_pred) < 0.0001:
    print("PASS: Reloaded model prediction matches original model.")
else:
    print("FAIL: Reloaded model prediction does not match.")
