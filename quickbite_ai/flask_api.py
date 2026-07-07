from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
MODEL_PATH = "delivery_model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    distance_km = float(data.get("distance_km", 5))
    item_count = int(data.get("item_count", 2))
    rain_flag = int(data.get("rain_flag", 0))

    if model:
        prediction = model.predict([[distance_km, item_count, rain_flag]])[0]
    else:
        prediction = 20 + distance_km * 4 + item_count * 2 + rain_flag * 10

    return jsonify({"delivery_time_min": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
