from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Send POST JSON data to predict churn"})

    data = request.json
    df_input = pd.DataFrame([data])

    df_input[NUMERIC_FEATURES] = df_input[NUMERIC_FEATURES].fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col in df_input.columns:
            df_input[col] = (
                df_input[col]
                .fillna("Unknown")
                .astype("category")
                .cat.codes
            )

    X_input = df_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prob = model.predict_proba(X_input)[:, 1][0]
    churn_pred = 1 if prob > 0.67 else 0

    return jsonify({
        "churn_prob": float(prob),
        "prediction": churn_pred,
        "threshold": 0.67
    })


if __name__ == "__main__":
    logger.info("Loading model...")
    model = joblib.load("models/retrained_model.pkl")

    logger.info("Flask server running at http://localhost:5050/predict")
    app.run(port=5050, debug=False, threaded=True)
