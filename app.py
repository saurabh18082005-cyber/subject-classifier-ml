import os
import pickle
from flask import Flask, request, jsonify, render_template

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask app configuration
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
    template_folder=os.path.join(BASE_DIR, "templates")
)

print("Starting Subject Classifier API...")

# Load trained ML model
model_path = os.path.join(BASE_DIR, "subject_classifier.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

# IMPORTANT: Render-compatible server start
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app
