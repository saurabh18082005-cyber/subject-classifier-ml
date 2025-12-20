import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

print("Starting Subject Classifier API...")

# LOAD MODEL
with open("subject_classifier.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
