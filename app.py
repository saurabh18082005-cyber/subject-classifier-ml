import os
import pickle
from flask import Flask, request, jsonify, render_template, send_from_directory

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask app with explicit static & template folders
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
    template_folder=os.path.join(BASE_DIR, "templates")
)

print("Starting Subject Classifier API...")

# Load trained model
with open(os.path.join(BASE_DIR, "subject_classifier.pkl"), "rb") as f:
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

# 🔥 MANUAL VIDEO TEST ROUTE (VERY IMPORTANT)
@app.route("/test-video")
def test_video():
    return send_from_directory(
        os.path.join(BASE_DIR, "static"),
        "bg.mp4"
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("port",
    1000))
    app.run(host="0.0.0.0.", port=port)
