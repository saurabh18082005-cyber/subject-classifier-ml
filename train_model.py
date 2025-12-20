import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("training_data.csv")

X = data["text"]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", MultinomialNB(alpha=0.1))
])

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
with open("subject_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as subject_classifier.pkl")

