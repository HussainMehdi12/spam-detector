from flask import Flask, render_template, request
import csv
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset (NO pandas)
messages = []
labels = []

file_path = os.path.join(os.path.dirname(__file__), "spam.csv")

with open(file_path, encoding="latin-1") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    for row in reader:
        labels.append(row[0])
        messages.append(row[1])

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        message = request.form["message"]

        vector = vectorizer.transform([message])
        result = model.predict(vector)
        prob = model.predict_proba(vector)[0][1]
        confidence = round(prob * 100, 2)

        if result[0] == "spam":
            prediction = "🚫 SPAM"
        else:
            prediction = "✅ HAM"

    return render_template("index.html", prediction=prediction, confidence=confidence)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)