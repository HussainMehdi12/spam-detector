from flask import Flask, render_template, request
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setup
app = Flask(__name__)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['message'] = data['message'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(data['message'])
y = data['label']


# Train model
model = MultinomialNB()
model.fit(X, y)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Clean and transform
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    # Prediction
    result = model.predict(vector)

    # Confidence calculation
    prob = model.predict_proba(vector)[0][1]
    confidence = round(prob * 100, 2)

    # Label
    if result[0] == 1:
        prediction = "🚫 SPAM"
    else:
        prediction = "✅ HAM"

    # ✅ THIS LINE MUST BE INSIDE FUNCTION
    return render_template('index.html', prediction=prediction, confidence=confidence)
# Run app
if __name__ == "__main__":
    app.run(debug=True)