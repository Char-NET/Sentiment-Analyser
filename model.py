import re
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean(text):
    text = str(text).lower()
    return re.sub(r'[^a-z\s]', '', text)

def predict_sentiment(text):
    text = clean(text)
    vec = vectorizer.transform([text])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    index = probs.argmax()
    confidence = probs[index]
    prediction = classes[index]

    if confidence < 0.6:
        prediction = "neutral"

    return prediction, round(confidence * 100, 2)