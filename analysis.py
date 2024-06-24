import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = joblib.load('political_leaning_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def analyze_text(text):
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)
    return prediction[0]