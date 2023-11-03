from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import joblib
import pandas as pd

# Charger votre modèle Spacy et initialisez le vectorizer
nlp = spacy.load('en_core_web_sm')
vectorizer = TfidfVectorizer()

# Charger le modèle d'IA à l'aide de joblib
model = joblib.load('model.joblib')

# Charger le vectorizer à l'aide de joblib
vectorizer = joblib.load('vectorizer.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    comment = request.json.get('comment')
    # Traitez le commentaire de la même manière que vous l'avez fait pour votre modèle
    spacy_comment = nlp(comment, disable=["parser", "tagger", "ner", "textcat"])
    treated_tokens = [w.text for w in spacy_comment if w.is_alpha and not w.is_stop]
    treated_comment = " ".join(treated_tokens)
    
    # Utilisez le vectorizer pour transformer le commentaire
    comment_vector = vectorizer.transform([treated_comment])
    
    # Appelez le modèle pour prédire la classification
    result = model.predict(comment_vector)

    return f"Ce commentaire est {'malpoli :( ' if result == 1 else 'poli'}."

if __name__ == '__main__':
    app.run()
