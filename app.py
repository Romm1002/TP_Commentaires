from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import joblib
import pandas as pd

# Chargez votre modèle Spacy et initialisez le vectorizer
nlp = spacy.load('en_core_web_sm')
vectorizer = TfidfVectorizer()

# Chargez votre modèle d'IA à l'aide de joblib
model = joblib.load('model.joblib')

# Chargez vos données d'entraînement et préparez df_clean
df = pd.read_csv('train.csv')
df_clean = df
df_clean['isToxic'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis=1).astype(int)
df_clean = df_clean[['comment_text', 'isToxic']].copy()
df_clean.rename(columns={'comment_text': 'Text'}, inplace=True)

app = Flask(__name__
            )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    comment = request.form['comment']

    # Traitez le commentaire de la même manière que vous l'avez fait pour votre modèle
    spacy_comment = nlp(comment, disable=["parser", "tagger", "ner", "textcat"])
    treated_tokens = [w.text for w in spacy_comment if w.is_alpha and not w.is_stop]
    treated_comment = " ".join(treated_tokens)
    
    # Ajustez le vectorizer avec le vocabulaire des données d'entraînement
    vectorizer.fit(df_clean['Text'])

    # Utilisez le vectorizer pour transformer le commentaire
    comment_vector = vectorizer.transform([treated_comment])
    
    # Appelez le modèle pour prédire la classification
    result = model.predict(comment_vector)

    return f"Le commentaire est {'toxique' if result == 1 else 'non toxique'}."

if __name__ == '__main__':
    app.run()
