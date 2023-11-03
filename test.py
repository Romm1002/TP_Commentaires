import pandas as pd
import spacy
import joblib

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('train.csv')
df_clean = df
df_clean['isToxic'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis=1).astype(int)
df_clean = df_clean[['comment_text', 'isToxic']].copy()
df_clean.rename(columns={'comment_text': 'Text'}, inplace=True)
df_toxic = df_clean[df_clean['isToxic'] == 1]
df_non_toxic = df_clean[df_clean['isToxic'] == 0].sample(n=len(df_toxic))
df_equilibre = pd.concat([df_toxic, df_non_toxic])

def treat_comment(comment):
    spacy_comment = nlp(comment, disable=["parser", "tagger", "ner", "textcat"])
    treated_tokens = [w.text for w in spacy_comment if w.is_alpha and not w.is_stop]
    return " ".join(treated_tokens)
df_equilibre['Text'] = df_equilibre['Text'].map(treat_comment)
df_equilibre.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_clean['Text'])
Y = df_clean['isToxic']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy_score(Y_test, y_pred)

joblib.dump(model, 'model.joblib')