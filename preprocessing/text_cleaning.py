import re
from textblob import TextBlob
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[\d\W]+', ' ', text)
    return text.strip()

def correct_spelling(text: str) -> str:
    corrected = TextBlob(text).correct()
    return str(corrected)

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.text.lower() == "left":
            lemmas.append("left")  # keep unchanged
        else:
            lemmas.append(token.lemma_)
    return " ".join(lemmas)

def preprocess_claim_description(df: pd.DataFrame, col: str = "ClaimDescription") -> pd.Series:
    text_series = df[col].astype(str).apply(clean_text).apply(correct_spelling).apply(lemmatize_text)
    
    # These are bandaid fixes for one common problem and another I caught by a chance inspection
    text_series = text_series.str.replace(r'\blower\b', 'low', regex=True)
    text_series = text_series.str.replace(r'\bcartoon\b', 'carton', regex=True)
    
    return text_series