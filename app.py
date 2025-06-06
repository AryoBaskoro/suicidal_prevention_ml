import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from scipy.sparse import hstack
import xgboost as xgb
import os

nltk_data_dir = './nltk_data'

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('tfidf_vectorizer_ml.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf1_vectorizer_ml.pkl', 'rb') as f:
    tfidf1 = pickle.load(f)

with open('label_encoder_ml.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    logreg = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_linear = pickle.load(f)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences)

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    allowed_chars = set(string.ascii_letters + "áéíóúãõàâêôç ")
    text = ''.join(c for c in text if c in allowed_chars)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

st.title("Sentiment Analysis with Machine Learning Models")
st.markdown("Input a Tweet, and predict the sentiment using the selected model.")

input_text = st.text_area("Enter Tweet for Prediction:")

model_choice = st.selectbox(
    "Select Model for Prediction:",
    ["XGBoost", "Naive Bayes", "Logistic Regression", "Linear SVM"]
)

if st.button("Check Tweet Sentiment"):
    if input_text:
        cleaned_text = clean_text(input_text)
        preprocessed_text = preprocess(cleaned_text)
        
        tfidf_vectorized = tfidf.transform([preprocessed_text])
        tfidf_vectorized1 = tfidf1.transform([preprocessed_text])

        num_features = [[len(input_text), count_sentences(input_text)]]
        num_features = hstack([tfidf_vectorized, num_features]) 

        if model_choice == "XGBoost":
            pred = xgb_model.predict(num_features)
            model_name = "XGBoost"
        elif model_choice == "Naive Bayes":
            pred = nb_model.predict(num_features)
            model_name = "Naive Bayes"
        elif model_choice == "Logistic Regression":
            pred = logreg.predict(num_features)
            model_name = "Logistic Regression"
        else:  
            pred = svm_linear.predict(tfidf_vectorized1.toarray())
            model_name = "Linear SVM"

        st.subheader(f"{model_name} Prediction: {label_encoder.inverse_transform(pred)}")

        st.subheader("Preprocessed Text:")
        st.write(preprocessed_text)
        
        tfidf_values = tfidf_vectorized.toarray()[0]
        tfidf_features = tfidf.get_feature_names_out()

        word_tfidf = {word: tfidf_values[i] for i, word in enumerate(tfidf_features) if tfidf_values[i] > 0}
        sorted_word_tfidf = dict(sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True))
        
        top_words = list(sorted_word_tfidf.keys())[:10]
        top_values = list(sorted_word_tfidf.values())[:10]

        fig, ax = plt.subplots()
        ax.barh(top_words, top_values)
        ax.set_xlabel('TF-IDF Value')
        ax.set_title('Top 10 TF-IDF Words')
        st.pyplot(fig)

    else:
        st.error("Please enter some text for prediction.")
