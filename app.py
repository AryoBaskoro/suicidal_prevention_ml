import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from scipy.sparse import hstack
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import xgboost as xgb # Pastikan xgboost terinstall

st.set_page_config(
    page_title="Sentimind - Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50; 
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .status-normal { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .status-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .status-danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

LSTM_MAX_LEN = 200 
LSTM_LABEL_MAP = {
    0: 'Normal',
    1: 'Depression',
    2: 'Suicidal',
    3: 'Anxiety',
    4: 'Bipolar',
    5: 'Stress',
    6: 'Personality disorder'
}

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
            if not hasattr(xgb_model, 'feature_types'): xgb_model.feature_types = None
            models['xgb_model'] = xgb_model
        
        with open('tfidf_vectorizer_ml.pkl', 'rb') as f: models['tfidf'] = pickle.load(f)
        with open('tfidf1_vectorizer_ml.pkl', 'rb') as f: models['tfidf1'] = pickle.load(f)
        with open('label_encoder_ml.pkl', 'rb') as f: models['label_encoder'] = pickle.load(f)
        with open('naive_bayes_model.pkl', 'rb') as f: models['nb_model'] = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f: models['logreg'] = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f: models['svm_linear'] = pickle.load(f)
        
        lstm_path = os.path.join('LSTM', 'lstm_suicidal_nlp_no_optimizer.h5')
        tokenizer_path = os.path.join('LSTM', 'lstm_tokenizer.pickle')
        
        if os.path.exists(lstm_path) and os.path.exists(tokenizer_path):
            models['lstm_model'] = load_model(lstm_path, compile=False) 
            with open(tokenizer_path, 'rb') as f:
                models['lstm_tokenizer'] = pickle.load(f)
        else:
            st.warning("⚠️ LSTM model files not found in ./LSTM/ folder.")

        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s.strip() for s in sentences if s.strip()])

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text) 
    text = re.sub(r"[^a-z\s.,!?']", '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_ml(text):
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    except:
        return text

models = load_models()

if models:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=80)
        st.title("Settings")
        
        model_type = st.radio(
            "Select Model Type:",
            ("Deep Learning (LSTM)", "Classic Machine Learning")
        )
        
        selected_model = None
        if model_type == "Classic Machine Learning":
            selected_model = st.selectbox(
                "Choose Algorithm:",
                ["XGBoost", "Naive Bayes", "Logistic Regression", "Linear SVM"]
            )
        else:
            st.info("Using LSTM (Long Short-Term Memory) neural network for advanced context understanding.")
            selected_model = "LSTM"

        st.markdown("---")
        st.markdown("### About")
        st.caption("This application uses Natural Language Processing to detect mental health status from text.")

    st.title("🧠 Sentimind: Mental Health Analyzer")
    st.markdown("#### How are you feeling today? Let's analyze your thoughts.")

    examples = {
        "Normal": [
            "Everything feels fine, just a normal day with coffee and work.",
            "I am planning to go to the movies this weekend with my friends.",
            "The weather is really nice today, maybe I'll go for a jog."
        ],
        "Negative": [
            "I can't sleep at night and my thoughts won't stop racing.",
            "I feel so overwhelmed with my workload, it's suffocating.",
            "Nothing seems to make me happy anymore, I'm just tired of everything."
        ],
        "Critical": [
            "Sometimes I feel like ending it all. What's the point?",
            "I feel like a burden to everyone, the world would be better without me.",
            "I have been thinking about ways to hurt myself, I can't take this pain."
        ]
    }

    if 'user_input' not in st.session_state: 
        st.session_state.user_input = ""

    def set_text(txt): 
        st.session_state.user_input = txt

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 🙂 Normal")
        for i, text in enumerate(examples["Normal"]):
            if st.button(f"Option {i+1}", key=f"norm_{i}", help=text):
                set_text(text)

    with col2:
        st.markdown("### 😟 Negative")
        for i, text in enumerate(examples["Negative"]):
            if st.button(f"Option {i+1}", key=f"neg_{i}", help=text):
                set_text(text)

    with col3:
        st.markdown("### 🆘 Critical")
        for i, text in enumerate(examples["Critical"]):
            if st.button(f"Option {i+1}", key=f"crit_{i}", help=text):
                set_text(text)

    st.divider()
    st.text_area("Input Text:", value=st.session_state.user_input, height=100)

    input_text = st.text_area(
        "Enter your text here (English):", 
        value=st.session_state.user_input, 
        height=150,
        placeholder="Type something here..."
    )

    if st.button("🔍 Analyze Sentiment"):
        if not input_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner('Analyzing patterns...'):
                cleaned_text = clean_text(input_text)
                
                if selected_model == "LSTM":
                    if 'lstm_model' in models:
                        # 1. Tokenize & Pad
                        seq = models['lstm_tokenizer'].texts_to_sequences([cleaned_text])
                        padded = pad_sequences(seq, maxlen=LSTM_MAX_LEN)
                        
                        pred_prob = models['lstm_model'].predict(padded)
                        pred_idx = np.argmax(pred_prob)
                        final_label = LSTM_LABEL_MAP[pred_idx]
                        confidence = np.max(pred_prob) * 100
                        
                        st.markdown("---")
                        c1, c2 = st.columns([2, 3])
                        
                        with c1:
                            color_class = "status-normal"
                            if final_label in ['Anxiety', 'Depression', 'Bipolar', 'Personality disorder']:
                                color_class = "status-warning"
                            elif final_label in ['Suicidal', 'Stress']: 
                                color_class = "status-danger"
                                
                            st.markdown(f"""
                            <div class="prediction-card {color_class}">
                                <h3>Status Detected</h3>
                                <h1 style="margin:0;">{final_label}</h1>
                                <p style="margin-top:10px;">Confidence: <b>{confidence:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with c2:
                            st.subheader("Probability Distribution")
                            probs_df = pd.DataFrame({
                                'Status': list(LSTM_LABEL_MAP.values()),
                                'Probability': pred_prob[0]
                            })
                            st.bar_chart(probs_df.set_index('Status'), color="#4CAF50")

                    else:
                        st.error("LSTM Model not loaded properly.")

                else:
                    preprocessed_text = preprocess_ml(cleaned_text)
                    
                    tfidf_vec = models['tfidf'].transform([preprocessed_text])
                    tfidf_vec1 = models['tfidf1'].transform([preprocessed_text])
                    num_feats = hstack([tfidf_vec, [[len(input_text), count_sentences(input_text)]]])
                    
                    pred_idx = None
                    if selected_model == "XGBoost":
                        pred_idx = models['xgb_model'].predict(num_feats.toarray())
                    elif selected_model == "Naive Bayes":
                        pred_idx = models['nb_model'].predict(num_feats)
                    elif selected_model == "Logistic Regression":
                        pred_idx = models['logreg'].predict(num_feats)
                    elif selected_model == "Linear SVM":
                        pred_idx = models['svm_linear'].predict(tfidf_vec1.toarray())
                    
                    final_label = models['label_encoder'].inverse_transform(pred_idx)[0]
                    
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="prediction-card status-normal">
                        <h3>Prediction ({selected_model})</h3>
                        <h1 style="margin:0;">{final_label}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 See Keyword Analysis (TF-IDF)"):
                        tfidf_values = tfidf_vec.toarray()[0]
                        feature_names = models['tfidf'].get_feature_names_out()
                        
                        word_scores = {feature_names[i]: tfidf_values[i] for i in range(len(feature_names)) if tfidf_values[i] > 0}
                        sorted_words = dict(sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:10])
                        
                        if sorted_words:
                            st.bar_chart(pd.DataFrame({'Score': sorted_words.values()}, index=sorted_words.keys()))
                        else:
                            st.info("No significant keywords found in the text vocabulary.")

else:
    st.error("❌ Critical Error: Models could not be loaded. Please check your file paths.")