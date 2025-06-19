import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from scipy.sparse import hstack
import xgboost as xgb
import os

# Configure NLTK data path first
nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the path before downloading
nltk.data.path.append(nltk_data_dir)

# Download NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)  
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)  # Add this for newer NLTK versions
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# Load models with error handling
@st.cache_resource
def load_models():
    models = {}
    try:
        # Load XGBoost model with compatibility fix
        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
            # Fix for XGBoost version compatibility
            if not hasattr(xgb_model, 'feature_types'):
                xgb_model.feature_types = None
            models['xgb_model'] = xgb_model
        
        with open('tfidf_vectorizer_ml.pkl', 'rb') as f:
            models['tfidf'] = pickle.load(f)
        
        with open('tfidf1_vectorizer_ml.pkl', 'rb') as f:
            models['tfidf1'] = pickle.load(f)
        
        with open('label_encoder_ml.pkl', 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        
        with open('naive_bayes_model.pkl', 'rb') as f:
            models['nb_model'] = pickle.load(f)
        
        with open('logistic_regression_model.pkl', 'rb') as f:
            models['logreg'] = pickle.load(f)
        
        with open('svm_model.pkl', 'rb') as f:
            models['svm_linear'] = pickle.load(f)
            
        return models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

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
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return text  # Return original text if preprocessing fails

# Streamlit UI
st.title("Sentiment Analysis with Machine Learning Models")
st.markdown("Input a Tweet, and predict the sentiment using the selected model.")

# Load models
models = load_models()

if models is not None:
    example_texts = {
        "Negative": [
            "I can't sleep at night and my thoughts won't stop racing.",
            "Lately, I've been feeling so numb and disconnected from everything.",
            "I'm afraid to go outside. The world feels overwhelming.",
            "Sometimes I feel like ending it all. What's the point?",
            "It ends tonight.I can't do it anymore. I quit.",
            "Is it worth it?Is all the trouble, work and anxiety really worth living for.",
            "do you ever feel empty and you had to pick a number from 15 depending how you feel about it i could write my funeral speech now but what is the point in afew years i would be forgotten anyway",
            "Lately I just feel like garbage. I havent left the house much in like 2 weeks, and I've been missing class. It all"
        ],
        "Neutral": [
            "Everything feels fine, just a normal day with coffee and work.",
            "Feeling a bit stressed, but I think I can manage it with deep breaths.",
            "How painful is it and what are the chances of dying from it"
        ],
        "Mixed": [
            "I'm trying to stay strong, but my mood swings are exhausting.",
            "My mind jumps from one idea to another. I can't stay still.",
            "I don't even recognize myself anymore. Who am I becoming?",
            "I heard from my therapist that anger and depression often go hand in hand, but I feel like it's kinda going off the rails for me. does anyone have more experience with this?",
            "i have been struggling with health anxiety and general spiraling lately, along with a host of other mental health issues. my psychiatrist recommended i try getting back on an antidepressant as they can sometimes help with anxiety (and my healthy dose of depression needs to be managed). we are thinking of prozac since i responded well to it when i was younger..",
            "I found out that in times where my social anxiety was really bad this would also heavily influence my own self confidence and self worth. I would beat myself up for everything I did which made it even harder to overcome my social anxiety. What do you think about this?",
            "I feel like my intrusive thoughts are winning right now, and I feel bad for every thought. I feel like a terrible person and I need to remove myself from everyone because I'm a terrible person and no one would want me around. My inner voice is very mean and I just don't know anymore. I'm just so tired of it all.",
            "The girl that I fell for 10 years ago also suffers from anxiety. I haven't got the guts to let her know how I feel about her. My anxiety recently developed into agoraphobia. I've been improving but I still can't travel far. That kind of demoralized me. My question is: Can a relationship still work if both suffer from anxiety?",
            "Lately I just feel like garbage. I havent left the house much in like 2 weeks, and I've been missing class. It all feels too overwhelming for me, but being at home makes me feel like trash too...I cant win. I cant sleep right either. I wake up every other hour and im so tired....",
            "i have been struggling with health anxiety and general spiraling lately, along with a host of other mental health issues",
            "I found out that in times where my social anxiety was really bad this would also heavily influence my own self",
            "I feel like my intrusive thoughts are winning right now, and I feel bad for every thought. I feel like a terrible",
            "The girl that I fell for 10 years ago also suffers from anxiety. I haven't got the guts to let her know how I feel about her. My anxiety recently developed i: Anxiety"
        ]
    }
    
    # Create columns for example buttons (removed positive)
    st.markdown("### Quick Examples:")
    col1, col2, col3 = st.columns(3)
    
    # Initialize session state for input text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Import random for selecting examples
    import random
    
    with col1:
        if st.button("😞 Negative"):
            st.session_state.input_text = random.choice(example_texts["Negative"])
            st.rerun()
    
    with col2:
        if st.button("😐 Neutral"):
            st.session_state.input_text = random.choice(example_texts["Neutral"])
            st.rerun()
    
    with col3:
        if st.button("🤔 Mixed"):
            st.session_state.input_text = random.choice(example_texts["Mixed"])
            st.rerun()
    
    # Text input area with session state
    input_text = st.text_area(
        "Enter Tweet for Prediction:",
        value=st.session_state.input_text,
        height=100,
        help="Click one of the example buttons above to fill with sample text, or type your own."
    )
    
    # Clear button
    if st.button("🗑️ Clear Text"):
        st.session_state.input_text = ""
        st.rerun()

    model_choice = st.selectbox(
        "Select Model for Prediction:",
        ["XGBoost", "Naive Bayes", "Logistic Regression", "Linear SVM"]
    )

    if st.button("Check Tweet Sentiment"):
        if input_text:
            try:
                cleaned_text = clean_text(input_text)
                preprocessed_text = preprocess(cleaned_text)
                
                tfidf_vectorized = models['tfidf'].transform([preprocessed_text])
                tfidf_vectorized1 = models['tfidf1'].transform([preprocessed_text])

                num_features = [[len(input_text), count_sentences(input_text)]]
                num_features = hstack([tfidf_vectorized, num_features]) 

                if model_choice == "XGBoost":
                    # Convert sparse matrix to dense for XGBoost compatibility
                    pred = models['xgb_model'].predict(num_features.toarray())
                    model_name = "XGBoost"
                elif model_choice == "Naive Bayes":
                    pred = models['nb_model'].predict(num_features)
                    model_name = "Naive Bayes"
                elif model_choice == "Logistic Regression":
                    pred = models['logreg'].predict(num_features)
                    model_name = "Logistic Regression"
                else:  
                    pred = models['svm_linear'].predict(tfidf_vectorized1.toarray())
                    model_name = "Linear SVM"

                st.subheader(f"{model_name} Prediction: {models['label_encoder'].inverse_transform(pred)}")

                st.subheader("Preprocessed Text:")
                st.write(preprocessed_text)
                
                # TF-IDF Analysis
                tfidf_values = tfidf_vectorized.toarray()[0]
                tfidf_features = models['tfidf'].get_feature_names_out()

                word_tfidf = {word: tfidf_values[i] for i, word in enumerate(tfidf_features) if tfidf_values[i] > 0}
                sorted_word_tfidf = dict(sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True))
                
                top_words = list(sorted_word_tfidf.keys())[:10]
                top_values = list(sorted_word_tfidf.values())[:10]

                if top_words:  # Only create visualization if there are words to display
                    st.subheader("Top 10 TF-IDF Words")
                    
                    # Create DataFrame for Streamlit chart
                    chart_data = pd.DataFrame({
                        'Words': top_words,
                        'TF-IDF Score': top_values
                    })
                    
                    # Display as horizontal bar chart using Streamlit
                    st.bar_chart(
                        chart_data.set_index('Words'),
                        height=400,
                        use_container_width=True
                    )
                    
                    # Alternative: Display as a table with color coding
                    st.subheader("TF-IDF Scores Table")
                    
                    # Create a styled dataframe
                    styled_df = chart_data.copy()
                    styled_df['TF-IDF Score'] = styled_df['TF-IDF Score'].round(4)
                    styled_df = styled_df.reset_index(drop=True)
                    styled_df.index = styled_df.index + 1  # Start index from 1
                    
                    # Display with metrics for top 3 words
                    col1, col2, col3 = st.columns(3)
                    if len(top_words) >= 3:
                        with col1:
                            st.metric(
                                label=f"🥇 Top Word: {top_words[0]}", 
                                value=f"{top_values[0]:.4f}"
                            )
                        with col2:
                            st.metric(
                                label=f"🥈 Second: {top_words[1]}", 
                                value=f"{top_values[1]:.4f}"
                            )
                        with col3:
                            st.metric(
                                label=f"🥉 Third: {top_words[2]}", 
                                value=f"{top_values[2]:.4f}"
                            )
                    
                    # Display full table
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=False
                    )
                    
                else:
                    st.warning("No significant words found for TF-IDF visualization.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter some text for prediction.")
else:
    st.error("Failed to load models. Please check if all model files are present in the current directory.")