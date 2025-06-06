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
    # Example texts for testing - 4 examples for each mood
    example_texts = {
        "Positive": [
            "What an absolutely incredible day I'm having! Started my morning with a perfect cup of coffee while watching the sunrise through my window. The weather is absolutely gorgeous - clear blue skies and just the right temperature. Then I got the most amazing call from my boss telling me about my promotion! I've been working so hard for this moment and it finally paid off. I'm practically bouncing off the walls with excitement! My family is so proud of me, and my friends have been sending congratulations all day. I feel like I'm on top of the world right now! Life is truly wonderful and I'm so grateful for all these blessings. Can't wait to see what other amazing things are coming my way! 😊✨🎉",
            "Just finished the most incredible workout session of my life and I am absolutely buzzing with energy! Hit all my personal records today - bench press, squats, deadlifts - everything was just perfect. My trainer said she's never seen such determination and progress in someone. The endorphins are flowing through my body like pure electricity! I feel like I could conquer the world right now. My fitness journey has been challenging but moments like these make every drop of sweat worth it. Looking in the mirror, I can see how much stronger and healthier I've become. My confidence is through the roof and I'm feeling unstoppable! Tonight I'm treating myself to a healthy celebration dinner because I've earned it. This is what living your best life feels like! 💪🔥🌟",
            "Today marks my 25th birthday and I'm having the most magical celebration surrounded by everyone I love! My family surprised me with a party that brought tears of happiness to my eyes. All my closest friends traveled from different cities just to be here with me. The decorations are absolutely beautiful, the cake is delicious, and the love in this room is overwhelming in the best possible way. I've received so many heartfelt messages, thoughtful gifts, and warm hugs that my heart feels like it might burst from joy. Looking back on this past year, I've grown so much as a person and achieved things I never thought possible. I'm entering this new year of life with so much excitement and gratitude. Best birthday ever and I'll remember this day forever! 🎂🎈❤️🥳",
            "I can barely contain my excitement - I just got accepted to my dream university with a full scholarship! After years of studying late into the night, countless practice tests, and pushing through moments of doubt, all my hard work has finally paid off in the most spectacular way. My parents are crying tears of joy and my little sister is telling everyone she meets about her 'genius' sibling. This acceptance letter represents not just my academic achievement, but the beginning of an incredible new chapter in my life. I'm going to study exactly what I'm passionate about, meet amazing people, and chase my dreams with everything I've got. The future has never looked brighter and I'm ready to embrace every opportunity that comes my way! 🎓📚🌟🚀"
        ],
        "Negative": [
            "This has to be the absolute worst day of my entire life and I honestly don't know how things can get any worse. Started my morning by spilling coffee all over my only clean work shirt, then got stuck in traffic for two hours making me late for the most important meeting of my career. When I finally arrived, I discovered that my presentation files were corrupted and I looked like a complete fool in front of the entire board. To make matters worse, my boss pulled me aside after the meeting and told me they're letting me go due to budget cuts. Twenty-four hours ago I had a job, a sense of purpose, and hope for the future. Now I'm sitting here wondering how I'm going to pay my bills, keep my apartment, or even face my family with this news. Everything feels like it's falling apart and I can't see any light at the end of this tunnel. 😢💔",
            "My world completely crumbled today when I got the call that I'd been dreading for weeks - I lost my job due to company downsizing, and as if the universe wanted to pile on more misery, my car decided to break down on the highway on my way home. I'm sitting here on the side of the road waiting for a tow truck I can't afford, trying to process how my stable life just disappeared in a single day. The repair costs are going to drain what little savings I have left, and now I don't even have reliable transportation to look for new work. I feel completely hopeless and overwhelmed, like I'm drowning in problems with no way to swim to the surface. My anxiety is through the roof and I can't stop thinking about how I'm going to survive the next few months. Everything feels impossible right now. 🚗💸😰",
            "I've been struggling with severe anxiety and depression for months now, and today feels like I've hit rock bottom. Every morning I wake up with this heavy weight on my chest that makes it hard to breathe, and getting out of bed feels like climbing Mount Everest. Simple tasks that used to be easy now feel overwhelming and impossible. I've been trying therapy and medication, but some days the darkness feels stronger than any treatment. My friends and family try to understand, but I can see the worry and helplessness in their eyes when they look at me. I used to be someone who could handle anything life threw at me, but now I feel like I'm losing a battle against my own mind every single day. I'm tired of pretending to be okay when I'm really falling apart inside. 🌧️😔💔",
            "My heart is completely shattered after the most devastating breakup of my life, and I feel like I'm drowning in an ocean of sadness with no shore in sight. We were together for three years and I thought we were building a future together - we talked about marriage, kids, growing old together. But yesterday they told me they've been feeling disconnected for months and need space to figure themselves out. All our shared dreams, inside jokes, and memories feel like they're mocking me now. I keep reaching for my phone to text them, forgetting for a moment that they're no longer mine to talk to. The apartment feels so empty without their laughter, and I don't know how to fill this gaping hole in my life. Every love song, every couple I see on the street, every place we used to go together feels like a knife twisting in my chest. 💔😭🌧️"
        ],
        "Neutral": [
            "Just finished my usual morning routine of coffee and reading through today's news headlines while sitting at my kitchen table. The weather outside looks decent enough for a regular Tuesday, nothing particularly exciting happening in the forecast. I've got my standard work project to continue today - reviewing quarterly reports and preparing for the upcoming client presentation next week. My commute should be about the same as always, probably around thirty minutes depending on traffic patterns. Planning to grab lunch at that sandwich place down the street from the office, maybe try their turkey club since I usually get the same ham and cheese. After work I'll probably stop by the grocery store to pick up ingredients for dinner, thinking about making pasta tonight since it's simple and I have most of the ingredients already. Overall, just another typical day in the routine. Nothing too exciting planned, but that's perfectly fine with me. 📅☕📰",
            "Attended our monthly department meeting this afternoon where we discussed the third quarter performance metrics and outlined our strategy for the next fiscal period. The presentation covered various aspects of our current projects, budget allocations, and timeline adjustments that need to be implemented. We reviewed the client feedback from recent deliverables and identified areas where we can improve our processes. The team leads provided updates on their respective areas of responsibility, and we scheduled follow-up meetings for more detailed discussions on specific initiatives. Overall, it was a productive session that covered all the necessary business items on our agenda. I took notes on the action items assigned to my department and will need to coordinate with other team members to ensure we meet our deadlines. Standard business operations continuing as expected. 📊💼📋",
            "Went to the grocery store this evening to pick up items for the week ahead, including fresh vegetables, some protein options, and basic household supplies. The store wasn't too crowded for a Wednesday night, so I was able to get through my shopping list fairly efficiently. I selected some broccoli, carrots, and bell peppers for meal prep, along with chicken breast and ground turkey for protein. Also grabbed some pasta, rice, and canned goods to keep the pantry stocked. The checkout process was smooth and the cashier was polite and professional. Now I'm back home putting everything away and planning out meals for the next few days. Thinking about making a stir-fry tomorrow night since I have all the ingredients now. Meal planning helps keep the week organized and saves time during busy workdays. 🛒🥬🍗",
            "Completed my daily commute to the office this morning, taking the usual route through downtown traffic. Left the house at 7:30 AM and arrived at the office parking garage by 8:15 AM, which is pretty standard timing for a regular workday. Traffic was moving at a reasonable pace with no major delays or accidents reported on the radio. I listened to a podcast about productivity tips during the drive, which was moderately interesting and helped pass the time. The weather was clear and visibility was good, making for safe driving conditions. Once I arrived at the office, I grabbed my laptop bag and headed up to the third floor where my desk is located. Started the workday by checking emails and reviewing my schedule for the day. Pretty routine morning overall, nothing out of the ordinary to report. 🚗🏢⏰"
        ],
        "Mixed": [
            "Just got back from watching the latest blockbuster movie that everyone's been talking about, and I have seriously mixed feelings about the whole experience. On one hand, the special effects were absolutely mind-blowing - the CGI was so realistic it felt like I was actually in another world, and the action sequences had me on the edge of my seat. The lead actors delivered powerful performances that really drew me into their characters' emotional journeys. The cinematography was stunning and the soundtrack perfectly complemented every scene. However, I have to say that the plot was incredibly confusing and seemed to jump around without clear connections between storylines. The ending felt rushed and left too many questions unanswered, which was really disappointing after investing three hours in the story. The pacing was also inconsistent - some parts dragged on too long while others felt rushed. Overall, it's a movie that looks amazing but doesn't quite deliver on storytelling. 🎬🤔✨",
            "Today I received news about my promotion at work, and honestly, I'm experiencing a whirlwind of conflicting emotions that I'm still trying to process. On the positive side, I'm absolutely thrilled about the recognition of my hard work over the past two years and the significant salary increase that comes with this new position. It validates all those late nights and weekend hours I've put in, and my family is so proud of this achievement. The new role will give me opportunities to lead projects I'm passionate about and work with talented people across different departments. However, I'm also feeling quite anxious about the increased responsibilities and pressure that comes with this position. The role requires longer hours, more travel, and managing a team of fifteen people, which feels overwhelming. I'm worried about maintaining work-life balance and whether I'll be able to handle the stress. It's exciting and terrifying at the same time. 📈😰🎉",
            "Just returned from what was supposed to be the perfect vacation getaway, and I'm dealing with a strange mix of relaxation and stress that's making my head spin. The destination was absolutely beautiful - pristine beaches, incredible sunsets, and the most peaceful environment I've experienced in years. I loved disconnecting from technology, reading books by the ocean, and just enjoying the simple pleasure of not having any schedule to follow. The local food was amazing, the people were friendly, and I created some wonderful memories that I'll treasure forever. My stress levels were at an all-time low and I felt completely recharged. But now that I'm back to reality, I'm facing a mountain of work emails, missed deadlines, and the overwhelming feeling that I'm already behind on everything. The transition back to normal life feels jarring and I'm struggling to readjust. The vacation was perfect, but coming back to real life is brutal. 🏖️📧😅",
            "The holidays are such a complex time for me emotionally, filled with moments of pure joy mixed with underlying stress that I can never quite shake off. I absolutely love spending quality time with my extended family - the laughter around the dinner table, watching my nieces and nephews get excited about presents, and the warm feeling of being surrounded by people who care about me. These moments remind me why family is so important and create memories that I'll cherish forever. The traditions we've built over the years bring such comfort and continuity to my life. However, the financial pressure of buying gifts for everyone, the exhaustion from traveling between multiple family gatherings, and the logistics of coordinating schedules with relatives across different states always leaves me feeling drained. The expectations to be cheerful and social when I'm already overwhelmed can be exhausting. It's wonderful and stressful simultaneously, and I always need a vacation after the holidays. 🎄💸😊"
        ]
    }
    
    # Create columns for example buttons
    st.markdown("### Quick Examples:")
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize session state for input text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Import random for selecting examples
    import random
    
    with col1:
        if st.button("😊 Positive"):
            st.session_state.input_text = random.choice(example_texts["Positive"])
            st.rerun()
    
    with col2:
        if st.button("😞 Negative"):
            st.session_state.input_text = random.choice(example_texts["Negative"])
            st.rerun()
    
    with col3:
        if st.button("😐 Neutral"):
            st.session_state.input_text = random.choice(example_texts["Neutral"])
            st.rerun()
    
    with col4:
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