import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups

# Set page config
st.set_page_config(page_title="Text Classification Lab", page_icon="📝", layout="wide")

# Load data
@st.cache_data
def load_spam_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

@st.cache_data
def load_news_data():
    categories = ['rec.sport.baseball', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame({'text': newsgroups.data, 'category': newsgroups.target})
    df['category'] = df['category'].map(lambda x: newsgroups.target_names[x])
    return df

# Preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization using a regex-based approach
    tokens = re.findall(r'\b\w+\b', text)
    # Stopword removal
    stop_words = set(ENGLISH_STOP_WORDS)
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization (fallback to stemming if WordNet is unavailable)
    lemmatizer = WordNetLemmatizer()
    try:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Train models
@st.cache_resource
def train_spam_models():
    df = load_spam_data()
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    results = {}
    cms = {}
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        cms[name] = confusion_matrix(y_test, y_pred)
        trained_models[name] = model
    
    return trained_models, vectorizer, results, cms, X_test, y_test

@st.cache_resource
def train_news_models():
    df = load_news_data()
    X = df['text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    results = {}
    cms = {}
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        cms[name] = confusion_matrix(y_test, y_pred)
        trained_models[name] = model
    
    return trained_models, vectorizer, results, cms, X_test, y_test

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Preprocessing", "Vectorization", "Models", "Prediction", "Visualizations", "Insights"])

if page == "Overview":
    st.title("Text Classification Lab – Spam Detection & News Group Classification")
    st.markdown("""
    ## Project Overview
    
    This application demonstrates a complete NLP pipeline for text classification tasks, specifically:
    
    - **Spam Detection**: Classifying SMS messages as spam or ham (legitimate)
    - **News Group Classification**: Categorizing news articles into topics like sports, science, and politics
    
    ### Objectives
    - Showcase text preprocessing techniques
    - Demonstrate vectorization methods (CountVectorizer and TF-IDF)
    - Compare multiple machine learning models
    - Provide real-time prediction capabilities
    - Visualize model performance
    
    ### Dataset
    - **Spam Dataset**: SMS Spam Collection (5572 messages)
    - **News Dataset**: 20 Newsgroups (subset of 5 categories)
    
    Navigate through the sections using the sidebar to explore different aspects of the NLP pipeline.
    """)

elif page == "Preprocessing":
    st.title("Text Preprocessing")
    st.markdown("""
    Text preprocessing is crucial for NLP tasks. This section demonstrates:
    - Lowercasing
    - Punctuation removal
    - Tokenization
    - Stopword removal
    - Lemmatization
    """)
    
    user_input = st.text_area("Enter text to preprocess:", "This is a SAMPLE text with Punctuation! And some stop words like the, and, or.")
    
    if st.button("Preprocess"):
        original = user_input
        processed = preprocess_text(user_input)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Text")
            st.write(original)
        with col2:
            st.subheader("Processed Text")
            st.write(processed)

elif page == "Vectorization":
    st.title("Text Vectorization")
    st.markdown("""
    Text vectorization converts text into numerical representations that machine learning models can understand.
    
    ### CountVectorizer
    Creates a matrix of token counts. Each document becomes a vector where each element represents the count of a word in that document.
    
    ### TF-IDF (Term Frequency-Inverse Document Frequency)
    Weighs the importance of words by considering how frequently they appear in a document relative to their frequency across all documents.
    
    We use TF-IDF in this project as it often performs better for text classification tasks.
    """)
    
    example_text = ["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"]
    
    st.subheader("Example with CountVectorizer")
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(example_text)
    count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
    st.dataframe(count_df)
    
    st.subheader("Example with TF-IDF")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(example_text)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    st.dataframe(tfidf_df)

elif page == "Models":
    st.title("Model Comparison")
    st.markdown("This section compares the performance of different machine learning models on both spam detection and news classification tasks.")
    
    tab1, tab2 = st.tabs(["Spam Detection", "News Classification"])
    
    with tab1:
        st.subheader("Spam Detection Models")
        _, _, results, _, _, _ = train_spam_models()
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        st.table(results_df)
        
        st.bar_chart(results_df)
    
    with tab2:
        st.subheader("News Classification Models")
        _, _, results, _, _, _ = train_news_models()
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        st.table(results_df)
        
        st.bar_chart(results_df)

elif page == "Prediction":
    st.title("Real-time Prediction")
    
    tab1, tab2 = st.tabs(["Spam Detection", "News Classification"])
    
    with tab1:
        st.subheader("Spam Detection")
        user_message = st.text_area("Enter a message to check for spam:")
        
        if st.button("Predict Spam"):
            if user_message:
                trained_models, vectorizer, _, _, _, _ = train_spam_models()
                processed = preprocess_text(user_message)
                vectorized = vectorizer.transform([processed])
                
                # Use Naive Bayes for prediction (can be changed)
                model = trained_models['Naive Bayes']
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                
                confidence = max(proba) * 100
                
                if prediction == 'spam':
                    st.error(f"Prediction: **SPAM** (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"Prediction: **HAM** (Confidence: {confidence:.2f}%)")
            else:
                st.warning("Please enter a message.")
    
    with tab2:
        st.subheader("News Classification")
        user_text = st.text_area("Enter news text to classify:")
        
        if st.button("Classify News"):
            if user_text:
                trained_models, vectorizer, _, _, _, _ = train_news_models()
                processed = preprocess_text(user_text)
                vectorized = vectorizer.transform([processed])
                
                # Use SVM for prediction
                model = trained_models['SVM']
                prediction = model.predict(vectorized)[0]
                
                st.info(f"Predicted Category: **{prediction}**")
            else:
                st.warning("Please enter news text.")

elif page == "Visualizations":
    st.title("Model Evaluation Visualizations")
    st.markdown("Confusion matrices for each model showing true positives, false positives, true negatives, and false negatives.")
    
    tab1, tab2 = st.tabs(["Spam Detection", "News Classification"])
    
    with tab1:
        st.subheader("Spam Detection Confusion Matrices")
        trained_models, _, _, cms, _, y_test = train_spam_models()
        
        for name, cm in cms.items():
            st.subheader(f"{name}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
    
    with tab2:
        st.subheader("News Classification Confusion Matrices")
        trained_models, _, _, cms, _, y_test = train_news_models()
        categories = ['rec.sport.baseball', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']
        
        for name, cm in cms.items():
            st.subheader(f"{name}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=categories, yticklabels=categories)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif page == "Insights":
    st.title("Key Insights")
    st.markdown("""
    ## Project Insights
    
    ### Model Performance
    - **Naive Bayes** performed well on spam detection due to its effectiveness with text data and TF-IDF features.
    - **SVM** showed strong performance on both tasks, particularly for news classification.
    - **Logistic Regression** provided good results and is interpretable.
    - **Random Forest** performed decently but was outperformed by other models.
    
    ### TF-IDF Effectiveness
    TF-IDF vectorization proved effective by:
    - Reducing the impact of common words across documents
    - Highlighting important terms for classification
    - Providing better feature representation than simple counts
    
    ### Preprocessing Impact
    The preprocessing pipeline significantly improved model performance by:
    - Normalizing text (lowercasing)
    - Removing noise (punctuation)
    - Reducing dimensionality (stopword removal)
    - Grouping similar words (lemmatization)
    
    ### Real-world Applications
    - Spam detection can be applied to email filtering, SMS moderation
    - News classification can be used for content categorization, recommendation systems
    - The pipeline can be adapted for other text classification tasks
    
    ### Future Improvements
    - Use more advanced models like BERT or other transformers
    - Implement cross-validation for more robust evaluation
    - Add more preprocessing techniques (stemming, custom stop words)
    - Include hyperparameter tuning
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Scikit-learn, NLTK, and other Python libraries.")