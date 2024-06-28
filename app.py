# Standard libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('stopwords')

# Import additional models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Load SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Function to remove punctuation
def remove_punctuation(text):
    return ''.join([ch for ch in text if ch not in string.punctuation])

# Function to remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Function to tokenize and lemmatize using SpaCy
def tokenize_and_lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Read the CSV file with specified encoding and column names
df = pd.read_csv(r'C:\Users\aaron\OneDrive\Desktop\projects\undertest\sentiment\data\all-data.csv', encoding='unicode_escape', names=['Sentiment', 'Text'])

# Convert the 'Text' column to lowercase
df['Text'] = df['Text'].str.lower()

# Remove punctuation
df['Text'] = df['Text'].apply(remove_punctuation)

# Remove stopwords and tokenize/lemmatize using SpaCy
df['Text'] = df['Text'].apply(remove_stopwords)
df['Text'] = df['Text'].apply(tokenize_and_lemmatize)

# Prepare data for WordCloud (concatenate tokens into a single string)
text_for_wordcloud = ' '.join(df['Text'])

# Generate WordCloud
wordcloud = WordCloud(width=1500, height=400, background_color='white', stopwords=None).generate(text_for_wordcloud)

# Define X and y for classifiers
X = df['Text']
y = df['Sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train classifiers
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_tfidf, y_train_encoded)

lgb_classifier = lgb.LGBMClassifier()
lgb_classifier.fit(X_train_tfidf, y_train_encoded)

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_tfidf, y_train_encoded)

# Function to preprocess new input data
def preprocess_input(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return tokenize_and_lemmatize(text)

# Define Streamlit app
def main():
    st.title('Classifying Public Opinion with NLP')
    
    # Input text box for user input
    user_input = st.text_area("Enter text for sentiment analysis:", "")
    
    if st.button('Predict'):
        # Preprocess user input
        processed_input = preprocess_input(user_input)
        processed_input_tfidf = tfidf_vectorizer.transform([processed_input])
        
        # Make predictions
        rf_pred = rf_classifier.predict(processed_input_tfidf)
        lgb_pred = lgb_classifier.predict(processed_input_tfidf)
        xgb_pred = xgb_classifier.predict(processed_input_tfidf)
        
        # Implement voting mechanism
        votes = [rf_pred[0], lgb_pred[0], xgb_pred[0]]
        final_prediction = max(set(votes), key=votes.count)
        final_prediction = label_encoder.inverse_transform([final_prediction])[0]  # Convert label back to original sentiment
        
        # Display prediction
        st.write(f"Predicted Sentiment: {final_prediction}")

    # Display WordCloud
    st.subheader("Word Cloud")
    st.image(wordcloud.to_array(), use_column_width=True)

if __name__ == '__main__':
    main()
