import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def main():
    st.title("Fake News Detection App")

    # Load the CSV file
    df = pd.read_csv("Fake.csv")

    # Combine relevant columns into a single "text" column
    df['text'] = df['title'] + ' ' + df['text']

    # Drop unnecessary columns
    df = df.drop(["title", "subject", "date"], axis=1)

    # Apply preprocessing to the text column
    df["text"] = df["text"].apply(preprocess_text)

    # Define features and target variable
    X = df["text"]
    y = df["class"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize and fit the TF-IDF vectorizer
    vectorization = TfidfVectorizer()
    X_train_tfidf = vectorization.fit_transform(X_train)
    X_test_tfidf = vectorization.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
