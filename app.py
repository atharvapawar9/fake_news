import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string

# Function to preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Load the trained models
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier(random_state=0)
RFC = RandomForestClassifier(random_state=0)

# Load the training data
df = pd.read_csv("Fake.csv")  # Replace "your_training_data.csv" with the path to your training data file
x_train = df["text"]
y_train = df["class"]

# Initialize and fit the TF-IDF vectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)

# Function for manual testing
def manual_testing(news):
    news = wordopt(news)
    new_x_test = [news]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return pred_LR[0], pred_DT[0], pred_GBC[0], pred_RFC[0]

# Streamlit app
def main():
    st.title("Fake News Detection App")
    st.write("Enter the news text below:")

    # Text input box
    news_input = st.text_area("Enter news here:")

    if st.button("Check"):
        # Perform manual testing
        pred_LR, pred_DT, pred_GBC, pred_RFC = manual_testing(news_input)

        st.subheader("Prediction Results:")
        st.write("Logistic Regression:", "Fake News" if pred_LR == 0 else "Not A Fake News")
        st.write("Decision Tree:", "Fake News" if pred_DT == 0 else "Not A Fake News")
        st.write("Gradient Boosting Classifier:", "Fake News" if pred_GBC == 0 else "Not A Fake News")
        st.write("Random Forest Classifier:", "Fake News" if pred_RFC == 0 else "Not A Fake News")

if __name__ == "__main__":
    main()
