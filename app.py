import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import pandas as pd

st.title("DISASTER OR NO DISASTER NLP CLASSIFICATION")
st.write("")
val = st.text_input("Your sentence")

df = pd.read_csv("train.csv")
X = df["text"]
y = df["target"]

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

button = st.button("Predict")

if button:
    st.balloons()
    val_c = [val]
    data = cv.transform(val_c)
    prediction = clf.predict(data)
    if prediction == 0:
        st.write("Not a disaster")
    if prediction == 1:
        st.write("Disaster")


