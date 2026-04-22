import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data.csv")
X = df[['hours studied']]
y = df[['exam score']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
st.title("EXAM SCORE PREDICTOR")
st.write(" ENTER HOURSE STUDIED TO PREDICT THE EXAM SCORE.")
hours = st.number_input("hours studied:", min_value=0.0, step=0.1)
if st.button("PREDICT SCORE"):
    predicted_score = model.predict([[hours]])[0]
    st.success(f"predicted score: {predicted_score:.2f}")
    st.write("###sample training data")
    st.detaframe(df)
