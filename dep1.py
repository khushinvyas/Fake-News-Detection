import joblib
import streamlit as st

# Load the models and vectorizers
logistic_regression_model_title = joblib.load('LogisticRegression_Title.pkl')
logistic_regression_model_text = joblib.load('LogisticRegression_Text.pkl')
logistic_regression_model_title_text = joblib.load('LogisticRegression_Title+Text.pkl')
naive_bayes_model_title = joblib.load('NaiveBayes_Title.pkl')
naive_bayes_model_text = joblib.load('NaiveBayes_Text.pkl')
naive_bayes_model_title_text = joblib.load('NaiveBayes_Title+Text.pkl')
knn_model_title = joblib.load('KNN_Title.pkl')
knn_model_text = joblib.load('KNN_Text.pkl')
knn_model_title_text = joblib.load('KNN_Title+Text.pkl')
tfidf_vectorizer_title = joblib.load('Title_vectorizer.pkl')
tfidf_vectorizer_text = joblib.load('Text_vectorizer.pkl')
tfidf_vectorizer_title_text = joblib.load('Title+Text_vectorizer.pkl')

# Create a Streamlit app
st.title("Fake News Detection")

# User input
user_input = st.text_area("Enter the news data:")
input_type = st.selectbox("Select input type:", ("Title", "Text", "Title + Text"))

if st.button("Submit"):
    if input_type == "Title":
        user_input_vectorized = tfidf_vectorizer_title.transform([user_input])
        logistic_regression_prediction = logistic_regression_model_title.predict(user_input_vectorized)
        naive_bayes_prediction = naive_bayes_model_title.predict(user_input_vectorized)
        knn_prediction = knn_model_title.predict(user_input_vectorized)
    elif input_type == "Text":
        user_input_vectorized = tfidf_vectorizer_text.transform([user_input])
        logistic_regression_prediction = logistic_regression_model_text.predict(user_input_vectorized)
        naive_bayes_prediction = naive_bayes_model_text.predict(user_input_vectorized)
        knn_prediction = knn_model_text.predict(user_input_vectorized)
    elif input_type == "Title + Text":
        user_input_vectorized = tfidf_vectorizer_title_text.transform([user_input])
        logistic_regression_prediction = logistic_regression_model_title_text.predict(user_input_vectorized)
        naive_bayes_prediction = naive_bayes_model_title_text.predict(user_input_vectorized)
        knn_prediction = knn_model_title_text.predict(user_input_vectorized)

    predictions = [logistic_regression_prediction, naive_bayes_prediction, knn_prediction]

    if predictions.count(1) > predictions.count(0):
        st.error("Prediction: The news is FAKE.")
    else:
        st.success("Prediction: The news is REAL.")
