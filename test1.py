# # %%
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# # %%
# # Load the dataset
# data = pd.read_csv("C:\\Users\\khush\\OneDrive\\Desktop\\Pds_Project\\train.csv\\test1\\train.csv")

# # %%
# # Data preprocessing
# data = data.drop("author", axis=1)
# data = data.drop("id", axis=1)
# data = data.dropna()
# data = data.reset_index()
# data['title+text'] = data['title'] + ' ' + data['text']
# data = data[:20203]

# # %%
# x = data.drop('label', axis=1)
# y = data['label']
# x1 = x['title'].values
# x2 = x['text'].values
# x3 = x['title+text'].values

# # %%
# # Vectorize the textual data
# vectorizer1 = TfidfVectorizer()
# vectorizer2 = TfidfVectorizer()
# vectorizer3 = TfidfVectorizer()
# x1 = vectorizer1.fit_transform(x1)
# x2 = vectorizer2.fit_transform(x2)
# x3 = vectorizer3.fit_transform(x3)

# # %%
# # Split the dataset into training and testing sets
# X1_train, X1_test, Y1_train, Y1_test = train_test_split(x1, y, test_size=0.2, stratify=y, random_state=2)
# X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y, test_size=0.2, stratify=y, random_state=2)
# X3_train, X3_test, Y3_train, Y3_test = train_test_split(x3, y, test_size=0.2, stratify=y, random_state=2)

# # %%
# # Train Logistic Regression models
# model1 = LogisticRegression()
# model1.fit(X1_train, Y1_train)
# model2 = LogisticRegression()
# model2.fit(X2_train, Y2_train)
# model3 = LogisticRegression()
# model3.fit(X3_train, Y3_train)

# # %%
# # Train Naive Bayes models
# nb_model1 = MultinomialNB()
# nb_model1.fit(X1_train, Y1_train)
# nb_model2 = MultinomialNB()
# nb_model2.fit(X2_train, Y2_train)
# nb_model3 = MultinomialNB()
# nb_model3.fit(X3_train, Y3_train)

# # %%
# # Train k-NN models
# knn_model1 = KNeighborsClassifier(n_neighbors=5)
# knn_model1.fit(X1_train, Y1_train)
# knn_model2 = KNeighborsClassifier(n_neighbors=5)
# knn_model2.fit(X2_train, Y2_train)
# knn_model3 = KNeighborsClassifier(n_neighbors=5)
# knn_model3.fit(X3_train, Y3_train)

# # %%
# # Save the models and vectorizers
# joblib.dump(model1, 'LogisticRegression_Title.pkl')
# joblib.dump(model2, 'LogisticRegression_Text.pkl')
# joblib.dump(model3, 'LogisticRegression_Title+Text.pkl')
# joblib.dump(nb_model1, 'NaiveBayes_Title.pkl')
# joblib.dump(nb_model2, 'NaiveBayes_Text.pkl')
# joblib.dump(nb_model3, 'NaiveBayes_Title+Text.pkl')
# joblib.dump(knn_model1, 'KNN_Title.pkl')
# joblib.dump(knn_model2, 'KNN_Text.pkl')
# joblib.dump(knn_model3, 'KNN_Title+Text.pkl')
# joblib.dump(vectorizer1, 'Title_vectorizer.pkl')
# joblib.dump(vectorizer2, 'Text_vectorizer.pkl')
# joblib.dump(vectorizer3, 'Title+Text_vectorizer.pkl')

# %%
import joblib
import streamlit as st

# %%
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

# %%
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


