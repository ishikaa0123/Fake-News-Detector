import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

# Sidebar navigation
page = st.sidebar.radio("Go to", ["About", "Home"])

# HOME PAGE
if page == "Home":
    st.title("📰 Fake News Detection App")
    st.write("Enter any news content below to check if it's real or fake.")

    user_input = st.text_area("Enter News Text:", height=200)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            input_vector = vectorizer.transform([user_input])
            result = model.predict(input_vector)[0]
            if result == 1:
                st.success("✅ This looks like **REAL** news.")
            else:
                st.error("🚨 This looks like **FAKE** news.")

# ABOUT PAGE
elif page == "About":
    st.title("📘 About Fake News Detector")
    st.markdown("""
    This web app was built as part of a Machine Learning project to classify fake and real news articles.

   - **Algorithm Used**: Logistic Regression
   - **Vectorization**: TF-IDF (Term Frequency–Inverse Document Frequency)
   - **Dataset**: A labeled dataset of real and fake news articles
   - **Language**: Python
   - **Libraries**: Streamlit, scikit-learn, pandas, nltk

   - **Built by**: *Ishika Bhattacharjee*
   - **References**: YouTube, Kaggle.

    
    You can test your own news texts in the "Home" section.
                

    This app is part of my self-initiated learning project on real-world machine learning applications.  
    For questions, feel free to reach out on [GitHub](https://github.com/ishikaa0123) or [LinkedIn](www.linkedin.com/in/ishika-bhattacharjee-117375254).
    """)
    
