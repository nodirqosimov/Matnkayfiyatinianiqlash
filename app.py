import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Model va vektorizerni yuklash
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit ilovasi
st.title("Sentiment Tahlil Ilovasi")
st.write("Matn kiriting va uning pozitiv yoki negativ ekanligini aniqlang!")

# Matn kiritish
user_input = st.text_area("Matnni kiriting:", placeholder="Bu ilova juda foydali!")

if st.button("Bashorat qilish"):
    if user_input.strip() == "":
        st.error("Iltimos, matn kiriting!")
    else:
        # Matnni vektorlashtirish
        text_vectorized = vectorizer.transform([user_input])

        # Sentimentni bashorat qilish
        prediction = model.predict(text_vectorized)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'

        # Natijani ko'rsatish
        st.success(f"Sentiment: {sentiment}")




