import streamlit as st
from transformers import pipeline
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

# Load the sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Streamlit app interface
st.title("Sentiment Analysis Web App")
st.write("Enter a sentence below to determine if it's positive, negative, or neutral.")

# Text input for the user
user_input = st.text_input("Enter a sentence:")

# Button to perform sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        result = sentiment_analyzer(user_input)
        label = result[0]['label']
        score = result[0]['score']

        if label == 'POSITIVE':
            st.markdown(f"**Sentiment:** Positive (Confidence: {score * 100:.2f}%)")
        elif label == 'NEGATIVE':
            st.markdown(f"**Sentiment:** Negative (Confidence: {score * 100:.2f}%)")
        else:
            st.markdown(f"**Sentiment:** Neutral (Confidence: {score * 100:.2f}%)")
    else:
        st.warning("Please enter a sentence to analyze.")