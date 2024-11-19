import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained model and tokenizer from Hugging Face
MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    labels = ["Negative", "Positive", "Neutral"]  # Update these labels if needed
    return labels[predicted_class], torch.softmax(logits, dim=1).tolist()[0]

# Streamlit page configuration
st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add a header with background styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .title {
            font-size: 3em;
            color: #6c63ff;
            text-align: center;
        }
        .description {
            text-align: center;
            font-size: 1.2em;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">Vietnamese Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Analyze Vietnamese text and classify its sentiment as Positive, Neutral, or Negative using state-of-the-art NLP technology.</p>', unsafe_allow_html=True)

# Input section with better layout
st.markdown("---")
st.subheader("Enter Your Text Below:")
user_input = st.text_area("Type your Vietnamese text here", height=150, placeholder="Nhập văn bản tiếng Việt để phân tích...")

# Button to analyze sentiment
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        # Analyze the sentiment
        sentiment, probabilities = classify_sentiment(user_input)

        # Display the sentiment result with color coding
        st.markdown("---")
        if sentiment == "Positive":
            st.success(f"**Predicted Sentiment:** {sentiment} 😃")
        elif sentiment == "Neutral":
            st.info(f"**Predicted Sentiment:** {sentiment} 😐")
        else:
            st.error(f"**Predicted Sentiment:** {sentiment} 😠")

        # Display confidence scores with shorter bars and percentages
        st.markdown("### Confidence Scores:")
        labels = ["Negative", "Positive", "Neutral"]
        for i, label in enumerate(labels):
            percentage = f"{probabilities[i] * 100:.2f}%"
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <span style="width: 100px; font-weight: bold;">{label}</span>
                    <div style="flex-grow: 1; margin-left: 10px; margin-right: 10px;">
                        <progress style="width: 80%;" value="{probabilities[i]}" max="1"></progress>
                    </div>
                    <span style="width: 50px; text-align: right;">{percentage}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.warning("🚨 Please enter some text to analyze.")

# Footer with credits
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed with ❤️ using <a href="https://huggingface.co/" target="_blank">Hugging Face</a> and <a href="https://streamlit.io/" target="_blank">Streamlit</a>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
