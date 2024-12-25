
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the HuggingFace model and tokenizer
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
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    return labels[predicted_class], probabilities