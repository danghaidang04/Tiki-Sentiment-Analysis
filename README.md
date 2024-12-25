# Tiki-Sentiment-Analysis

Tiki Sentiment Analysis is a deep learning-based project that leverages the **PhoBERT model** to accurately classify customer reviews on the Tiki e-commerce platform as **positive**, **negative**, or **neutral**. This project aims to provide actionable insights into customer sentiment, helping businesses improve their services.

---

## Features
- **Sentiment Classification**: Classifies text into three categories: positive, negative, or neutral.
- **PhoBERT Model**: Utilizes the state-of-the-art Vietnamese language processing model.
- **Streamlit Interface**: Deploys a user-friendly web application for sentiment analysis.
- **Customizable**: Flexible to adapt for other Vietnamese language sentiment analysis use cases.

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   # Make a local clone of the repository
   mkdir Tiki Sentiment Analysis
   # Clone from my github repository and change directory to the project folder
   git clone https://github.com/danghaidang04/Tiki-Sentiment-Analysis.git
   cd Tiki-Sentiment-Analysis
    ```
2. **Install the dependencies**:
    ```bash
   pip install transformers torch streamlit plotly wordcloud
    ```
3. **Run the Streamlit application**:
    ```bash
   streamlit run alternative_app.py
    ```
4. **Access the web application**:
    Open the provided URL in your web browser to access the Streamlit application.
5. **Input the text**:
    Enter the text you want to analyze in the input box and click the "Analyze" button.