
import streamlit as st
import plotly.express as px
import pandas as pd
from sentiment_analyzer import classify_sentiment
from tiki_scraper import extract_product_id, scrape_tiki_reviews
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# --- Custom CSS ---
st.markdown("""
    <style>
        /* General page styling */
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
        }
        .main-header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #6c63ff;
            margin-top: -30px;
        }
        .menu-bar {
            background-color: #6c63ff;
            padding: 10px;
            color: white;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #6c63ff;
            color: white;
        }
        .footer a {
            color: #ffd700;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Sentiment Analysis App</h1>', unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.selectbox(
    "Choose an Application",
    ["Normal Sentiment Analysis", "Tiki Review Analysis"]
)

# --- Normal Sentiment Analysis ---
if menu == "Normal Sentiment Analysis":
    st.subheader("🧠 Vietnamese Sentiment Analysis")
    user_input = st.text_area("Enter Your Text Below:", height=150, placeholder="Type your Vietnamese text here...")

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
            st.markdown("### Confidence Scores")
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

# --- Tiki Review Sentiment Analysis ---
elif menu == "Tiki Review Analysis":
    st.subheader("🛒 Tiki Product Review Sentiment Analysis")
    st.write("Fetch and analyze customer reviews from Tiki products.")

    # User input for Tiki product URL
    tiki_url = st.text_input("Enter Tiki Product URL:")
    if st.button("Fetch & Analyze Reviews"):
        if tiki_url.strip():
            # Extract product ID
            product_id = extract_product_id(tiki_url)
            if product_id:
                st.info(f"Extracted Product ID: {product_id}")

                # Fetch reviews
                with st.spinner("Fetching reviews..."):
                    reviews_df = scrape_tiki_reviews(product_id)

                if not reviews_df.empty:
                    st.success(f"Fetched {len(reviews_df)} reviews successfully!")

                    # Perform sentiment analysis
                    with st.spinner("Analyzing sentiment..."):
                        reviews_df["Sentiment"], review_probabilities = zip(*reviews_df["Review"].apply(classify_sentiment))

                    # Display results
                    st.write("### Sentiment Analysis Results")
                    st.dataframe(reviews_df)

                    # Enhanced bar chart for sentiment distribution
                    st.write("### Sentiment Distribution")
                    sentiment_counts = reviews_df["Sentiment"].value_counts()
                    # st.bar_chart(sentiment_counts) # For the previous code

                    # Convert to a DataFrame for better control in Plotly
                    sentiment_df = sentiment_counts.reset_index()
                    sentiment_df.columns = ["Sentiment", "Count"]

                    # Customize the bar chart
                    fig = px.bar(
                        sentiment_df,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment",
                        color_discrete_map={"Positive": "green", "Neutral": "yellow", "Negative": "red"},
                        labels={"Sentiment": "Review Sentiment", "Count": "Number of Reviews"},
                        title="Sentiment Distribution of Reviews",
                        template="plotly_dark",  # Choose template or use default
                        text="Count",  # Display count on top of each bar
                    )

                    # Customize the layout further
                    fig.update_layout(
                        title="Sentiment Distribution of Reviews",
                        xaxis_title="Sentiment",
                        yaxis_title="Number of Reviews",
                        font=dict(family="Arial, sans-serif", size=14),
                        plot_bgcolor="#f9f9f9",  # Set background color to light gray
                        paper_bgcolor="#f9f9f9",  # Paper background color
                        title_x=0.5,  # Center the title
                        showlegend=False  # Hide the legend if unnecessary
                    )

                    # Show the figure in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


                    # Show the timeline chart
                    st.write("### Timeline Chart")

                    # Ensure 'Date' column is in datetime format
                    reviews_df['Date'] = pd.to_datetime(reviews_df['Date'], errors='coerce')  # Handle any invalid dates

                    # Group the data by year-month and sentiment, then count the number of reviews
                    reviews_df['Year-Month'] = reviews_df['Date'].dt.to_period('M').astype(
                        str)  # Convert Period to string (YYYY-MM)

                    # Group by 'Year-Month' and 'Sentiment' to count the number of reviews for each sentiment type
                    sentiment_over_time = reviews_df.groupby(['Year-Month', 'Sentiment']).size().reset_index(
                        name='Count')

                    # Create line chart with custom colors
                    fig = px.line(
                        sentiment_over_time,
                        x="Year-Month",
                        y="Count",
                        color="Sentiment",
                        title="Sentiment Over Time (Monthly Breakdown)",
                        labels={"Count": "Number of Reviews", "Year-Month": "Month", "Sentiment": "Review Sentiment"},
                        line_shape="linear",  # To have straight lines between points
                        template="plotly_dark",
                        color_discrete_map={"Positive": "green", "Neutral": "yellow", "Negative": "red"}
                        # Custom colors for each sentiment
                    )

                    # Customize layout
                    fig.update_layout(
                        title="Sentiment Distribution Over Time (Monthly)",
                        xaxis_title="Month",
                        yaxis_title="Number of Reviews",
                        xaxis=dict(
                            tickmode='array',  # Use specific tick values
                            tickvals=sentiment_over_time['Year-Month'].unique(),
                            # Set the x-tick values to the unique Year-Month
                            tickangle=90,  # Rotate x-tick labels for readability
                            dtick=1,  # This controls the spacing between the x-axis ticks, 1 means one per month
                            tickfont=dict(size=10),  # Set smaller font size for x-axis ticks
                            showgrid=True,  # Show grid lines on the x-axis
                            zeroline=True  # Hide the zero line on the x-axis
                        ),
                        yaxis=dict(
                            showgrid=True,  # Show grid lines on the y-axis
                            zeroline=True  # Hide the zero line on the y-axis
                        ),
                        font=dict(family="Arial, sans-serif", size=14),
                        plot_bgcolor="#f9f9f9",  # Set background color to light gray
                        paper_bgcolor="#f9f9f9",  # Paper background color
                        title_x=0.5,  # Center the title
                        showlegend=True  # Show legend for sentiment lines
                    )

                    # Show chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


                    st.write("### Most Frequent Terms")

                    # Step 1: Preprocess reviews to get words for each sentiment category
                    positive_reviews = ' '.join(reviews_df[reviews_df['Sentiment'] == 'Positive']['Review'])
                    negative_reviews = ' '.join(reviews_df[reviews_df['Sentiment'] == 'Negative']['Review'])
                    neutral_reviews = ' '.join(reviews_df[reviews_df['Sentiment'] == 'Neutral']['Review'])

                    # Step 2: Split reviews into words for each sentiment
                    positive_words = set(positive_reviews.split())
                    negative_words = set(negative_reviews.split())
                    neutral_words = set(neutral_reviews.split())

                    # Step 3: Find the common words that appear in all three sentiments
                    common_words = positive_words & negative_words & neutral_words


                    # Step 4: Remove common words from each sentiment's reviews
                    def remove_common_words(text, common_words):
                        # Remove words that are in common_words
                        words = text.split()
                        filtered_words = [word for word in words if word.lower() not in common_words]
                        return ' '.join(filtered_words)


                    # Apply the function to remove common words
                    all_reviews_filtered = ' '.join([
                        remove_common_words(positive_reviews, common_words),
                        remove_common_words(negative_reviews, common_words),
                        remove_common_words(neutral_reviews, common_words)
                    ])

                    # Step 5: Generate word cloud for the filtered reviews (after removing common words)
                    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(
                        all_reviews_filtered)

                    # Step 6: Display the word cloud using Matplotlib
                    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.set_title("Word Cloud (After Removing Common Words)", fontsize=16)
                    ax.axis('off')  # Hide axes

                    # Show the word cloud
                    st.pyplot(fig)


                else:
                    st.warning("No reviews found for this product.")
            else:
                st.error("Could not extract Product ID from the provided URL. Please check the URL format.")
        else:
            st.warning("Please enter a valid Tiki product URL.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed with ❤️ using <a href="https://huggingface.co/" target="_blank">Hugging Face</a> and <a href="https://streamlit.io/" target="_blank">Streamlit</a>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)