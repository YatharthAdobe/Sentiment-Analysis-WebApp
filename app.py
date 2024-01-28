import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import base64

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def process_csv_file(file):
    df = pd.read_csv(file)

    # Perform sentiment analysis on each sentence
    df['Sentiment'] = df['Sentence'].apply(analyze_sentiment)

    return df

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def get_sentiment_color(sentiment):
    if sentiment == 'Positive':
        return 'Positive'
    elif sentiment == 'Negative':
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.set_page_config(page_title="Sentiment Analysis Web App", layout="wide")

    # Header and Welcome Message
    st.title("Sentiment Analysis Web App")
    st.markdown("""
        Welcome! This web app performs sentiment analysis on text input and CSV files.
        Feel free to use the options below to analyze sentiment and explore visualizations.
    """)

    # User input for sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Analyze Sentiment Button
    if st.button("Analyze Sentiment"):
        results = []

        # Perform sentiment analysis on user input
        if user_input:
            sentiment_label = analyze_sentiment(user_input)
            results.append({'Sentence': user_input, 'Sentiment': sentiment_label})

        # Process CSV file and perform sentiment analysis
        if uploaded_file:
            df = process_csv_file(uploaded_file)
            results.extend(df.to_dict(orient='records'))

        # Display the results
        st.markdown("### Sentiment Analysis Results:")
        results_df = pd.DataFrame(results)

        # Change color of sentiment labels
        results_df['Sentiment'] = results_df['Sentiment'].apply(get_sentiment_color)

        # Set no color for Sentiment column
        results_table_style = results_df.style.applymap(lambda x: '')

        st.dataframe(results_table_style, height=300)

        # Create and display a pie chart
        st.markdown("### Sentiment Distribution:")
        fig = px.pie(results_df, names='Sentiment', title='Sentiment Distribution', color='Sentiment',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'amber'})
        st.plotly_chart(fig)

        # Generate and display a word cloud
        if user_input or uploaded_file:
            st.markdown("### Word Cloud:")
            text_for_wordcloud = ' '.join(results_df['Sentence'])
            wordcloud_plot = generate_wordcloud(text_for_wordcloud)
            st.pyplot(wordcloud_plot)

        # Option to download the results as CSV
        if st.button("Download Results as CSV"):
            st.download_button(label="Download CSV", data=results_df.to_csv(index=False), file_name="sentiment_results.csv", key='csv_download')

if __name__ == "__main__":
    main()
