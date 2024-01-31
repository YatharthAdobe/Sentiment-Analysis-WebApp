import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
import base64

# Download stopwords dataset
import nltk
nltk.download('stopwords')

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

def most_frequent_words(text, num_words=10):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in text.split() if word.isalpha() and word.lower() not in stop_words]
    word_counts = Counter(words)
    common_words = dict(word_counts.most_common(num_words))
    return common_words

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt


def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    st.set_page_config(page_title="Sentiment Analysis Checker", layout="wide")
    img = get_img_as_base64("bg2.jpg")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img}");
    background-position: center;
    background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Header and Welcome Message
    st.title("Sentiment Analysis Checker")
    st.markdown("""
        Leverage Sentiment Analysis to quickly fetch overall feeling from a text or large pool of reviews.

        Plug in or send us your sample data, and we will take care of the rest. Valence Aware Dictionary for Sentiment Reasoning (VADER) Model generates the insights in a matter of minutes.
        Feel free to use the options below to analyze sentiment and explore visualizations.
    """)
    st.divider()
    # Navigation bar
    nav_option = option_menu(
        None,
        ["Single Text", "Bulk Text (Upload CSV)"],
        default_index=0,
        orientation="horizontal"
    )

    # User input for sentiment analysis
    if nav_option == "Single Text":
        user_input = st.text_area("Enter text for sentiment analysis:")

        # Analyze Sentiment Button
        if st.button("Analyze Sentiment"):
            results = []

            # Perform sentiment analysis on user input
            if user_input:
                sentiment_label = analyze_sentiment(user_input)
                results.append({'Sentence': user_input, 'Sentiment': sentiment_label})

            # Display the results based on selected radio button
            display_results(results)

    elif nav_option == "Bulk Text (Upload CSV)":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # Analyze Sentiment Button
        if st.button("Analyze Sentiment"):
            results = []

            # Process CSV file and perform sentiment analysis
            if uploaded_file:
                df = process_csv_file(uploaded_file)
                results.extend(df.to_dict(orient='records'))

            # Display the results based on selected radio button
            display_results(results)

def display_results(results):

    # Display the sentiment analysis results in a single column
    st.markdown("## Sentiment Analysis Results:")
    results_df = pd.DataFrame(results)
    results_table_style = results_df.style.applymap(lambda x: '')
    st.dataframe(results_table_style, height=200)

    # Create a 2x2 grid layout for visualizations
    col1, col2 = st.columns(2)

    # Pie Chart in the first column
    col1.markdown("## Sentiment Distribution:")
    fig_pie = px.pie(results_df, names='Sentiment', title='Positive , Negative & Neutral Split', color='Sentiment',
                    color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'amber'},hole=.4)
    fig_pie.update_layout(showlegend=True,legend=dict(x=0, y=0, traceorder='normal', orientation='h'))  # Add this line to show the legend
    col1.plotly_chart(fig_pie)

    # Histogram in the second column
    col2.markdown("## Most Frequent Words:")
    if results:
        text_for_word_chart = ' '.join(results_df['Sentence'])
        common_words = most_frequent_words(text_for_word_chart, num_words=10)
        fig_bar = px.bar(x=list(common_words.keys()), y=list(common_words.values()),
                         labels={'x': 'Word', 'y': 'Count'}, title='Top 10 Most Frequent Important Words')
        col2.plotly_chart(fig_bar)

    # Word Cloud below both columns
    st.markdown("## Word Cloud of Most Frequent Words:")
    if results:
        text_for_wordcloud = ' '.join(results_df['Sentence'])
        wordcloud_plot = generate_wordcloud(text_for_wordcloud)
        st.pyplot(wordcloud_plot)

    # Option to download the results as CSV
    if st.button("Download Results as CSV"):
        st.download_button(label="Download CSV", data=results_df.to_csv(index=False), file_name="sentiment_results.csv", key='csv_download')

if __name__ == "__main__":
    main()
