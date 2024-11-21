import time
import requests
import emoji
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline, AutoTokenizer
import praw

# Use a fine-tuned distilbert model for sentiment analysis (fine-tuned on SST-2 dataset)
english_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Preprocessing function to clean text
def preprocess_text(text):
    if text is None:
        return ""
    # Clean up URLs, mentions, and extra spaces
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Remove mentions (@username)
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub(r":[a-z_]+:", lambda m: m.group(0).replace("_", " "), text)  # Clean up emoji text
    return text.strip()

def analyze_sentiment(text):
    # Ensure the text is a valid string
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Unknown", 0

    # Preprocess the text (clean URLs, mentions, spaces, emojis)
    text = preprocess_text(text)

    # Perform sentiment analysis
    sentiment_result = english_sentiment(text)[0]
    sentiment = sentiment_result['label']
    score = sentiment_result['score']

    # Return sentiment and score
    return sentiment, score

# Set up Reddit API client
reddit = praw.Reddit(
    client_id='Ls7FVXPiUTZAsKUX3Qvc8A',
    client_secret='tW0MKkSLG3gJok5gAINeIoJr4bIPtA',
    user_agent='SentimentAnalysisApp:v2.0 (by u/--fei--)'
)

def fetch_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=limit):
        posts.append(post.title)  # Collect post titles (you can also get other content like post.body)

    return posts

# Analyze posts for sentiment and create DataFrame
def analyze_subreddit(subreddit, limit=100):
    posts = fetch_posts(subreddit, limit)
    analyzed_posts = []

    for post in posts:
        sentiment, score = analyze_sentiment(post)
        analyzed_posts.append({"post": post, "sentiment": sentiment, "score": score})

    # Return a DataFrame with the analyzed data
    return pd.DataFrame(analyzed_posts)

# Visualize sentiment distribution
def visualize_sentiment(df):
    # Pie chart for sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct="%1.1f%%", startangle=270, colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Sentiment Distribution")
    plt.ylabel("")
    plt.savefig("sentiment_distribution.png")  # Save as an image file
    plt.close()  # Close the plot to prevent memory leaks

    # Word cloud for positive posts
    positive_posts = df["post"][df["sentiment"] == "POSITIVE"]
    if not positive_posts.empty:
        stop_words = ["https", "co", "RT"] + list(STOPWORDS)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stop_words).generate(
            " ".join(positive_posts)
        )
        plt.figure()
        plt.title("Word Cloud - Positive Posts")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("positive_wordcloud.png")  # Save as an image file
        plt.close()  # Close the plot to prevent memory leaks
    else:
        print("No positive posts found to generate a word cloud.")

# Main execution
if __name__ == "__main__":
    subreddit = "soap"  # Replace with any subreddit you want to analyze
    limit = 100  # Number of posts to analyze

    print(f"Fetching {limit} posts from subreddit '{subreddit}'...")
    df = analyze_subreddit(subreddit, limit=limit)

    print("Analyzing posts for sentiment...")
    print(df.head())  # Display the first few rows of the DataFrame for review

    print("Visualizing sentiment distribution...")
    visualize_sentiment(df)

    # Display sample results
    print("\nSample Positive Post:")
    positive_posts = df[df["sentiment"] == "POSITIVE"]
    if not positive_posts.empty:
        print(positive_posts.head(1)["post"].values[0])

    print("\nSample Neutral Post:")
    neutral_posts = df[df["sentiment"] == "NEUTRAL"]
    if not neutral_posts.empty:
        print(neutral_posts.head(1)["post"].values[0])

    print("\nSample Negative Post:")
    negative_posts = df[df["sentiment"] == "NEGATIVE"]
    if not negative_posts.empty:
        print(negative_posts.head(1)["post"].values[0])
