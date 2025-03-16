import pandas as pd
import numpy as np
import pymysql
from db_config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import praw

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}'

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent
)

# Database connection
def connect_db(database=None):
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=database if database else None,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.MySQLError as e:
        st.error(f"Database Connection Error: {e}")
        return None

# Create a new database for subreddit and table
def create_subreddit_db(subreddit):
    conn = connect_db()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cursor.execute(f"CREATE DATABASE {db_name}")
        cursor.execute(f"USE {db_name}")
        cursor.execute("""
            CREATE TABLE posts (
                id VARCHAR(50) PRIMARY KEY,
                title TEXT,
                body TEXT,
                upvotes INT,
                created_at DATETIME,
                subreddit VARCHAR(100)
            )
        """)
        conn.commit()
        return db_name
    except pymysql.MySQLError as e:
        st.error(f"Database Creation Error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

# Fetch posts from Reddit API for the last 120 days
def fetch_subreddit_posts(subreddit, days_ago=30):
    try:
        sub = reddit.subreddit(subreddit[2:])  # Remove 'r/' prefix
        posts = []
        cutoff_time = datetime.now() - timedelta(days=days_ago)
        for submission in sub.new(limit=1000):  # Increased limit
            created_time = datetime.fromtimestamp(submission.created_utc)
            if created_time >= cutoff_time:
                posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'body': submission.selftext if submission.selftext else '',
                    'upvotes': submission.score,
                    'created_at': created_time,
                    'subreddit': subreddit
                })
            else:
                break  # Stop if posts are older than cutoff
        st.info(f"Fetched {len(posts)} posts from {subreddit} within the last {days_ago} days.")
        return posts
    except Exception as e:
        st.error(f"Error fetching posts from Reddit: {e}")
        return []

# Store posts in the database
def store_posts(subreddit, posts):
    db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
    conn = connect_db(db_name)
    if not conn:
        return
    try:
        cursor = conn.cursor()
        for post in posts:
            cursor.execute("""
                INSERT IGNORE INTO posts (id, title, body, upvotes, created_at, subreddit)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (post['id'], post['title'], post['body'], post['upvotes'], post['created_at'], post['subreddit']))
        conn.commit()
    except pymysql.MySQLError as e:
        st.error(f"Error storing posts: {e}")
    finally:
        cursor.close()
        conn.close()

# Fetch posts from the current subreddit database
@st.cache_data
def fetch_posts_from_db(subreddit, days_ago=365):
    db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
    conn = connect_db(db_name)
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        query = """
        SELECT * FROM posts 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        ORDER BY created_at DESC
        """
        cursor.execute(query, (days_ago,))
        posts = cursor.fetchall()
        return posts
    except pymysql.MySQLError as e:
        st.error(f"MySQL Error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Sentiment analysis with TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.05:
        sentiment = "positive"
    elif analysis.sentiment.polarity < -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"polarity": analysis.sentiment.polarity, "subjectivity": analysis.sentiment.subjectivity, "sentiment": sentiment}

# Time series analysis
@st.cache_data
def analyze_post_trends(posts):
    df = pd.DataFrame(posts)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='post_count')
    df['sentiment_score'] = df['body'].apply(lambda x: analyze_sentiment_textblob(x)['polarity'])
    daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index(name='avg_sentiment')
    return pd.merge(daily_counts, daily_sentiment, on='date')

# Common topics analysis
@st.cache_data
def analyze_common_topics(posts, top_n=20):
    stop_words = set(stopwords.words('english'))
    additional_stops = {"like", "just", "feel", "feeling", "really", "i'm", "im", "don't", "dont"}
    stop_words.update(additional_stops)
    all_words = []
    for post in posts:
        text = f"{post['title']} {post['body']}"
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n), word_counts

# Generate visualizations
@st.cache_data
def generate_visualizations(posts, trend_data, top_words, word_counts):
    visualizations = {}
    
    plt.figure(figsize=(12, 6))
    plt.plot(trend_data['date'], trend_data['post_count'], marker='o')
    plt.title('Post Frequency Over Time (Last 120 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('post_frequency.png')
    plt.close()
    visualizations['Post Frequency'] = 'post_frequency.png'

    plt.figure(figsize=(12, 6))
    plt.plot(trend_data['date'], trend_data['avg_sentiment'], marker='o', color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Average Sentiment Over Time (Last 120 Days)')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_trends.png')
    plt.close()
    visualizations['Sentiment Trends'] = 'sentiment_trends.png'

    df = pd.DataFrame(posts)
    df['sentiment'] = df['body'].apply(lambda x: analyze_sentiment_textblob(x)['sentiment'])
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    sentiment_counts.plot(kind='bar', color=[colors[x] for x in sentiment_counts.index])
    plt.title('Distribution of Sentiment in Posts (Last 120 Days)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Posts')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()
    visualizations['Sentiment Distribution'] = 'sentiment_distribution.png'

    plt.figure(figsize=(12, 8))
    words, counts = zip(*top_words)
    plt.barh(list(reversed(words)), list(reversed(counts)))
    plt.title('Top 20 Words in Posts (Last 120 Days)')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('word_frequency.png')
    plt.close()
    visualizations['Word Frequency'] = 'word_frequency.png'

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate_from_frequencies(dict(word_counts))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud (Last 120 Days)')
    plt.tight_layout()
    plt.savefig('wordcloud.png')
    plt.close()
    visualizations['Word Cloud'] = 'wordcloud.png'

    return visualizations

# Statistics
@st.cache_data
def get_statistics(posts):
    df = pd.DataFrame(posts)
    total_posts = len(posts)
    avg_upvotes = df['upvotes'].mean()
    sentiments = [analyze_sentiment_textblob(post['body']) for post in posts]
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    avg_polarity = sentiment_df['polarity'].mean()
    avg_subjectivity = sentiment_df['subjectivity'].mean()
    df['body_length'] = df['body'].apply(len)
    avg_length = df['body_length'].mean()
    return {
        'total_posts': total_posts,
        'avg_upvotes': avg_upvotes,
        'sentiment_distribution': sentiment_counts.to_dict(),
        'avg_polarity': avg_polarity,
        'avg_subjectivity': avg_subjectivity,
        'avg_post_length': avg_length
    }

# Gemini API response
def get_gemini_response(user_input, context):
    headers = {'Content-Type': 'application/json'}
    prompt = f"Given this context: {context}\n\nUser question: {user_input}\n\nProvide a clear, concise explanation."
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"Error: {e}")
        return f"Error: {e}"

# Streamlit App
def main():
    st.title("Subreddit Analysis with Chatbot 🚀")
    st.write("Select a subreddit to analyze its posts over the last 120 days!")

    # List of subreddits for the dropdown
    subreddit_options = ['r/anxiety', 'r/depression', 'r/mentalhealth', 'r/suicide', 'r/stress', '/rIndianStockMarket', 'r/NSEbets', 'r/wallstreetbets', 'r/investing']
    selected_subreddit = st.selectbox("Select Subreddit", subreddit_options)

    # Store selected subreddit in session state to detect changes
    if 'current_subreddit' not in st.session_state or st.session_state['current_subreddit'] != selected_subreddit:
        db_name = create_subreddit_db(selected_subreddit)
        if db_name:
            posts = fetch_subreddit_posts(selected_subreddit)
            store_posts(selected_subreddit, posts)
            st.session_state['current_subreddit'] = selected_subreddit
            st.cache_data.clear()

    # Fetch posts from the current subreddit's database
    posts = fetch_posts_from_db(selected_subreddit)
    if not posts:
        st.error(f"No posts found for {selected_subreddit} within the last 120 days.")
        return

    # Perform analysis
    trend_data = analyze_post_trends(posts)
    top_words, word_counts = analyze_common_topics(posts)
    visualizations = generate_visualizations(posts, trend_data, top_words, word_counts)
    stats = get_statistics(posts)

    # Context for chatbot
    context_base = (
        f"Analysis of {stats['total_posts']} posts from {selected_subreddit} over the last 120 days:\n"
        f"- Average Upvotes: {stats['avg_upvotes']:.2f}\n"
        f"- Sentiment Distribution: {stats['sentiment_distribution']}\n"
        f"- Average Sentiment Polarity: {stats['avg_polarity']:.4f}\n"
        f"- Top Words: {', '.join([w for w, _ in top_words])}\n"
    )

    # Dropdown to select visualization
    viz_options = list(visualizations.keys())
    selected_viz = st.selectbox("Select Visualization", viz_options)

    # Display selected visualization
    st.header(selected_viz)
    st.image(visualizations[selected_viz], caption=f"{selected_viz} for {selected_subreddit}", use_column_width=True)

    # Specific context for the selected visualization
    viz_context = context_base + f"Current Visualization: {selected_viz}. This shows {selected_viz.lower()} for {selected_subreddit} over the last 120 days."

    # Chatbot section
    st.subheader("Chatbot")
    if f"chat_history_{selected_viz}_{selected_subreddit}" not in st.session_state:
        st.session_state[f"chat_history_{selected_viz}_{selected_subreddit}"] = []

    user_input = st.text_input(f"Ask about {selected_viz} in {selected_subreddit}", key=f"chat_input_{selected_viz}_{selected_subreddit}")
    if st.button("Submit Question", key=f"submit_{selected_viz}_{selected_subreddit}"):
        if user_input:
            with st.spinner("Getting response from Gemini..."):
                response = get_gemini_response(user_input, viz_context)
                st.session_state[f"chat_history_{selected_viz}_{selected_subreddit}"].append({"role": "user", "text": user_input})
                st.session_state[f"chat_history_{selected_viz}_{selected_subreddit}"].append({"role": "assistant", "text": response})

    if st.session_state[f"chat_history_{selected_viz}_{selected_subreddit}"]:
        for message in st.session_state[f"chat_history_{selected_viz}_{selected_subreddit}"]:
            with st.chat_message(message["role"]):
                st.write(message["text"])

if __name__ == "__main__":
    main()