import pandas as pd
import numpy as np
from db_config import get_db_connection
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
import praw
import pymysql.cursors

# Set NLTK data path for Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {e}")
    st.stop()

# Load secrets from Streamlit Cloud
api_key = st.secrets["API_KEY"]  # Ensure this is set in Streamlit secrets with your key
reddit_client_id = st.secrets["reddit_client_id"]
reddit_client_secret = st.secrets["reddit_client_secret"]
reddit_user_agent = st.secrets["reddit_user_agent"]

if not all([api_key, reddit_client_id, reddit_client_secret, reddit_user_agent]):
    st.error("Missing secrets. Check Streamlit Cloud secrets configuration.")
    st.stop()

API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}'

# Initialize Reddit API client
try:
    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent
    )
except Exception as e:
    st.error(f"Failed to initialize Reddit API client: {e}")
    st.stop()

# Create or use existing subreddit database
@st.cache_resource
def create_subreddit_db(subreddit):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
        cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {db_name}")
        cursor.execute(f"USE {db_name}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
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

# Fetch posts from Reddit with caching
@st.cache_data(ttl=3600)
def fetch_subreddit_posts(subreddit, days_ago=120):
    try:
        sub = reddit.subreddit(subreddit[2:])
        posts = []
        cutoff_time = datetime.now() - timedelta(days=days_ago)
        for submission in sub.new(limit=500):
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
                break
        return posts
    except Exception as e:
        st.error(f"Error fetching posts from Reddit: {e}")
        return []

# Store posts in the database
def store_posts(subreddit, posts):
    db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
    conn = get_db_connection(db_name)
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

# Fetch posts from database with caching
@st.cache_data
def fetch_posts_from_db(subreddit, days_ago=120):
    db_name = f"reddit_{subreddit.replace('/', '_').lower()}"
    conn = get_db_connection(db_name)
    if not conn:
        st.error("Failed to connect to the database.")
        return []
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
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

# Sentiment analysis
@st.cache_data
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return {
        "polarity": polarity,
        "subjectivity": analysis.sentiment.subjectivity,
        "sentiment": "positive" if polarity > 0.05 else "negative" if polarity < -0.05 else "neutral"
    }

# Time series analysis
@st.cache_data
def analyze_post_trends(posts):
    if not posts:
        st.warning("No posts available for trend analysis.")
        return pd.DataFrame()
    df = pd.DataFrame(posts)
    if 'created_at' not in df.columns:
        st.error("Data from database is missing 'created_at' column.")
        return pd.DataFrame()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='post_count')
    df['sentiment_score'] = df['body'].apply(lambda x: analyze_sentiment_textblob(x)['polarity'])
    daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index(name='avg_sentiment')
    return pd.merge(daily_counts, daily_sentiment, on='date')

# Common topics analysis
@st.cache_data
def analyze_common_topics(posts, top_n=20):
    stop_words = set(stopwords.words('english')).union({"like", "just", "feel", "really", "i'm", "dont"})
    all_words = []
    for post in posts:
        text = f"{post['title']} {post['body']}".lower()
        words = [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words and len(word) > 2]
        all_words.extend(words)
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n), word_counts

# Generate visualizations
@st.cache_data
def generate_visualizations(posts, trend_data, top_words, word_counts):
    visualizations = {}
    plt.style.use('fast')

    plt.figure(figsize=(10, 5))
    plt.plot(trend_data['date'], trend_data['post_count'], marker='o')
    plt.title('Post Frequency')
    plt.xlabel('Date')
    plt.ylabel('Posts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('post_frequency.png', dpi=100)
    plt.close()
    visualizations['Post Frequency'] = 'post_frequency.png'

    plt.figure(figsize=(10, 5))
    plt.plot(trend_data['date'], trend_data['avg_sentiment'], marker='o', color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Sentiment Trends')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_trends.png', dpi=100)
    plt.close()
    visualizations['Sentiment Trends'] = 'sentiment_trends.png'

    df = pd.DataFrame(posts)
    df['sentiment'] = df['body'].apply(lambda x: analyze_sentiment_textblob(x)['sentiment'])
    plt.figure(figsize=(8, 4))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    sentiment_counts.plot(kind='bar', color=[colors[x] for x in sentiment_counts.index])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Posts')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=100)
    plt.close()
    visualizations['Sentiment Distribution'] = 'sentiment_distribution.png'

    plt.figure(figsize=(10, 6))
    words, counts = zip(*top_words)
    plt.barh(list(reversed(words)), list(reversed(counts)))
    plt.title('Top Words')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('word_frequency.png', dpi=100)
    plt.close()
    visualizations['Word Frequency'] = 'word_frequency.png'

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate_from_frequencies(dict(word_counts))
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('wordcloud.png', dpi=100)
    plt.close()
    visualizations['Word Cloud'] = 'wordcloud.png'

    return visualizations

# Statistics
@st.cache_data
def get_statistics(posts):
    df = pd.DataFrame(posts)
    return {
        'total_posts': len(posts),
        'avg_upvotes': df['upvotes'].mean(),
        'sentiment_distribution': df['body'].apply(lambda x: analyze_sentiment_textblob(x)['sentiment']).value_counts().to_dict(),
        'avg_polarity': df['body'].apply(lambda x: analyze_sentiment_textblob(x)['polarity']).mean(),
        'avg_subjectivity': df['body'].apply(lambda x: analyze_sentiment_textblob(x)['subjectivity']).mean(),
        'avg_post_length': df['body'].apply(len).mean()
    }

# Gemini API response with enhanced debugging and timeout
def get_gemini_response(user_input, context):
    headers = {'Content-Type': 'application/json'}
    prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer clearly."
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        # Add timeout to prevent hanging
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        st.write(f"API Request Status: {response.status_code}")  # Debug: Show status code
        if response.status_code == 200:
            result = response.json()
            st.write("API Response:", result)  # Debug: Show full response
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error(f"API Returned Status {response.status_code}: {response.text}")
            return f"Error: API returned status {response.status_code}"
    except requests.exceptions.Timeout:
        st.error("API request timed out after 10 seconds.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.write("Error Details:", e.response.text)
        return f"Error: Failed to connect to Gemini API - {str(e)}"
    except KeyError as e:
        st.error(f"API Response Parsing Error: {e}")
        return f"Error: Unexpected response format - {str(e)}"
    except Exception as e:
        st.error(f"Unexpected Error in Gemini API Call: {e}")
        return f"Error: {str(e)}"

# Main app
def main():
    st.title("Subreddit Analysis with Chatbot 🚀")
    st.write("Analyze subreddit posts over the last 120 days!")

    subreddit_options = ['r/anxiety', 'r/depression', 'r/mentalhealth', 'r/suicide', 'r/stress', '/rIndianStockMarket', 'r/NSEbets', 'r/wallstreetbets', 'r/investing']
    selected_subreddit = st.selectbox("Select Subreddit", subreddit_options)

    if 'current_subreddit' not in st.session_state or st.session_state['current_subreddit'] != selected_subreddit:
        with st.spinner("Fetching and processing data..."):
            db_name = create_subreddit_db(selected_subreddit)
            if db_name:
                posts = fetch_subreddit_posts(selected_subreddit)
                if posts:
                    store_posts(selected_subreddit, posts)
                st.session_state['current_subreddit'] = selected_subreddit
                st.cache_data.clear()

    posts = fetch_posts_from_db(selected_subreddit)
    if not posts:
        st.error(f"No posts found for {selected_subreddit}.")
        return

    with st.spinner("Analyzing data..."):
        trend_data = analyze_post_trends(posts)
        if trend_data.empty:
            st.warning("Trend analysis failed due to missing or malformed data.")
            return
        top_words, word_counts = analyze_common_topics(posts)
        visualizations = generate_visualizations(posts, trend_data, top_words, word_counts)
        stats = get_statistics(posts)

    context_base = (
        f"Analysis of {stats['total_posts']} posts from {selected_subreddit}:\n"
        f"- Avg Upvotes: {stats['avg_upvotes']:.2f}\n"
        f"- Sentiment: {stats['sentiment_distribution']}\n"
        f"- Avg Polarity: {stats['avg_polarity']:.4f}\n"
        f"- Top Words: {', '.join([w for w, _ in top_words])}\n"
    )

    viz_options = list(visualizations.keys())
    selected_viz = st.selectbox("Select Visualization", viz_options)
    st.header(selected_viz)
    st.image(visualizations[selected_viz], caption=f"{selected_viz} for {selected_subreddit}", use_container_width=True)

    viz_context = context_base + f"Current Visualization: {selected_viz}."
    if st.button("💬 Chat about this"):
        with st.expander("Chat with Gemini", expanded=True):
            chat_key = f"chat_history_{selected_viz}_{selected_subreddit}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []
            user_input = st.chat_input(f"Ask about {selected_viz} in {selected_subreddit}")
            if user_input:
                with st.spinner("Getting response from Gemini..."):
                    st.session_state[chat_key].append({"role": "user", "text": user_input})
                    response = get_gemini_response(user_input, viz_context)
                    st.session_state[chat_key].append({"role": "assistant", "text": response})
                    # Force a rerun to ensure chat updates
                    st.rerun()
            for message in st.session_state[chat_key]:
                with st.chat_message(message["role"]):
                    st.write(message["text"])

if __name__ == "__main__":
    main()