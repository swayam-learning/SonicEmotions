# Reddit Sentiment Analysis & Chatbot Integration

## Overview
This project extracts and analyzes user comments from mental health-related subreddits (e.g., `r/anxiety`, `r/stress`, `r/depression`) to determine sentiment trends. It integrates an AI-powered chatbot that provides insights based on sentiment analysis results.

## Features
- **Real-time Data Extraction:** Uses PRAW to scrape top comments from selected subreddits.
- **Database Storage:** Stores structured data (comments, upvotes, timestamps) in TiDB Cloud (MySQL-based).
- **Sentiment Analysis:** Applies NLP techniques (TF-IDF, BoW, VADER) to classify comments as `positive`, `neutral`, or `negative`.
- **AI Chatbot Integration:** Uses Gemini 2.0 to explain insights based on sentiment trends.
- **Interactive Dashboard:** Built using Streamlit to visualize sentiment trends and chatbot interactions.

## Tech Stack
- **Programming Language:** Python
- **Libraries & Tools:** PRAW, Pandas, NLTK, Scikit-learn, Streamlit
- **Database:** TiDB Cloud (MySQL-based)
- **AI & NLP:** VADER, TF-IDF, BoW, Gemini 2.0
- **Deployment:** Streamlit Cloud

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reddit-sentiment-analysis.git
   cd reddit-sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Reddit API credentials (PRAW):
   - Create a `.env` file and add:
     ```
     CLIENT_ID=your_client_id
     CLIENT_SECRET=your_client_secret
     USER_AGENT=your_user_agent
     ```
4. Run the data extraction script:
   ```bash
   python extract_comments.py
   ```
5. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Project Workflow
1. **Scrape Comments:** Extracts top comments from selected subreddits.
2. **Store Data:** Saves data in a structured MySQL-based database.
3. **Analyze Sentiment:** Applies NLP techniques to classify sentiment.
4. **Generate Insights:** AI chatbot provides explanations based on analysis.
5. **Visualize Results:** Streamlit dashboard displays sentiment trends.

## Future Improvements
- Fine-tune AI chatbot for more detailed sentiment explanations.
- Expand to other subreddits and generalize the sentiment model.
- Optimize real-time data processing and storage efficiency.

## Contributors
- **Swayam Swarup Barik** – [GitHub Profile](https://github.com/swayam-learning)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
