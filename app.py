from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import PyPDF2
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from transformers import pipeline
import re
from dotenv import load_dotenv
import tweepy
import praw
import math
import string
from collections import Counter
import logging

# ----------------- Project Setup & Configuration -----------------
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO)

# ----------------- AI Models -----------------
try:
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    logging.info("Hugging Face models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Hugging Face models: {e}")
    sentiment_model = None
    classifier = None
    emotion_model = None

# ----------------- API Clients -----------------
def create_api_clients():
    clients = {}
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if YOUTUBE_API_KEY:
        try:
            from googleapiclient.discovery import build
            clients['youtube'] = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        except Exception as e:
            logging.error(f"Failed to create YouTube API client: {e}")

    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_TOKEN_KEY")
    if TWITTER_BEARER_TOKEN:
        try:
            clients['twitter'] = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        except Exception as e:
            logging.error(f"Failed to create Twitter API client: {e}")

    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    if REDDIT_CLIENT_ID:
        try:
            clients['reddit'] = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=os.getenv("REDDIT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT")
            )
        except Exception as e:
            logging.error(f"Failed to create Reddit API client: {e}")
    
    FB_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
    if FB_ACCESS_TOKEN:
        clients['facebook_token'] = FB_ACCESS_TOKEN

    return clients

api_clients = create_api_clients()

# ----------------- Data Fetching Functions -----------------
def clean_text(text):
    text = re.sub(r'http\S+|@\w+|#\S+', '', text)
    text = re.sub(r'\b(not|no)\s+(\w+)', r'not_\2', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def fetch_youtube_data_from_url(url):
    video_id_match = re.search(r"(?:v=|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})", url)
    video_id = video_id_match.group(1) if video_id_match else None
    if not video_id or 'youtube' not in api_clients: return [], {}

    posts, video_meta = [], {}
    try:
        response = api_clients['youtube'].videos().list(part="snippet,statistics", id=video_id).execute()
        if response["items"]:
            snippet, stats = response["items"][0]["snippet"], response["items"][0]["statistics"]
            posts.append({
                "text": snippet.get('title', '') + " " + snippet.get('description', ''),
                "likes": int(stats.get("likeCount", 0)),
                "views": int(stats.get("viewCount", 0)),
                "platform": "youtube"
            })
            video_meta = {"platform": "youtube", "views": posts[0].get("views", 0)}
        
        response = api_clients['youtube'].commentThreads().list(part="snippet", videoId=video_id, maxResults=10, textFormat="plainText").execute()
        for comment in response.get("items", []):
            posts.append({
                "text": comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                "platform": "youtube"
            })
    except Exception as e:
        logging.error(f"YouTube API error: {e}")
    return posts, video_meta

def fetch_twitter_data_from_url(url):
    tweet_id_match = re.search(r"twitter\.com/\w+/status/(\d+)", url)
    tweet_id = tweet_id_match.group(1) if tweet_id_match else None
    if not tweet_id or 'twitter' not in api_clients: return []
    
    try:
        response = api_clients['twitter'].get_tweet(id=tweet_id, tweet_fields=['public_metrics'])
        if response.data:
            tweet = response.data
            return [{
                "text": tweet.text,
                "likes": tweet.public_metrics.get("like_count", 0),
                "retweets": tweet.public_metrics.get("retweet_count", 0),
                "replies": tweet.public_metrics.get("reply_count", 0),
                "platform": "twitter"
            }]
    except Exception as e:
        logging.error(f"Error fetching single tweet: {e}")
    return []

def fetch_reddit_data_from_url(url):
    post_id_match = re.search(r"reddit\.com/r/\w+/comments/(\w+)", url)
    post_id = post_id_match.group(1) if post_id_match else None
    if not post_id or 'reddit' not in api_clients: return []
    
    try:
        submission = api_clients['reddit'].submission(id=post_id)
        return [{
            "text": submission.title + " " + (submission.selftext or ""),
            "upvotes": submission.score,
            "comments": submission.num_comments,
            "platform": "reddit"
        }]
    except Exception as e:
        logging.error(f"Error fetching single reddit post: {e}")
    return []

def fetch_twitter_data_by_keyword(keyword):
    if 'twitter' not in api_clients: return []
    tweets_data = []
    try:
        response = api_clients['twitter'].search_recent_tweets(
            query=keyword, max_results=10, tweet_fields=['public_metrics']
        )
        for tweet in response.data or []:
            tweets_data.append({
                "text": tweet.text,
                "likes": tweet.public_metrics.get("like_count", 0),
                "retweets": tweet.public_metrics.get("retweet_count", 0),
                "replies": tweet.public_metrics.get("reply_count", 0),
                "platform": "twitter"
            })
    except Exception as e:
        logging.error(f"Twitter API error: {e}")
    return tweets_data

def fetch_reddit_data_by_subreddit(subreddit_name):
    if 'reddit' not in api_clients: return []
    posts_data = []
    try:
        for post in api_clients['reddit'].subreddit(subreddit_name).hot(limit=10):
            posts_data.append({
                "text": post.title + " " + (post.selftext or ""),
                "upvotes": post.score,
                "comments": post.num_comments,
                "platform": "reddit"
            })
    except Exception as e:
        logging.error(f"Reddit API error: {e}")
    return posts_data

def fetch_facebook_posts(page_id):
    if 'facebook_token' not in api_clients: return []
    posts_data = []
    try:
        url = f"https://graph.facebook.com/v17.0/{page_id}/posts?fields=message,likes.summary(true),comments.summary(true),shares&access_token={api_clients['facebook_token']}&limit=5"
        response = requests.get(url).json()
        for post in response.get("data", []):
            posts_data.append({
                "text": post.get("message", ""),
                "likes": post.get("likes", {}).get("summary", {}).get("total_count", 0),
                "comments": post.get("comments", {}).get("summary", {}).get("total_count", 0),
                "shares": post.get("shares", {}).get("count", 0),
                "platform": "facebook"
            })
    except Exception as e:
        logging.error(f"Facebook API error: {e}")
    return posts_data

# ----------------- Analysis Functions -----------------
def analyze_posts(posts):
    if not sentiment_model: return "ERROR", 0, []
    
    pos_score, neg_score = 0, 0
    sentiment_results = []
    for p in posts:
        if not p.get('text'): continue
        res = sentiment_model(p['text'][:512])[0]
        engagement = sum([p.get(k,0) for k in ["likes", "comments", "shares", "retweets", "upvotes"]])
        weighted_score = res['score'] * (1 + math.log1p(engagement))
        
        if res['label'] == "POSITIVE":
            pos_score += weighted_score
        else:
            neg_score += weighted_score
        
        sentiment_results.append({**p, "sentiment": res})
        
    total_score = pos_score + neg_score
    sentiment = "POSITIVE" if pos_score >= neg_score else "NEGATIVE"
    sentiment_score = round((pos_score / total_score) * 100, 2) if total_score > 0 else 50
    return sentiment, sentiment_score, sentiment_results

def perform_absa(text):
    if not classifier or not sentiment_model: return {}
    aspect_sentiments = {}
    candidate_labels = ["product quality", "customer service", "shipping time", "price", "design"]
    
    chunks = re.split(r'\.|\n', text)
    for chunk in chunks:
        if not chunk.strip(): continue
        
        classification = classifier(chunk, candidate_labels, multi_label=True)
        for i, aspect in enumerate(classification['labels']):
            if classification['scores'][i] > 0.5:
                sentiment_res = sentiment_model(chunk)
                sentiment = sentiment_res[0]['label']
                aspect_sentiments.setdefault(aspect, {"positive": 0, "negative": 0})
                if sentiment == "POSITIVE":
                    aspect_sentiments[aspect]["positive"] += 1
                else:
                    aspect_sentiments[aspect]["negative"] += 1
    
    absa_summary = {}
    for aspect, data in aspect_sentiments.items():
        total = data["positive"] + data["negative"]
        if total > 0:
            sentiment = "POSITIVE" if data["positive"] >= data["negative"] else "NEGATIVE"
            score = (data["positive"] / total)
            absa_summary[aspect] = {"sentiment": sentiment, "score": score}
            
    return absa_summary

def extract_keywords(text):
    words = re.findall(r'\b\w{3,}\b', text.lower())
    word_counts = Counter(words)
    return dict(word_counts.most_common(10))

# ----------------- Routes -----------------
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == "POST":
        user_info = {k: request.form.get(k, "") for k in ["name", "age", "address", "mobile"]}
        return render_template("data_form.html", user_info=user_info)
    return render_template("index.html")

@app.route('/submit_data', methods=['POST'])
def submit_data():
    data_type = request.form.get("data_type")
    posts, extracted_text, video_meta = [], "", {}

    try:
        if data_type in ["csv", "pdf", "ss"]:
            file = request.files.get("file")
            if not file: return "⚠️ No file was uploaded.", 400
            
            if data_type == "csv":
                df = pd.read_csv(file)
                extracted_text = " ".join(df.astype(str).values.flatten())
            elif data_type == "pdf":
                reader = PyPDF2.PdfReader(file)
                extracted_text = " ".join(page.extract_text() or "" for page in reader.pages)
            elif data_type == "ss":
                img = Image.open(file).convert("RGB")
                extracted_text = pytesseract.image_to_string(img)
            
            posts = [{"text": extracted_text}]
        else:
            input_value = request.form.get("url") or request.form.get("keyword") or request.form.get("subreddit") or request.form.get("fb_page")
            
            if not input_value:
                return "⚠️ No input value was provided.", 400
            
            if data_type == "url":
                if "youtube.com" in input_value or "youtu.be" in input_value:
                    posts, video_meta = fetch_youtube_data_from_url(input_value)
                elif "twitter.com" in input_value:
                    posts = fetch_twitter_data_from_url(input_value)
                elif "reddit.com" in input_value:
                    posts = fetch_reddit_data_from_url(input_value)
                else: # General URL
                    response = requests.get(input_value, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(response.text, 'html.parser')
                    extracted_text = " ".join(p.get_text() for p in soup.find_all("p"))
                    posts = [{"text": extracted_text}]
            elif data_type == "twitter":
                posts = fetch_twitter_data_by_keyword(input_value)
            elif data_type == "reddit":
                posts = fetch_reddit_data_by_subreddit(input_value)
            elif data_type == "facebook":
                posts = fetch_facebook_posts(input_value)
        
        if not extracted_text:
            extracted_text = " ".join([p["text"] for p in posts if p.get("text")])
        extracted_text = clean_text(extracted_text)
        
        if not extracted_text.strip():
            return render_template("result.html", message="⚠️ No text found in the provided data.")

        sentiment, sentiment_score, sentiment_results = analyze_posts(posts)
        
        is_harmful, harmful_category, harmful_score = False, "", 0
        if classifier:
            classification = classifier(extracted_text[:512], ["cyberbullying", "misinformation", "fake"], multi_label=True)
            if classification['scores'][0] >= 0.70:
                is_harmful = True
                harmful_category = classification['labels'][0].capitalize()
                harmful_score = round(classification['scores'][0] * 100, 2)

        if is_harmful:
            category, category_score = harmful_category, harmful_score
        else:
            classification = classifier(extracted_text[:512], ["technology", "politics", "sports", "entertainment", "science", "business"])
            category, category_score = classification["labels"][0].capitalize(), round(classification["scores"][0]*100, 2)
        
        absa_results = perform_absa(extracted_text)
        emotion_results = emotion_model(extracted_text) if emotion_model else []
        keywords = extract_keywords(extracted_text)
        
        return render_template(
            "result.html",
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            category=category,
            category_score=category_score,
            extracted_text=extracted_text[:2000],
            video_meta=video_meta,
            sentiment_results=sentiment_results,
            absa_results=absa_results,
            emotion_results=emotion_results,
            keywords=keywords
        )

    except Exception as e:
        logging.error(f"Error in submit_data route: {e}")
        return render_template("result.html", message=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True)