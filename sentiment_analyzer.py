from transformers import pipeline
import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing sentiment analysis models...")
        try:
            # Initialize RoBERTa sentiment analyzer
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # Use CPU. Change to 0 for GPU
            )
            
            # Initialize emotion classifier
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=-1  # Use CPU. Change to 0 for GPU
            )
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def get_sentiment_scores(self, text):
        """Get sentiment scores using RoBERTa"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {'label': 'NEUTRAL', 'score': 0.5}
            
            # Truncate text if it's too long (model limit is usually 512 tokens)
            text = ' '.join(text.split()[:300])
            result = self.sentiment_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': float(result['score'])
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def get_emotion_scores(self, text):
        """Get emotion scores"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {}
            
            # Truncate text if it's too long
            text = ' '.join(text.split()[:300])
            emotions = self.emotion_pipeline(text)[0]
            return {emotion['label']: float(emotion['score']) for emotion in emotions}
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {}

    def get_textblob_sentiment(self, text):
        """Get additional sentiment metrics using TextBlob"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            analysis = TextBlob(text)
            return {
                'polarity': float(analysis.sentiment.polarity),
                'subjectivity': float(analysis.sentiment.subjectivity)
            }
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}

    def analyze_sentiment(self, text):
        """Complete sentiment analysis"""
        if not isinstance(text, str) or not text.strip():
            return {
                'roberta_sentiment': {'label': 'NEUTRAL', 'score': 0.5},
                'emotions': {},
                'textblob': {'polarity': 0.0, 'subjectivity': 0.0}
            }

        # Combine all sentiment analyses
        sentiment_scores = self.get_sentiment_scores(text)
        emotion_scores = self.get_emotion_scores(text)
        textblob_scores = self.get_textblob_sentiment(text)

        return {
            'roberta_sentiment': sentiment_scores,
            'emotions': emotion_scores,
            'textblob': textblob_scores
        }

    def analyze_dataframe(self, df):
        """Analyze sentiment for all content in DataFrame"""
        try:
            logger.info("Starting sentiment analysis on DataFrame...")
            df_sentiment = df.copy()
            
            # Create a progress bar
            tqdm.pandas(desc="Analyzing sentiments")
            
            # Analyze sentiment for processed content
            logger.info("Performing sentiment analysis...")
            sentiment_results = df_sentiment['processed_content'].progress_apply(self.analyze_sentiment)
            
            # Extract sentiment features into separate columns
            logger.info("Extracting sentiment features...")
            
            # RoBERTa sentiment
            df_sentiment['sentiment_label'] = sentiment_results.apply(
                lambda x: x['roberta_sentiment']['label'])
            df_sentiment['sentiment_score'] = sentiment_results.apply(
                lambda x: x['roberta_sentiment']['score'])
            
            # Emotion scores
            emotion_columns = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
            for emotion in emotion_columns:
                df_sentiment[f'emotion_{emotion}'] = sentiment_results.apply(
                    lambda x: x['emotions'].get(emotion, 0.0))
            
            # TextBlob scores
            df_sentiment['textblob_polarity'] = sentiment_results.apply(
                lambda x: x['textblob']['polarity'])
            df_sentiment['textblob_subjectivity'] = sentiment_results.apply(
                lambda x: x['textblob']['subjectivity'])
            
            # Calculate aggregate sentiment metrics
            df_sentiment['sentiment_intensity'] = abs(df_sentiment['textblob_polarity'])
            
            logger.info("Sentiment analysis completed successfully")
            return df_sentiment
            
        except Exception as e:
            logger.error(f"Error in DataFrame analysis: {e}")
            raise

    def get_sentiment_summary(self, df):
        """Generate summary statistics for sentiment analysis"""
        try:
            summary = {
                'overall_sentiment': {
                    'positive': (df['sentiment_label'] == 'POSITIVE').mean(),
                    'negative': (df['sentiment_label'] == 'NEGATIVE').mean(),
                    'neutral': (df['sentiment_label'] == 'NEUTRAL').mean()
                },
                'average_scores': {
                    'sentiment_score': df['sentiment_score'].mean(),
                    'polarity': df['textblob_polarity'].mean(),
                    'subjectivity': df['textblob_subjectivity'].mean()
                },
                'emotion_distribution': {
                    emotion: df[f'emotion_{emotion}'].mean()
                    for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
                }
            }
            return summary
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            raise