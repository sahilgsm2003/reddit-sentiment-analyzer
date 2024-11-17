from reddit_fetcher import RedditDataFetcher
from preprocessor import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentiment_analysis():
    try:
        # 1. Fetch Data
        logger.info("Fetching Reddit data...")
        fetcher = RedditDataFetcher()
        posts_data = fetcher.fetch_posts("iPhone 13", limit=3)
        df = fetcher.create_dataframe(posts_data)
        
        # 2. Preprocess Data
        logger.info("Preprocessing text...")
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(df)
        
        # 3. Analyze Sentiment
        logger.info("Analyzing sentiment...")
        analyzer = SentimentAnalyzer()
        sentiment_df = analyzer.analyze_dataframe(processed_df)
        
        # 4. Generate Summary and Save Results
        logger.info("Generating sentiment summary...")
        sentiment_summary = analyzer.get_sentiment_summary(sentiment_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame with sentiment analysis
        output_file = f'sentiment_analysis_{timestamp}.csv'
        sentiment_df.to_csv(output_file, index=False)
        logger.info(f"Saved sentiment analysis results to {output_file}")
        
        # Save sentiment summary
        summary_file = f'sentiment_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(sentiment_summary, f, indent=4)
        logger.info(f"Saved sentiment summary to {summary_file}")
        
        # Print sample results
        print("\nSample Results:")
        sample_idx = min(len(sentiment_df) - 1, 2)
        sample = sentiment_df.iloc[sample_idx]
        
        print("\nOriginal Text:")
        print(sample['text'][:200] if pd.notnull(sample['text']) else "No text")
        
        print("\nProcessed Text:")
        print(sample['processed_content'][:200])
        
        print("\nSentiment Analysis:")
        print(f"Label: {sample['sentiment_label']}")
        print(f"Score: {sample['sentiment_score']:.3f}")
        print("\nEmotion Scores:")
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']:
            print(f"{emotion}: {sample[f'emotion_{emotion}']:.3f}")
        
        print("\nTextBlob Scores:")
        print(f"Polarity: {sample['textblob_polarity']:.3f}")
        print(f"Subjectivity: {sample['textblob_subjectivity']:.3f}")
        
        print("\nOverall Sentiment Summary:")
        print(json.dumps(sentiment_summary, indent=2))
        
        return sentiment_df, sentiment_summary
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis test: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        sentiment_df, summary = test_sentiment_analysis()
        logger.info("Sentiment analysis test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")