import pandas as pd
from reddit_fetcher import RedditDataFetcher
from preprocessor import TextPreprocessor
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    try:
        # 1. Test Reddit Data Fetching
        logger.info("Testing Reddit data fetching...")
        fetcher = RedditDataFetcher()
        keyword = "water bottle"
        posts_data = fetcher.fetch_posts(keyword, limit=3)
        
        if not posts_data:
            logger.error("No data fetched from Reddit")
            return
        
        logger.info(f"Successfully fetched {len(posts_data)} posts")
        
        # Convert to DataFrame
        df = fetcher.create_dataframe(posts_data)
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = f'raw_data_{timestamp}.csv'
        df.to_csv(raw_filename, index=False)
        logger.info(f"Saved raw data to {raw_filename}")
        
        # 2. Test Preprocessing
        logger.info("Testing text preprocessing...")
        preprocessor = TextPreprocessor()
        
        # Process the DataFrame
        processed_df = preprocessor.preprocess_dataframe(df)
        
        # Save processed data
        processed_filename = f'processed_data_{timestamp}.csv'
        processed_df.to_csv(processed_filename, index=False)
        logger.info(f"Saved processed data to {processed_filename}")
        
        # Print sample results
        logger.info("\nSample Results:")
        sample_idx = min(len(processed_df) - 1, 2)  # Get a valid index
        sample_row = processed_df.iloc[sample_idx]
        
        print("\nOriginal Text:")
        print(sample_row['text'][:500] if pd.notnull(sample_row['text']) else "No text")
        print("\nProcessed Text:")
        print(sample_row['processed_content'][:500])
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        processed_df = test_pipeline()
        logger.info("Pipeline test completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")