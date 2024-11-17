# main.py (continued)
from reddit_fetcher import RedditDataFetcher
import pandas as pd
from datetime import datetime

def main():
    # Initialize fetcher
    fetcher = RedditDataFetcher()
    
    # Fetch data for a keyword
    keyword = "iPhone 13"
    posts_data = fetcher.fetch_posts(keyword, limit=5)
    
    # Convert to DataFrame
    df = fetcher.create_dataframe(posts_data)
    
    # Basic data info
    print(f"Total records: {len(df)}")
    print("\nSample data:")
    print(df.head())
    
    # Save to CSV for later use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'reddit_data_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main()