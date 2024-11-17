import praw
import pandas as pd
from datetime import datetime
import config  # We'll create this file for API credentials

class RedditDataFetcher:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=config.CLIENT_ID,
            client_secret=config.CLIENT_SECRET,
            user_agent=config.USER_AGENT
        )

    def fetch_posts(self, keyword, limit=10):
        """
        Fetch Reddit posts and their comments based on keyword
        """
        posts_data = []
        
        # Search for posts
        for submission in self.reddit.subreddit('all').search(keyword, limit=limit):
            # Get post data
            post_data = {
                'post_id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'num_comments': submission.num_comments,
                'url': submission.url,
                'comments': []
            }
            
            # Get comments
            submission.comments.replace_more(limit=0)  # Remove MoreComments objects
            for comment in submission.comments.list():
                comment_data = {
                    'comment_id': comment.id,
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc)
                }
                post_data['comments'].append(comment_data)
            
            posts_data.append(post_data)
        
        return posts_data

    def create_dataframe(self, posts_data):
        """
        Convert posts data to pandas DataFrame
        """
        # Flatten posts and comments into a single DataFrame
        rows = []
        
        for post in posts_data:
            # Add post
            rows.append({
                'id': post['post_id'],
                'type': 'post',
                'title': post['title'],
                'text': post['text'],
                'score': post['score'],
                'created_utc': post['created_utc'],
                'url': post['url']
            })
            
            # Add comments
            for comment in post['comments']:
                rows.append({
                    'id': comment['comment_id'],
                    'type': 'comment',
                    'title': None,
                    'text': comment['text'],
                    'score': comment['score'],
                    'created_utc': comment['created_utc'],
                    'url': post['url']
                })
        
        return pd.DataFrame(rows)