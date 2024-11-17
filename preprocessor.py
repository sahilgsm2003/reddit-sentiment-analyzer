import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Load spaCy model for better tokenization
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text.strip()
    
    def remove_emoji(self, text):
        """Remove emojis from text"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Process with spaCy
        doc = self.nlp(text)
        
        # Get tokens that aren't stop words or punctuation
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct]
        
        return tokens
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        text = self.remove_emoji(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(text)
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text

    # preprocessor.py (continued)
    def preprocess_dataframe(self, df):
        """Preprocess all text in the DataFrame"""
        # Create copy of DataFrame
        df_processed = df.copy()
        
        # Preprocess title and text columns
        if 'title' in df_processed.columns:
            df_processed['processed_title'] = df_processed['title'].apply(
                lambda x: self.preprocess_text(x) if pd.notnull(x) else '')
            
        if 'text' in df_processed.columns:
            df_processed['processed_text'] = df_processed['text'].apply(
                lambda x: self.preprocess_text(x) if pd.notnull(x) else '')
            
        # Combine processed title and text for posts
        df_processed['processed_content'] = df_processed.apply(
            lambda row: ' '.join(filter(None, [
                row['processed_title'] if 'processed_title' in row else '',
                row['processed_text'] if 'processed_text' in row else ''
            ])), axis=1)
        
        return df_processed