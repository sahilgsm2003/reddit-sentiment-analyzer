from sentiment_analyzer import SentimentAnalyzer
from emotion_model_trainer import EmotionModelTrainer
import logging
import torch
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, emotion_model_path=None):
        super().__init__()
        
        # Load fine-tuned emotion model if path is provided
        self.emotion_model = None
        if emotion_model_path and os.path.exists(emotion_model_path):
            logger.info("Loading fine-tuned emotion model...")
            try:
                self.emotion_model = EmotionModelTrainer(output_dir=emotion_model_path)
                logger.info("Fine-tuned emotion model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading fine-tuned model: {e}")
                logger.info("Falling back to default emotion classifier")
    
    def get_emotion_scores(self, text):
        """Get emotion scores using fine-tuned model if available"""
        try:
            if self.emotion_model:
                # Use fine-tuned model
                predictions = self.emotion_model.predict([text])[0]
                return predictions
            else:
                # Fall back to original emotion classifier
                return super().get_emotion_scores(text)
                
        except Exception as e:
            logger.error(f"Error in enhanced emotion analysis: {e}")
            return super().get_emotion_scores(text)
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with fine-tuned emotions"""
        if not isinstance(text, str) or not text.strip():
            return {
                'roberta_sentiment': {'label': 'NEUTRAL', 'score': 0.5},
                'emotions': {},
                'textblob': {'polarity': 0.0, 'subjectivity': 0.0}
            }

        # Get base sentiment analysis
        sentiment_scores = self.get_sentiment_scores(text)
        emotion_scores = self.get_emotion_scores(text)
        textblob_scores = self.get_textblob_sentiment(text)

        return {
            'roberta_sentiment': sentiment_scores,
            'emotions': emotion_scores,
            'textblob': textblob_scores
        }

def test_enhanced_analyzer():
    try:
        # Initialize enhanced analyzer
        logger.info("Initializing enhanced sentiment analyzer...")
        analyzer = EnhancedSentimentAnalyzer(
            emotion_model_path="emotion_model_latest"  # Update with your model path
        )
        
        # Test texts
        test_texts = [
            "I absolutely love this product! It's amazing!",
            "This is the worst experience ever. I'm so disappointed.",
            "The results are somewhat mixed, but generally okay.",
            "I'm feeling anxious about the upcoming changes.",
            "This made me laugh so hard! 😂"
        ]
        
        # Analyze texts
        logger.info("Testing enhanced sentiment analysis...")
        for text in test_texts:
            results = analyzer.analyze_sentiment(text)
            
            print(f"\nText: {text}")
            print("Sentiment:", results['roberta_sentiment']['label'])
            print("Confidence:", f"{results['roberta_sentiment']['score']:.3f}")
            print("Emotions:")
            for emotion, score in results['emotions'].items():
                print(f"  {emotion}: {score:.3f}")
            print("TextBlob Metrics:")
            print(f"  Polarity: {results['textblob']['polarity']:.3f}")
            print(f"  Subjectivity: {results['textblob']['subjectivity']:.3f}")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Error in enhanced analyzer testing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        analyzer = test_enhanced_analyzer()
        logger.info("Enhanced sentiment analyzer testing completed successfully!")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")