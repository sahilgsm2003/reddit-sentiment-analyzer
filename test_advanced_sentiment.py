# test_advanced_sentiment.py

import logging
from advanced_sentiment import AdvancedSentimentAnalyzer
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_advanced_sentiment():
    try:
        # Initialize analyzer
        analyzer = AdvancedSentimentAnalyzer()
        
        # Fine-tune model (comment out if loading pre-trained model)
        analyzer.fine_tune_emotion_model()
        
        # Test data
        test_texts = [
            "I absolutely love this phone! The camera is amazing and battery life is fantastic!",
            "Yeah, right... Another 'amazing' phone with terrible battery life. So impressed... /s",
            "The camera is good but battery life is disappointing. Mixed feelings about this one.",
            "Don't waste your money on this. Worst phone I've ever had.",
            "Actually, I've had this phone for 6 months and it's been great. No issues at all."
        ]
        
        # Analyze thread
        logger.info("Analyzing thread...")
        results = analyzer.analyze_thread(test_texts)
        
        # Save results
        with open('sentiment_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print sample results
        print("\nThread Analysis Results:")
        print("\nDominant emotions per text:")
        for i, analysis in enumerate(results['individual_analyses']):
            print(f"\nText {i+1}:")
            print(f"Dominant emotion: {analysis['dominant_emotion']}")
            print(f"Sarcasm score: {analysis['sarcasm']['sarcasm_score']:.2f}")
        
        print("\nContradictions found:", len(results['contradictions']))
        print("Sarcasm detected in thread:", results['sarcasm_detected'])
        
        return results
        
    except Exception as e:
        logger.error(f"Error in advanced sentiment test: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = test_advanced_sentiment()
        logger.info("Advanced sentiment analysis test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")