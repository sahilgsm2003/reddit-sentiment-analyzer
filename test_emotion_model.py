from emotion_model_trainer import EmotionModelTrainer
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_emotion_model():
    try:
        # Initialize trainer
        logger.info("Initializing emotion model trainer...")
        trainer = EmotionModelTrainer(
            output_dir=f"emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Train model
        logger.info("Starting model training...")
        train_metrics, eval_metrics = trainer.train(
            batch_size=16,
            num_epochs=3
        )
        
        # Print metrics
        logger.info("\nTraining Metrics:")
        print(train_metrics)
        logger.info("\nEvaluation Metrics:")
        print(eval_metrics)
        
        # Test predictions
        test_texts = [
            "I'm so happy about this amazing achievement!",
            "This makes me really angry and frustrated.",
            "I'm feeling quite anxious about the upcoming presentation."
        ]
        
        logger.info("\nTesting predictions...")
        predictions = trainer.predict(test_texts)
        
        # Print results
        for text, emotions in zip(test_texts, predictions):
            print(f"\nText: {text}")
            print("Predicted emotions:")
            for emotion, score in emotions.items():
                print(f"  {emotion}: {score:.3f}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in emotion model testing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        trainer = test_emotion_model()
        logger.info("Emotion model testing completed successfully!")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")