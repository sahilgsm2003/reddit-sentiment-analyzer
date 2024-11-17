import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    def __init__(self, model_name="roberta-base", output_dir="emotion_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = 128
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Load GoEmotions dataset
        self.dataset = load_dataset("go_emotions", "simplified")
        
        # Setup model
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model with the correct number of labels"""
        # Get number of labels from dataset
        self.num_labels = len(self.dataset['train'].features['labels'].feature.names)
        
        # Initialize model
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        self.model.to(self.device)
        
    def preprocess_data(self, examples):
        """Preprocess the data"""
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        # Convert labels to multi-hot encoding
        labels = [
            [1.0 if i in labels else 0.0 for i in range(self.num_labels)]
            for labels in examples["labels"]
        ]
        
        tokenized["labels"] = labels
        return tokenized
    
    def compute_metrics(self, pred):
        """Compute metrics for evaluation"""
        predictions = (pred.predictions > 0.5).astype(int)
        labels = pred.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.flatten(),
            predictions.flatten(),
            average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, batch_size=16, num_epochs=3):
        """Train the model"""
        try:
            logger.info("Starting model training...")
            
            # Preprocess datasets
            processed_datasets = self.dataset.map(
                self.preprocess_data,
                batched=True,
                remove_columns=self.dataset["train"].column_names
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_dir=f"{self.output_dir}/logs",
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["validation"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                data_collator=DataCollatorWithPadding(self.tokenizer)
            )
            
            # Train the model
            logger.info("Training model...")
            train_result = trainer.train()
            
            # Save the final model
            logger.info(f"Saving model to {self.output_dir}")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Log training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            # Evaluate the model
            logger.info("Evaluating model...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            
            return metrics, eval_metrics
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise
    
    def predict(self, texts):
        """Make predictions on new texts"""
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.sigmoid(outputs.logits)
            
            # Convert predictions to labels
            predictions = predictions.cpu().numpy()
            predicted_labels = (predictions > 0.5).astype(int)
            
            # Get emotion labels
            emotion_labels = self.dataset['train'].features['labels'].feature.names
            
            # Format results
            results = []
            for pred, scores in zip(predicted_labels, predictions):
                emotions = {
                    emotion_labels[i]: float(scores[i])
                    for i in range(len(emotion_labels))
                    if pred[i] == 1
                }
                results.append(emotions)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise