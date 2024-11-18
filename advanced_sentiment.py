# advanced_sentiment.py

import torch
import pandas as pd
import numpy as np
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments, 
    Trainer,
    RobertaConfig
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import datasets
import logging
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class AdvancedSentimentAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.emotion_model = None
        self.sarcasm_model = None
        self.tokenizer = None
        self.emotion_labels = None
        self.contradiction_threshold = 0.7
        
        # Load emotion labels
        self.load_emotion_labels()
        
    def load_emotion_labels(self):
        """Load GoEmotions labels"""
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
        ]

    def prepare_goemotions_data(self):
        """Prepare GoEmotions dataset for fine-tuning"""
        logger.info("Loading GoEmotions dataset...")
        dataset = datasets.load_dataset('go_emotions', 'simplified')
        
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['labels']
        
        # Convert multi-hot labels to single label (take the strongest emotion)
        train_labels = [
            label.index(max(label)) for label in train_labels
        ]
        
        return train_texts, train_labels

    def fine_tune_emotion_model(self, checkpoint="roberta-base", batch_size=16, epochs=3):
        """Fine-tune RoBERTa model on GoEmotions dataset"""
        logger.info(f"Fine-tuning emotion model using {checkpoint}...")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
        config = RobertaConfig.from_pretrained(
            checkpoint,
            num_labels=len(self.emotion_labels)
        )
        self.emotion_model = RobertaForSequenceClassification.from_pretrained(
            checkpoint,
            config=config
        ).to(self.device)

        # Prepare dataset
        train_texts, train_labels = self.prepare_goemotions_data()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1
        )

        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = EmotionDataset(val_texts, val_labels, self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.emotion_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        self.emotion_model.save_pretrained('./emotion_model')
        self.tokenizer.save_pretrained('./emotion_model')
        logger.info("Fine-tuning completed and model saved")

    def load_fine_tuned_model(self, model_path='./emotion_model'):
        """Load a previously fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.emotion_model = RobertaForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)

    def detect_sarcasm(self, text: str) -> Dict[str, float]:
        """Detect sarcasm in text using contextual features"""
        # Implementation using sentiment contrast and linguistic markers
        sentiment_score = self.get_emotion_scores(text)
        
        # Sarcasm indicators
        indicators = {
            'sentiment_contrast': 0.0,
            'exaggeration': 0.0,
            'contradiction': 0.0
        }
        
        # Check for sentiment contrast
        if 'joy' in sentiment_score and 'disappointment' in sentiment_score:
            indicators['sentiment_contrast'] = abs(
                sentiment_score['joy'] - sentiment_score['disappointment']
            )
        
        # Check for exaggeration markers
        exaggeration_markers = ['so', 'totally', 'absolutely', 'obviously', 'clearly']
        indicators['exaggeration'] = sum(
            1 for marker in exaggeration_markers if marker in text.lower()
        ) / len(exaggeration_markers)
        
        # Combine indicators for final sarcasm score
        sarcasm_score = (
            indicators['sentiment_contrast'] * 0.5 +
            indicators['exaggeration'] * 0.3 +
            indicators['contradiction'] * 0.2
        )
        
        return {
            'sarcasm_score': float(sarcasm_score),
            'indicators': indicators
        }

    def detect_contradictions(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect contradicting opinions in a list of texts"""
        contradictions = []
        
        # Get emotion scores for all texts
        emotion_scores = [self.get_emotion_scores(text) for text in texts]
        
        # Compare pairs of texts
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Calculate emotion vector similarity
                similarity = self._calculate_emotion_similarity(
                    emotion_scores[i],
                    emotion_scores[j]
                )
                
                # If emotions are very different, might indicate contradiction
                if similarity < self.contradiction_threshold:
                    contradictions.append({
                        'text1': texts[i],
                        'text2': texts[j],
                        'similarity': float(similarity),
                        'emotions1': emotion_scores[i],
                        'emotions2': emotion_scores[j]
                    })
        
        return contradictions

    def _calculate_emotion_similarity(
        self, 
        emotions1: Dict[str, float], 
        emotions2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two emotion vectors"""
        # Convert emotion dictionaries to vectors
        vec1 = np.array([emotions1.get(label, 0.0) for label in self.emotion_labels])
        vec2 = np.array([emotions2.get(label, 0.0) for label in self.emotion_labels])
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        ) if np.linalg.norm(vec1) * np.linalg.norm(vec2) != 0 else 0.0
        
        return float(similarity)

    def get_emotion_scores(self, text: str) -> Dict[str, float]:
        """Get emotion scores for a text"""
        if not isinstance(text, str) or not text.strip():
            return {label: 0.0 for label in self.emotion_labels}

        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to dictionary
        emotion_scores = {
            self.emotion_labels[i]: float(probs[0][i])
            for i in range(len(self.emotion_labels))
        }

        return emotion_scores

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Complete analysis of a single text"""
        return {
            'emotions': self.get_emotion_scores(text),
            'sarcasm': self.detect_sarcasm(text),
            'dominant_emotion': max(
                self.get_emotion_scores(text).items(),
                key=lambda x: x[1]
            )[0]
        }

    def analyze_thread(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze a complete thread of texts"""
        # Individual analysis for each text
        individual_analyses = [self.analyze_text(text) for text in texts]
        
        # Thread-level analysis
        contradictions = self.detect_contradictions(texts)
        
        # Aggregate emotions across thread
        thread_emotions = {}
        for emotion in self.emotion_labels:
            scores = [
                analysis['emotions'][emotion]
                for analysis in individual_analyses
            ]
            thread_emotions[emotion] = {
                'mean': float(np.mean(scores)),
                'max': float(np.max(scores)),
                'variance': float(np.var(scores))
            }
        
        return {
            'individual_analyses': individual_analyses,
            'thread_emotions': thread_emotions,
            'contradictions': contradictions,
            'sarcasm_detected': any(
                analysis['sarcasm']['sarcasm_score'] > 0.5
                for analysis in individual_analyses
            )
        }