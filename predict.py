import torch
import pandas as pd
import numpy as np
import pickle
from model import MultiTaskTransformer, ModelConfig

class CTRPredictor:
    def __init__(self, model_path='best_ctr_model.pth', encoders_path='encoders.pkl'):
        self.config = ModelConfig()
        self.model = MultiTaskTransformer(self.config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction
        data: dict with keys like user_id, item_id, category, device, etc.
        """
        processed = {}
        
        # Encode categorical features
        categorical_cols = ['user_id', 'item_id', 'category', 'device', 'time_of_day']
        for col in categorical_cols:
            if col in data:
                encoded_col = f"{col}_encoded"
                try:
                    processed[encoded_col] = torch.tensor([
                        self.encoders[col].transform([data[col]])[0]
                    ], dtype=torch.long)
                except ValueError:
                    # Handle unseen categories
                    processed[encoded_col] = torch.tensor([0], dtype=torch.long)
        
        # Process dense features
        dense_features = []
        for feat in self.config.dense_feats:
            value = data.get(feat, 0.0)
            dense_features.append(float(value))
        
        processed['dense_features'] = torch.tensor([dense_features], dtype=torch.float32)
        
        return processed
    
    def predict(self, data):
        """
        Make CTR prediction for single instance
        Returns: dict with p_ctr, p_cvr, p_ctcvr
        """
        processed_input = self.preprocess_input(data)
        
        with torch.no_grad():
            predictions = self.model(processed_input)
        
        return {
            'p_ctr': predictions['p_ctr'].item(),
            'p_cvr': predictions['p_cvr'].item(),
            'p_ctcvr': predictions['p_ctcvr'].item()
        }
    
    def predict_batch(self, data_list):
        """
        Make CTR predictions for batch of instances
        """
        results = []
        for data in data_list:
            results.append(self.predict(data))
        return results

def demo_predictions():
    """Demo function showing how to use the predictor"""
    
    # Initialize predictor
    predictor = CTRPredictor()
    
    # Example prediction inputs
    examples = [
        {
            'user_id': 'user_123',
            'item_id': 'item_456',
            'category': 'electronics',
            'device': 'mobile',
            'time_of_day': 'evening',
            'user_age': 25,
            'price': 299.99,
            'item_rating': 4.5,
            'user_historical_ctr': 0.15,
            'position': 2,
            'page_views': 3,
            'session_duration': 45.2
        },
        {
            'user_id': 'user_789',
            'item_id': 'item_101',
            'category': 'fashion',
            'device': 'desktop',
            'time_of_day': 'afternoon',
            'user_age': 35,
            'price': 89.99,
            'item_rating': 4.2,
            'user_historical_ctr': 0.08,
            'position': 5,
            'page_views': 7,
            'session_duration': 120.5
        }
    ]
    
    print("CTR Prediction Demo")
    print("="*50)
    
    for i, example in enumerate(examples):
        pred = predictor.predict(example)
        
        print(f"\nExample {i+1}:")
        print(f"User: {example['user_id']}, Item: {example['item_id']}")
        print(f"Category: {example['category']}, Device: {example['device']}")
        print(f"Price: ${example['price']}, Position: {example['position']}")
        print(f"Predictions:")
        print(f"  CTR: {pred['p_ctr']:.4f} ({pred['p_ctr']*100:.2f}%)")
        print(f"  CVR: {pred['p_cvr']:.4f} ({pred['p_cvr']*100:.2f}%)")
        print(f"  CTCVR: {pred['p_ctcvr']:.4f} ({pred['p_ctcvr']*100:.2f}%)")

if __name__ == "__main__":
    demo_predictions()