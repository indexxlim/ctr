import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import random

def generate_ctr_data(n_samples=10000, n_users=5000, n_items=1000):
    """Generate synthetic CTR prediction dataset"""
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    # Generate categorical features
    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]
    categories = ['electronics', 'fashion', 'books', 'sports', 'home']
    devices = ['mobile', 'desktop', 'tablet']
    times = ['morning', 'afternoon', 'evening', 'night']
    
    for i in range(n_samples):
        user_id = random.choice(users)
        item_id = random.choice(items)
        category = random.choice(categories)
        device = random.choice(devices)
        time_of_day = random.choice(times)
        
        # Generate dense features
        user_age = np.random.randint(18, 65)
        price = np.random.uniform(10, 1000)
        item_rating = np.random.uniform(3.0, 5.0)
        user_historical_ctr = np.random.uniform(0.01, 0.3)
        
        # Generate position and context features
        position = np.random.randint(1, 11)  # ad position 1-10
        page_views = np.random.poisson(5)
        session_duration = np.random.exponential(30)  # minutes
        
        # Create realistic CTR based on features
        base_ctr = 0.05
        
        # Feature effects
        if device == 'mobile':
            base_ctr *= 1.2
        elif device == 'desktop':
            base_ctr *= 0.9
            
        if time_of_day in ['afternoon', 'evening']:
            base_ctr *= 1.1
            
        if position <= 3:
            base_ctr *= 1.5
        elif position <= 6:
            base_ctr *= 1.2
            
        if price < 50:
            base_ctr *= 1.3
        elif price > 500:
            base_ctr *= 0.8
            
        if item_rating > 4.5:
            base_ctr *= 1.2
            
        # Add user effect
        base_ctr *= (0.5 + user_historical_ctr * 2)
        
        # Generate click label
        click_prob = min(base_ctr, 0.5)  # cap at 50%
        clicked = np.random.random() < click_prob
        
        # Generate conversion (only if clicked)
        converted = False
        if clicked:
            conv_prob = 0.1 if category == 'electronics' else 0.05
            if price < 100:
                conv_prob *= 1.5
            converted = np.random.random() < conv_prob
        
        data.append({
            'user_id': user_id,
            'item_id': item_id,
            'category': category,
            'device': device,
            'time_of_day': time_of_day,
            'user_age': user_age,
            'price': price,
            'item_rating': item_rating,
            'user_historical_ctr': user_historical_ctr,
            'position': position,
            'page_views': page_views,
            'session_duration': session_duration,
            'clicked': int(clicked),
            'converted': int(converted)
        })
    
    df = pd.DataFrame(data)
    
    # Create label encoders for categorical features
    encoders = {}
    categorical_cols = ['user_id', 'item_id', 'category', 'device', 'time_of_day']
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col])
    
    return df, encoders

if __name__ == "__main__":
    # Generate data
    df, encoders = generate_ctr_data(n_samples=50000)
    
    # Save data
    df.to_csv('ctr_data.csv', index=False)
    
    # Save encoders
    import pickle
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"Generated {len(df)} samples")
    print(f"CTR: {df['clicked'].mean():.4f}")
    print(f"CVR: {df['converted'].sum() / df['clicked'].sum():.4f}")
    print("\nData shape:", df.shape)
    print("\nFeature columns:")
    print(df.columns.tolist())