import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
import pickle
import os
import tensorflow as tf

# Define paths
OUTPUT_MODEL_PATH = "phishing_detection_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "target_encoder.pkl"

# Function to create dummy data similar to your training data
def create_dummy_data():
    """Create dummy data with structure similar to the training data"""
    data = {
        'TLD': ['com', 'org', 'net', 'edu', 'gov', 'io', 'co'],
        'NumberOfSpecialCharacters': [5, 10, 2, 3, 7, 1, 4],
        'NumberOfSubdomains': [2, 1, 3, 0, 1, 2, 1],
        'HTTPS': [1, 0, 1, 1, 1, 0, 1],
        'VectorSpaceModelScore': [0.75, 0.45, 0.85, 0.65, 0.92, 0.32, 0.68],
        'AlexaRank': [1000, 500000, 10000, 5000, 2000, 300000, 8000],
        'NumberOfRedirects': [0, 2, 1, 0, 0, 3, 1],
        'DNSRecord': [1, 0, 1, 1, 1, 0, 1],
        'GoogleIndexing': [1, 0, 1, 1, 1, 0, 1],
        'PageRank': [7, 3, 6, 8, 9, 2, 5],
        'IFramePresence': [0, 1, 0, 0, 0, 1, 0],
        'LinksPointingToPage': [20, 5, 15, 25, 30, 3, 12],
        'GoogleAnalytics': [1, 0, 1, 1, 1, 0, 1],
        'AgeOfDomain': [730, 30, 365, 1095, 2190, 60, 183],
        'StatisticalReports': [0, 1, 0, 0, 0, 1, 0],
        'label': [0, 1, 0, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

def build_model(input_shape):
    """Recreate the model architecture from the notebook"""
    model = Sequential()
    
    model.add(Input(shape=(input_shape,)))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(0.7))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.004)))
    model.add(Dropout(0.7))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.004)))
    model.add(Dropout(0.7))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    try:
        # Create dummy data
        print("Creating dummy data...")
        dummy_data = create_dummy_data()
        
        # Prepare X and y
        X = dummy_data.drop('label', axis=1)
        y = dummy_data['label']
        
        # Initialize and fit target encoder
        print("Fitting target encoder...")
        categorical_cols = ['TLD']
        target_encoder = TargetEncoder(cols=categorical_cols)
        X_encoded = target_encoder.fit_transform(X, y)
        
        # Initialize and fit scaler
        print("Fitting scaler...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Save the preprocessors
        print("Saving preprocessors...")
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(target_encoder, f)
        
        # Build and save the model
        print("Building model...")
        model = build_model(X_scaled.shape[1])
        
        print("Saving model...")
        model.save(OUTPUT_MODEL_PATH)
        
        print("Conversion completed successfully!")
        print(f"Model saved to: {OUTPUT_MODEL_PATH}")
        print(f"Scaler saved to: {SCALER_PATH}")
        print(f"Target encoder saved to: {ENCODER_PATH}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main()