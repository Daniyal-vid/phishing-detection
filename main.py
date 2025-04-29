import os
import pandas as pd
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()  # Explicitly close the plot

def load_and_preprocess_data(file_path):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Drop unnecessary columns
        columns_to_drop = ['FILENAME', 'URL', 'Domain', 'Title', 'URLLength', 'DomainTitleMatchScore', 'URLSimilarityIndex']
        data = data.drop(columns=columns_to_drop, axis=1, errors='ignore')

        # Split dataset
        X = data.drop('label', axis=1)
        y = data['label']

        # Apply target encoding to categorical feature
        target_encoder = TargetEncoder(cols=['TLD'])
        X_encoded = target_encoder.fit_transform(X, y)

        # Apply MinMax scaling to all numerical features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Save feature names to ensure correct order during inference
        feature_names = X_encoded.columns.tolist()
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)

        return X_resampled, y_resampled, target_encoder, scaler, feature_names

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None

def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
        Dropout(0.7),
        Dense(64, activation='relu', kernel_regularizer=l2(0.004)),
        Dropout(0.7),
        Dense(32, activation='relu', kernel_regularizer=l2(0.004)),
        Dropout(0.7),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_and_save_model(X_train, X_val, y_train, y_val, target_encoder, scaler, feature_names):
    model = create_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('phishing_detection_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Save preprocessors
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    return model, history

def main():
    print("Current Working Directory:", os.getcwd())
    print("Files in current directory:", os.listdir())

    train_file_path = 'PhiUSIIL_Phishing_URL_Dataset.csv'  # Update with your actual filename
    
    # Load and preprocess data
    X_resampled, y_resampled, target_encoder, scaler, feature_names = load_and_preprocess_data(train_file_path)
    
    if X_resampled is None:
        print("Failed to load dataset. Check file path and contents.")
        return
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, 
        test_size=0.2, 
        random_state=42
    )

    model, history = train_and_save_model(X_train, X_val, y_train, y_val, target_encoder, scaler, feature_names)
    
    plot_training_history(history)

    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
