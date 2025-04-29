import pandas as pd
import requests
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Number of instances to test - adjust this number as needed
MAX_SAMPLES = 100

# Load your test dataset
test_data = pd.read_csv('phishing_test_dataset.csv')

# Print the actual column names to debug
print("Column names in CSV:", test_data.columns.tolist())

# Take a balanced sample of the data to ensure we get both positive and negative examples
if len(test_data) > MAX_SAMPLES:
    print(f"Dataset has {len(test_data)} rows. Sampling {MAX_SAMPLES} rows for testing...")
    
    # Convert column names to lowercase temporarily for sampling
    temp_data = test_data.copy()
    temp_data.columns = [col.lower() for col in temp_data.columns]
    
    # Try to get a balanced sample if there's a 'label' column
    if 'label' in temp_data.columns:
        # Get unique labels
        labels = temp_data['label'].unique()
        
        if len(labels) > 1:
            # Calculate how many samples to take from each class
            samples_per_class = MAX_SAMPLES // len(labels)
            
            # Take stratified sample
            sample_dfs = []
            for label in labels:
                label_data = test_data[temp_data['label'] == label]
                if len(label_data) > samples_per_class:
                    sample_dfs.append(label_data.sample(samples_per_class))
                else:
                    sample_dfs.append(label_data)  # Take all if we have fewer than needed
            
            test_data = pd.concat(sample_dfs)
            
            # If we still don't have enough samples (due to rounding), take more randomly
            if len(test_data) < MAX_SAMPLES:
                remaining = MAX_SAMPLES - len(test_data)
                excluded = test_data.index.tolist()
                additional = test_data[~test_data.index.isin(excluded)].sample(min(remaining, len(test_data) - len(excluded)))
                test_data = pd.concat([test_data, additional])
        else:
            # If only one class, take a random sample
            test_data = test_data.sample(MAX_SAMPLES)
    else:
        # If no label column, take a random sample
        test_data = test_data.sample(MAX_SAMPLES)
        
    print(f"Sampled dataset has {len(test_data)} rows.")

# Convert column names to lowercase if needed
test_data.columns = [col.lower() for col in test_data.columns]

# Your API endpoint
API_URL = "http://127.0.0.1:5000/predict"

results = []
actual_labels = []

# Send each URL to your API
total_urls = len(test_data)
print(f"Processing {total_urls} URLs...")

for i, (index, row) in enumerate(test_data.iterrows()):
    # Progress update
    if (i + 1) % 10 == 0 or i + 1 == total_urls:
        print(f"Processing URL {i + 1}/{total_urls} ({((i + 1)/total_urls*100):.1f}%)")
    
    url = row['url']  # Now using lowercase 'url'
    actual_label = row['label']  # Now using lowercase 'label'
    actual_labels.append(actual_label)
    
    # Add delay if specified
   
        
    # Call your API with timeout to prevent hanging
    try:
        response = requests.post(API_URL, json={"url": url}, timeout=10)
        
        # Parse the response - adapt this to your specific API response format
        response_data = response.json()
        
        # Adjust based on how your API returns predictions
        # This assumes your API returns either a dictionary with a 'prediction' key
        # or directly returns the prediction as a string/number
        if isinstance(response_data, dict):
            prediction = response_data.get("prediction", "")
            confidence = response_data.get("confidence", None)
        else:
            prediction = response_data
            confidence = None
        
        results.append({
            "url": url,
            "actual": actual_label,
            "predicted": prediction,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error processing URL {url}: {e}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Print sample of results to debug
print("\nSample of results (first 5 entries):")
print(results_df[['url', 'actual', 'predicted']].head(5))

# Ensure predictions are in the right format for metrics calculation
predictions = results_df['predicted'].values
actual_labels_array = np.array(actual_labels)

# Print the unique values and their types
print("\nUnique actual labels:", set(actual_labels))
print("Unique predicted labels:", set(predictions))

# Convert all labels to the same type (strings)
actual_labels_str = [str(label).lower() for label in actual_labels]
predictions_str = [str(pred).lower() for pred in predictions]

# Now map strings to integers consistently
label_map = {'bad': 0, 'good': 1, '0': 0, '1': 1, 'true': 1, 'false': 0}
actual_labels_int = [label_map.get(label, 0) for label in actual_labels_str]
predictions_int = [label_map.get(pred, 0) for pred in predictions_str]

# Verify the conversion
print("\nAfter conversion:")
print("Unique actual labels (int):", set(actual_labels_int))
print("Unique predicted labels (int):", set(predictions_int))

# Try to calculate metrics - wrapped in try/except to handle any remaining issues
try:
    accuracy = accuracy_score(actual_labels_int, predictions_int)
    precision = precision_score(actual_labels_int, predictions_int, average='binary')
    recall = recall_score(actual_labels_int, predictions_int, average='binary')
    f1 = f1_score(actual_labels_int, predictions_int, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(actual_labels_int, predictions_int))
except Exception as e:
    print(f"\nError calculating metrics: {e}")
    print("Continuing with analysis of individual results...")

# Add converted labels to the results DataFrame
results_df['actual_int'] = actual_labels_int
results_df['predicted_int'] = predictions_int

# Analyze misclassifications using the integer labels
misclassified = results_df[results_df['actual_int'] != results_df['predicted_int']]
print(f"\nMisclassified examples: {len(misclassified)}")
if not misclassified.empty:
    print(misclassified[['url', 'actual', 'predicted', 'actual_int', 'predicted_int']].head(10))