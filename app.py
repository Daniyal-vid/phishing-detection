# import os
# import pandas as pd
# import numpy as np
# import pickle
# import re

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from urllib.parse import urlparse

# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from category_encoders import TargetEncoder

# app = Flask(__name__)
# CORS(app)

# # Load the saved model and preprocessors
# MODEL_PATH = "phishing_detection_model.h5"
# SCALER_PATH = "scaler.pkl"
# ENCODER_PATH = "target_encoder.pkl"

# # Load model
# model = load_model(MODEL_PATH)

# # Load preprocessors
# with open(ENCODER_PATH, 'rb') as f:
#     target_encoder = pickle.load(f)

# with open(SCALER_PATH, 'rb') as f:
#     scaler = pickle.load(f)

# def extract_features(url):
#     """
#     Extract features from a URL to match the training dataset exactly
#     """
#     # Define all features from the original dataset
#     features = {
#         'TLD': '',
#         'NumberOfSpecialCharacters': 0,
#         'NumberOfSubdomains': 0,
#         'HTTPS': 0,
#         'VectorSpaceModelScore': 0.5,
#         'AlexaRank': 500000,
#         'NumberOfRedirects': 0,
#         'DNSRecord': 1,
#         'GoogleIndexing': 1,
#         'PageRank': 5,
#         'IFramePresence': 0,
#         'LinksPointingToPage': 10,
#         'GoogleAnalytics': 1,
#         'AgeOfDomain': 365,
#         'StatisticalReports': 0
#     }
    
#     try:
#         # Parse the URL
#         parsed_url = urlparse(url)
#         domain = parsed_url.netloc
        
#         # Remove 'www.' if present
#         if domain.startswith('www.'):
#             domain = domain[4:]
        
#         # TLD extraction
#         tld_parts = domain.split('.')
#         features['TLD'] = tld_parts[-1] if len(tld_parts) > 1 else domain
        
#         # Number of special characters
#         features['NumberOfSpecialCharacters'] = len(re.findall(r'[^a-zA-Z0-9\s]', url))
        
#         # Number of subdomains
#         features['NumberOfSubdomains'] = domain.count('.')
        
#         # HTTPS check
#         features['HTTPS'] = 1 if url.startswith('https://') else 0
    
#     except Exception as e:
#         print(f"Error extracting features: {e}")
    
#     return pd.DataFrame([features])

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get URL from request
#         data = request.get_json()
#         url = data.get('url', '')
        
#         if not url:
#             return jsonify({'error': 'No URL provided'}), 400
        
#         # Extract features
#         features_df = extract_features(url)
        
#         # Apply target encoding
#         X_encoded = target_encoder.transform(features_df)
        
#         # Apply scaling
#         X_scaled = scaler.transform(X_encoded)
        
#         # Make prediction
#         prediction = model.predict(X_scaled)
#         probability = float(prediction[0][0])
#         is_phishing = bool(probability > 0.5)
        
#         # Return prediction
#         return jsonify({
#             'url': url,
#             'is_phishing': is_phishing,
#             'probability': probability,
#             'status': 'success'
#         })
        
#     except Exception as e:
#         return jsonify({
#             'error': str(e),
#             'status': 'error',
#             'details': str(e)
#         }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         'status': 'healthy', 
#         'input_shape': model.input_shape,
#         'feature_count': model.input_shape[1]
#     }), 200

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)
import os
import json
import re
from urllib.parse import urlparse
import google.generativeai as genai

class PhishingDetector:
    def __init__(self):
        # Initialize Gemini API
        api_key = "AIzaSyCJKzbbXj026q7KttGNt9pjCfWOvG_AXGk"
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def _construct_phishing_prompt(self, url):
        """
        Construct a comprehensive prompt for in-depth phishing detection
        """
        return f"""
        Perform a comprehensive security analysis of the following URL: {url}

        Detailed Evaluation Criteria:
        1. Domain Characteristics
        - Top-Level Domain (TLD) legitimacy
        - Domain age and registration history
        - Presence of IP address instead of domain name
        - Number of subdomains
        - WHOIS information transparency

        2. URL Structure Analysis
        - Unusual character patterns
        - Presence of suspicious special characters
        - URL length and complexity
        - Use of URL shortening services
        - Suspicious redirects or obfuscation techniques

        3. Security Indicators
        - HTTPS/SSL certificate status
        - Presence of padlock icon
        - Certificate authority reputation
        - DNS record authenticity

        4. Content and Behavioral Indicators
        - Presence of suspicious iframes
        - Popup window behavior
        - Right-click disabling techniques
        - Abnormal link structures
        - Suspicious form handling

        5. Web Reputation Metrics
        - Google indexing status
        - Alexa/web traffic ranking
        - External link analysis
        - Presence in statistical reporting databases
        - PageRank or domain authority

        Provide a comprehensive JSON response with:
        {{
            "is_phishing": true/false,
            "confidence_score": 0.0-1.0,
            "risk_level": "Low"/"Medium"/"High"/"Critical",
            "detailed_analysis": {{
                "domain_age": "Description of domain age",
                "ssl_status": "Certificate details",
                "suspicious_indicators": ["list", "of", "flags"],
                "recommended_actions": ["User warnings", "Further investigation"]
            }},
            "technical_reasoning": "Comprehensive explanation of detection logic"
        }}

        Ensure the response is a valid, structured JSON with deep, actionable insights.
        """

    def detect_phishing(self, url):
        """
        Detect phishing using Gemini API with comprehensive analysis
        """
        try:
            # Generate detection prompt
            prompt = self._construct_phishing_prompt(url)
            
            # Generate content
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL | re.MULTILINE)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            
            # Parse the JSON
            result = json.loads(json_match.group(0))
            
            return {
                'url': url,
                'is_phishing': result.get('is_phishing', False),
                'confidence_score': result.get('confidence_score', 0.0),
                'risk_level': result.get('risk_level', 'Unknown'),
                'detailed_analysis': result.get('detailed_analysis', {}),
                'technical_reasoning': result.get('technical_reasoning', 'No detailed reasoning'),
                'method': 'gemini'
            }
        
        except Exception as e:
            return {
                'url': url,
                'is_phishing': False,
                'confidence_score': 0.0,
                'risk_level': 'Error',
                'detailed_analysis': {},
                'technical_reasoning': f'Detection failed: {str(e)}',
                'method': 'gemini_error'
            }

# Flask application setup
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the phishing detector
phishing_detector = PhishingDetector()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Detect phishing using Gemini
        result = phishing_detector.detect_phishing(url)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'gemini_api_configured': bool(os.environ.get('GEMINI_API_KEY'))
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)