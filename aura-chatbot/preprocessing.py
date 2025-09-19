import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import os

# Ensure you've run: nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs, non-alphanumeric characters (except basic punctuation)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\'.?!,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_pii(text):
    """Remove personally identifiable information."""
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE>', text)
    # Remove common names (basic list)
    common_names = ['john', 'jane', 'mike', 'sarah', 'david', 'lisa', 'chris', 'emily', 'alex', 'sam']
    for name in common_names:
        text = re.sub(r'\b' + name + r'\b', '<NAME>', text, flags=re.IGNORECASE)
    return text

def is_quality_text(text, min_words=3, max_words=150):
    """Check if text meets quality criteria."""
    if not text or len(text.strip()) == 0:
        return False
    
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Check for deleted/removed content
    if any(phrase in text.lower() for phrase in ['[deleted]', '[removed]', 'deleted', 'removed']):
        return False
    
    # Check for excessive special characters
    if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.3:
        return False
    
    return True

def filter_supportive_response(response):
    """Filter out non-supportive responses."""
    # Keywords that indicate non-supportive content
    negative_keywords = [
        'stupid', 'idiot', 'dumb', 'hate', 'kill yourself', 'suicide',
        'worthless', 'pathetic', 'loser', 'shut up', 'nobody cares'
    ]
    
    response_lower = response.lower()
    for keyword in negative_keywords:
        if keyword in response_lower:
            return False
    
    # Check for excessive profanity
    profanity_count = len(re.findall(r'\b(fuck|shit|damn|bitch|asshole)\b', response_lower))
    if profanity_count > 2:
        return False
    
    return True

def load_and_preprocess_data(filepath):
    """Load data from CSV and apply cleaning."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Creating empty dataset.")
        return [], []
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], []
    
    # Check for required columns
    if 'prompt' not in df.columns or 'response' not in df.columns:
        print("Error: CSV must contain 'prompt' and 'response' columns")
        return [], []
    
    df.dropna(inplace=True)
    print(f"After dropping NaN: {len(df)} rows")
    
    # Clean and filter data
    cleaned_prompts = []
    cleaned_responses = []
    
    for idx, row in df.iterrows():
        prompt = str(row['prompt']).strip()
        response = str(row['response']).strip()
        
        # Clean text
        prompt = clean_text(prompt)
        response = clean_text(response)
        
        # Remove PII
        prompt = remove_pii(prompt)
        response = remove_pii(response)
        
        # Quality checks
        if (is_quality_text(prompt) and 
            is_quality_text(response) and 
            filter_supportive_response(response)):
            
            # Add start and end tokens to the target sequences
            response_with_tokens = f'<start> {response} <end>'
            cleaned_prompts.append(prompt)
            cleaned_responses.append(response_with_tokens)
    
    print(f"After cleaning and filtering: {len(cleaned_prompts)} pairs")
    return cleaned_prompts, cleaned_responses

def create_sample_dataset():
    """Create a sample dataset for testing purposes."""
    sample_data = [
        ("I'm feeling really anxious today", "I understand that anxiety can be overwhelming. Take some deep breaths and remember that this feeling will pass."),
        ("I've been having trouble sleeping", "Sleep issues can be really frustrating. Have you tried establishing a relaxing bedtime routine?"),
        ("I feel lonely and isolated", "You're not alone in feeling this way. Many people experience loneliness, and it's okay to reach out for support."),
        ("I'm stressed about work", "Work stress is very common. Remember to take breaks and be kind to yourself during difficult times."),
        ("I don't know what to do with my life", "It's completely normal to feel uncertain about the future. Take things one step at a time."),
        ("I'm having relationship problems", "Relationships can be challenging. Communication and understanding are key to working through difficulties."),
        ("I feel like I'm not good enough", "You are valuable and worthy just as you are. Everyone has strengths and areas for growth."),
        ("I'm worried about my health", "Health concerns can be scary. It's important to talk to healthcare professionals about your worries."),
        ("I feel overwhelmed by everything", "Feeling overwhelmed is understandable. Try breaking tasks into smaller, manageable steps."),
        ("I'm struggling with motivation", "Lack of motivation is common, especially during difficult times. Be patient with yourself and celebrate small wins.")
    ]
    
    # Expand the dataset by creating variations
    expanded_data = []
    for prompt, response in sample_data:
        expanded_data.append((prompt, response))
        # Add some variations
        expanded_data.append((f"How do I deal with {prompt.lower()}", response))
        expanded_data.append((f"I need help with {prompt.lower()}", response))
    
    return expanded_data

def save_sample_dataset(filepath):
    """Save a sample dataset to CSV for testing."""
    sample_data = create_sample_dataset()
    df = pd.DataFrame(sample_data, columns=['prompt', 'response'])
    df.to_csv(filepath, index=False)
    print(f"Sample dataset saved to {filepath} with {len(df)} rows")
    return len(df)
