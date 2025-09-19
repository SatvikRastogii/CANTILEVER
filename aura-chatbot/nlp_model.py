import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import clean_text
import os
import random

class Chatbot:
    def __init__(self, model_path='saved_model'):
        """Initialize the chatbot with trained models."""
        self.model_path = model_path
        self.encoder_model = None
        self.decoder_model = None
        self.tokenizer = None
        self.params = None
        self.reverse_word_index = None
        
        # Load models and parameters
        self._load_models()
        self._load_tokenizer()
        self._load_params()
        
        # Fallback responses for when model fails
        self.fallback_responses = [
            "I understand you're going through a difficult time. You're not alone in this.",
            "Thank you for sharing that with me. It takes courage to open up about your feelings.",
            "I hear you, and I want you to know that your feelings are valid.",
            "It's okay to feel overwhelmed sometimes. Remember to be kind to yourself.",
            "You're doing the best you can, and that's enough. Take things one step at a time.",
            "I'm here to listen and support you. You don't have to face this alone.",
            "Your feelings are completely understandable. Many people experience similar challenges.",
            "It's important to remember that seeking help shows strength, not weakness.",
            "You're not alone in this journey. Many people have felt this way before.",
            "It's okay to not have all the answers right now. Healing takes time."
        ]
    
    def _load_models(self):
        """Load the trained encoder and decoder models."""
        try:
            encoder_path = os.path.join(self.model_path, 'encoder_model.h5')
            decoder_path = os.path.join(self.model_path, 'decoder_model.h5')
            
            if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                self.encoder_model = tf.keras.models.load_model(encoder_path)
                self.decoder_model = tf.keras.models.load_model(decoder_path)
                print("Models loaded successfully!")
            else:
                print("Model files not found. Using fallback responses.")
                self.encoder_model = None
                self.decoder_model = None
        except Exception as e:
            print(f"Error loading models: {e}")
            self.encoder_model = None
            self.decoder_model = None
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            tokenizer_path = os.path.join(self.model_path, 'tokenizer.json')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                    self.tokenizer = tokenizer_from_json(tokenizer_data)
                print("Tokenizer loaded successfully!")
            else:
                print("Tokenizer file not found.")
                self.tokenizer = None
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None
    
    def _load_params(self):
        """Load model parameters."""
        try:
            params_path = os.path.join(self.model_path, 'params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r', encoding='utf-8') as f:
                    self.params = json.load(f)
                
                # Create reverse word index
                if 'word_index' in self.params:
                    self.reverse_word_index = {v: k for k, v in self.params['word_index'].items()}
                
                print("Parameters loaded successfully!")
            else:
                print("Parameters file not found.")
                self.params = {}
        except Exception as e:
            print(f"Error loading parameters: {e}")
            self.params = {}
    
    def _get_fallback_response(self):
        """Get a random fallback response."""
        return random.choice(self.fallback_responses)
    
    def _preprocess_input(self, input_text):
        """Preprocess input text for the model."""
        # Clean the input text
        clean_input = clean_text(input_text)
        
        # Handle empty or very short inputs
        if not clean_input or len(clean_input.strip()) < 2:
            return None
        
        return clean_input
    
    def _encode_input(self, input_text):
        """Encode input text to sequences."""
        if not self.tokenizer:
            return None
        
        try:
            # Convert text to sequence
            input_seq = self.tokenizer.texts_to_sequences([input_text])
            
            # Pad sequence
            max_length = self.params.get('max_sequence_length', 150)
            input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
            
            return input_seq
        except Exception as e:
            print(f"Error encoding input: {e}")
            return None
    
    def _decode_sequence(self, input_seq):
        """Decode sequence to text using the trained model."""
        if not self.encoder_model or not self.decoder_model or not self.tokenizer:
            return None
        
        try:
            # Encode the input to get the state vectors
            states_value = self.encoder_model.predict(input_seq, verbose=0)
            
            # Start decoding with the '<start>' token
            target_seq = np.zeros((1, 1))
            if '<start>' in self.tokenizer.word_index:
                target_seq[0, 0] = self.tokenizer.word_index['<start>']
            else:
                # If no start token, use the first word in vocabulary
                target_seq[0, 0] = 1
            
            stop_condition = False
            decoded_sentence = ''
            max_length = self.params.get('max_sequence_length', 150)
            step = 0
            
            while not stop_condition and step < max_length:
                # Get predictions
                output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value, verbose=0)
                
                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                
                # Get the word
                if sampled_token_index in self.reverse_word_index:
                    sampled_word = self.reverse_word_index[sampled_token_index]
                else:
                    sampled_word = '<unk>'
                
                # Check for stop conditions
                if (sampled_word == '<end>' or 
                    sampled_word == '<unk>' or 
                    len(decoded_sentence.split()) > max_length):
                    stop_condition = True
                else:
                    decoded_sentence += ' ' + sampled_word
                
                # Update the target sequence
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index
                
                # Update states
                states_value = [h, c]
                step += 1
            
            return decoded_sentence.strip()
        
        except Exception as e:
            print(f"Error decoding sequence: {e}")
            return None
    
    def _postprocess_response(self, response):
        """Postprocess the generated response."""
        if not response:
            return None
        
        # Clean up the response
        response = response.strip()
        
        # Remove any remaining special tokens
        response = response.replace('<start>', '').replace('<end>', '').strip()
        
        # Ensure response is not empty
        if not response or len(response) < 3:
            return None
        
        # Capitalize first letter
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
        
        # Ensure it ends with proper punctuation
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def generate_response(self, input_text):
        """Generate a response to the input text."""
        try:
            # Preprocess input
            clean_input = self._preprocess_input(input_text)
            if not clean_input:
                return self._get_fallback_response()
            
            # Check if models are available
            if not self.encoder_model or not self.decoder_model or not self.tokenizer:
                return self._get_fallback_response()
            
            # Encode input
            input_seq = self._encode_input(clean_input)
            if input_seq is None:
                return self._get_fallback_response()
            
            # Decode sequence
            raw_response = self._decode_sequence(input_seq)
            if not raw_response:
                return self._get_fallback_response()
            
            # Postprocess response
            response = self._postprocess_response(raw_response)
            if not response:
                return self._get_fallback_response()
            
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._get_fallback_response()
    
    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            'models_loaded': self.encoder_model is not None and self.decoder_model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'params_loaded': bool(self.params),
            'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else 0,
            'max_sequence_length': self.params.get('max_sequence_length', 'Unknown'),
            'latent_dim': self.params.get('latent_dim', 'Unknown'),
            'embedding_dim': self.params.get('embedding_dim', 'Unknown')
        }
        return info

# Singleton instance to avoid reloading the model on every request
aura_bot = None

def get_aura_bot():
    """Get the singleton chatbot instance."""
    global aura_bot
    if aura_bot is None:
        aura_bot = Chatbot()
    return aura_bot

def test_chatbot():
    """Test the chatbot with sample inputs."""
    bot = get_aura_bot()
    
    test_inputs = [
        "I'm feeling really anxious today",
        "I've been having trouble sleeping",
        "I feel lonely and isolated",
        "I'm stressed about work",
        "I don't know what to do with my life"
    ]
    
    print("Testing Aura chatbot...")
    print("=" * 50)
    
    for test_input in test_inputs:
        response = bot.generate_response(test_input)
        print(f"Input: {test_input}")
        print(f"Response: {response}")
        print("-" * 30)
    
    # Print model info
    print("\nModel Information:")
    info = bot.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_chatbot()
