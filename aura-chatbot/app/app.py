import sys
import os
from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the chatbot
try:
    from nlp_model import get_aura_bot
    chatbot_available = True
except ImportError as e:
    print(f"Warning: Could not import chatbot: {e}")
    chatbot_available = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize chatbot
if chatbot_available:
    try:
        aura_bot = get_aura_bot()
        logger.info("Aura chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        aura_bot = None
        chatbot_available = False
else:
    aura_bot = None

@app.route("/")
def home():
    """Render the main chat interface."""
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    """Handle chat requests and return bot responses."""
    try:
        # Get user message from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message'].strip()
        
        # Validate message
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        if len(user_message) > 1000:  # Limit message length
            return jsonify({"error": "Message too long"}), 400
        
        # Log the interaction
        logger.info(f"User message: {user_message[:100]}...")
        
        # Generate response
        if chatbot_available and aura_bot:
            try:
                response = aura_bot.generate_response(user_message)
                logger.info(f"Bot response: {response[:100]}...")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = "I'm having trouble processing your message right now. Please try again."
        else:
            # Fallback responses when model is not available
            fallback_responses = [
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
            import random
            response = random.choice(fallback_responses)
        
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Error in get_bot_response: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "chatbot_available": chatbot_available,
            "model_loaded": aura_bot is not None
        }
        
        if aura_bot:
            model_info = aura_bot.get_model_info()
            status.update(model_info)
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/model_info")
def model_info():
    """Get information about the loaded model."""
    if not chatbot_available or not aura_bot:
        return jsonify({"error": "Chatbot not available"}), 503
    
    try:
        info = aura_bot.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": "Could not retrieve model information"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.before_request
def log_request_info():
    """Log request information for debugging."""
    logger.info(f"Request: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Add headers after request processing."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False for production
    )
