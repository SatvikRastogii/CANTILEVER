#!/usr/bin/env python3
"""
Setup and test script for Aura chatbot.
This script will:
1. Create a sample dataset
2. Test the preprocessing
3. Verify the Flask app can start
4. Run basic functionality tests
"""

import os
import sys
import subprocess
import time
import requests
from threading import Thread
import signal

def print_step(step_num, description):
    """Print a formatted step description."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\nRunning: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print("❌ Error!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def test_dataset_creation():
    """Test dataset creation."""
    print_step(1, "Testing Dataset Creation")
    
    # Import and test preprocessing
    try:
        from preprocessing import save_sample_dataset
        print("✅ Preprocessing module imported successfully")
        
        # Create a small sample dataset
        dataset_path = 'data/aura_dataset.csv'
        os.makedirs('data', exist_ok=True)
        
        num_rows = save_sample_dataset(dataset_path)
        print(f"✅ Sample dataset created with {num_rows} rows")
        
        # Test loading the dataset
        from preprocessing import load_and_preprocess_data
        prompts, responses = load_and_preprocess_data(dataset_path)
        print(f"✅ Dataset loaded successfully: {len(prompts)} conversation pairs")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False

def test_model_imports():
    """Test that all model components can be imported."""
    print_step(2, "Testing Model Imports")
    
    try:
        # Test preprocessing import
        import preprocessing
        print("✅ Preprocessing module imported")
        
        # Test NLP model import
        import nlp_model
        print("✅ NLP model module imported")
        
        # Test Flask app import
        sys.path.append('app')
        import app
        print("✅ Flask app module imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_chatbot_functionality():
    """Test basic chatbot functionality."""
    print_step(3, "Testing Chatbot Functionality")
    
    try:
        from nlp_model import get_aura_bot
        
        # Get chatbot instance
        bot = get_aura_bot()
        print("✅ Chatbot instance created")
        
        # Test model info
        info = bot.get_model_info()
        print(f"✅ Model info retrieved: {info}")
        
        # Test response generation (will use fallback since no trained model)
        test_input = "I'm feeling anxious today"
        response = bot.generate_response(test_input)
        print(f"✅ Response generated: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot functionality test failed: {e}")
        return False

def start_flask_app():
    """Start Flask app in a separate thread."""
    try:
        os.chdir('app')
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Flask app error: {e}")

def test_flask_app():
    """Test Flask app functionality."""
    print_step(4, "Testing Flask Application")
    
    try:
        # Start Flask app in background
        flask_thread = Thread(target=start_flask_app, daemon=True)
        flask_thread.start()
        
        # Wait for app to start
        print("Waiting for Flask app to start...")
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:5000/health', timeout=10)
            if response.status_code == 200:
                print("✅ Health endpoint working")
                health_data = response.json()
                print(f"Health data: {health_data}")
            else:
                print(f"❌ Health endpoint returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to Flask app: {e}")
            return False
        
        # Test chat endpoint
        try:
            chat_data = {"message": "Hello, I'm feeling sad today"}
            response = requests.post('http://localhost:5000/get_response', 
                                   json=chat_data, timeout=10)
            if response.status_code == 200:
                print("✅ Chat endpoint working")
                response_data = response.json()
                print(f"Chat response: {response_data.get('response', 'No response')[:100]}...")
            else:
                print(f"❌ Chat endpoint returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Chat endpoint test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print_step(5, "Testing File Structure")
    
    required_files = [
        'requirements.txt',
        'Procfile',
        '.gitignore',
        'preprocessing.py',
        'create_dataset.py',
        'train.py',
        'nlp_model.py',
        'app/app.py',
        'app/templates/index.html',
        'app/static/styles.css',
        'README.md'
    ]
    
    required_dirs = [
        'app',
        'app/static',
        'app/templates',
        'data',
        'saved_model'
    ]
    
    all_good = True
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_good = False
    
    return all_good

def main():
    """Main test function."""
    print("🚀 AURA CHATBOT - SETUP AND TEST SCRIPT")
    print("This script will test all components of the Aura chatbot project.")
    
    # Change to project directory
    if not os.path.exists('aura-chatbot'):
        print("❌ Please run this script from the parent directory of aura-chatbot")
        return False
    
    os.chdir('aura-chatbot')
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    tests = [
        test_file_structure,
        test_dataset_creation,
        test_model_imports,
        test_chatbot_functionality,
        test_flask_app
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print_step(6, "TEST SUMMARY")
    
    test_names = [
        "File Structure",
        "Dataset Creation", 
        "Model Imports",
        "Chatbot Functionality",
        "Flask Application"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! The Aura chatbot is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python create_dataset.py' to generate the full dataset")
        print("2. Run 'python train.py' to train the model")
        print("3. Run 'python app/app.py' to start the web application")
        print("4. Open http://localhost:5000 in your browser")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
