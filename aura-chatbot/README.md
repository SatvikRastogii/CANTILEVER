# Aura - Empathetic AI Chatbot

Aura is a sequence-to-sequence NLP chatbot designed to provide empathetic and supportive responses for users seeking mental health support and emotional guidance.

## 🌟 Features

- **Empathetic Conversations**: Trained on mental health and supportive dialogue data
- **Modern Web Interface**: Clean, minimal, and responsive design
- **Real-time Chat**: Instant responses with typing indicators
- **Privacy-Focused**: No data storage, anonymous conversations
- **Mobile-Friendly**: Responsive design for all devices
- **Accessibility**: WCAG compliant with dark mode support

## 🚀 Quick Start

### Prerequisites

- Python 3.13.x
- pip (Python package installer)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd aura-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

### Dataset Creation

1. **Generate the training dataset**
   ```bash
   python create_dataset.py
   ```
   This will create a large dataset (180,000+ conversation pairs) in `data/aura_dataset.csv`.

### Model Training

1. **Train the chatbot model**
   ```bash
   python train.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train a sequence-to-sequence LSTM model with attention
   - Save the trained models to `saved_model/`

### Running Locally

1. **Start the Flask application**
   ```bash
   python app/app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
aura-chatbot/
├── app/
│   ├── static/
│   │   └── styles.css          # Modern CSS styling
│   ├── templates/
│   │   └── index.html          # Chat interface
│   └── app.py                  # Flask web application
├── data/
│   └── aura_dataset.csv        # Training dataset (generated)
├── saved_model/
│   ├── encoder_model.h5        # Trained encoder model
│   ├── decoder_model.h5        # Trained decoder model
│   ├── tokenizer.json          # Text tokenizer
│   └── params.json             # Model parameters
├── create_dataset.py           # Dataset generation script
├── preprocessing.py            # Data preprocessing utilities
├── train.py                    # Model training script
├── nlp_model.py                # Chatbot inference class
├── requirements.txt            # Python dependencies
├── Procfile                    # Heroku deployment config
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🎯 Model Architecture

- **Encoder-Decoder Architecture**: LSTM-based sequence-to-sequence model
- **Attention Mechanism**: Improves response quality and relevance
- **Embedding Layer**: 256-dimensional word embeddings
- **Vocabulary Size**: 30,000 words
- **Sequence Length**: 150 tokens maximum
- **Latent Dimension**: 512 units

## 🚀 Deployment to Heroku

### Prerequisites for Deployment

1. **Heroku CLI** installed
2. **Git** repository initialized
3. **Heroku account** created

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku app**
   ```bash
   heroku create aura-chatbot-unique-name
   ```
   Replace `aura-chatbot-unique-name` with your unique app name.

3. **Initialize Git repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

4. **Add Heroku remote**
   ```bash
   heroku git:remote -a aura-chatbot-unique-name
   ```

5. **Deploy to Heroku**
   ```bash
   git push heroku main
   ```

6. **Open your deployed app**
   ```bash
   heroku open
   ```

### Important Notes for Heroku Deployment

- **Model Files**: The trained models are excluded from Git (see `.gitignore`). For production deployment, you'll need to:
  - Train the model locally first
  - Use a cloud storage service (AWS S3, Google Cloud Storage) to host model files
  - Modify the code to download models on app startup
  - Or use Heroku's ephemeral filesystem (models will be retrained on each deployment)

- **Memory Requirements**: The model requires significant memory. Consider upgrading to a paid Heroku dyno.

- **Buildpacks**: Heroku will automatically detect Python and install dependencies.

## 🔧 Configuration

### Environment Variables

You can set these environment variables for customization:

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Flask environment (development/production)

### Model Parameters

Key parameters in `train.py`:

```python
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 512
VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 256
```

## 🧪 Testing

### Test the Chatbot

```bash
python nlp_model.py
```

This will run a series of test conversations to verify the model is working correctly.

### Health Check

Visit `/health` endpoint to check if the application is running properly:

```bash
curl http://localhost:5000/health
```

## 📊 Dataset Information

The training dataset includes:

- **180,000+ conversation pairs**
- **Synthetic mental health conversations**
- **Reddit-style supportive dialogues**
- **Professional counseling responses**
- **Cleaned and filtered for quality**

### Data Sources

- Synthetic data generation with empathetic templates
- Mental health conversation patterns
- Supportive response frameworks
- Professional counseling dialogue structures

## 🛡️ Safety and Ethics

### Important Disclaimers

- **Not a Substitute**: Aura is not a replacement for professional mental health care
- **Crisis Situations**: For mental health emergencies, contact:
  - National Suicide Prevention Lifeline: 988
  - Crisis Text Line: Text HOME to 741741
  - Emergency Services: 911

### Privacy

- No conversation data is stored permanently
- All interactions are anonymous
- No personal information is collected

## 🔍 Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `saved_model/` directory exists with trained models
   - Check file permissions
   - Verify model files are not corrupted

2. **Memory errors during training**
   - Reduce `BATCH_SIZE` in `train.py`
   - Use a machine with more RAM
   - Consider using Google Colab or AWS for training

3. **Poor response quality**
   - Increase training epochs
   - Adjust model architecture parameters
   - Ensure dataset quality and size

4. **Heroku deployment issues**
   - Check Heroku logs: `heroku logs --tail`
   - Verify all dependencies in `requirements.txt`
   - Ensure `Procfile` is correct

### Performance Optimization

- **Model Size**: Consider model quantization for faster inference
- **Caching**: Implement response caching for common queries
- **Load Balancing**: Use multiple dynos for high traffic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with TensorFlow/Keras for deep learning
- Flask for web framework
- NLTK for natural language processing
- Heroku for deployment platform
- Inspired by the need for accessible mental health support

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the GitHub issues
3. Create a new issue with detailed information

---

**Remember**: Aura is designed to provide supportive conversations, but it's not a replacement for professional mental health care. Always seek professional help when needed.
