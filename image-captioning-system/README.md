# Image Captioning System

A comprehensive multi-tier image captioning system with research/prototype and production capabilities.

## Features

- **Multi-tier Architecture**: Fast prototyping with CNN+Transformer, production with ViT+Transformer
- **Large-scale Datasets**: Support for COCO, Conceptual Captions, and custom datasets
- **REST API**: Complete FastAPI backend with multiple endpoints
- **Web UI**: Lightweight interface for image upload and captioning
- **Safety & Privacy**: Built-in content filtering and PII protection
- **Monitoring**: Comprehensive logging and metrics
- **Deployment Ready**: Docker containers and deployment configurations

## Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository>
cd image-captioning-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models and datasets
python scripts/download_models.py
python scripts/download_datasets.py
```

### 2. Training

```bash
# Train baseline model
python train.py --config configs/baseline.yaml

# Train production model
python train.py --config configs/production.yaml
```

### 3. API Server

```bash
# Start the API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Web UI

```bash
# Start the web interface
python -m web.main
```

## API Endpoints

- `POST /v1/caption` - Generate captions for images
- `POST /v1/caption/batch` - Batch captioning
- `POST /v1/alt_text` - Generate accessible alt text
- `POST /v1/style_caption` - Style-specific captions
- `GET /v1/health` - Health check

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Architecture

- **Baseline Tier**: ResNet50 + Transformer decoder for fast prototyping
- **Production Tier**: ViT + Transformer with CLIP integration for high quality
- **API Layer**: FastAPI with async processing and caching
- **UI Layer**: Modern web interface with real-time preview
- **Monitoring**: Prometheus metrics and structured logging

## Datasets

- MS COCO (validation and fine-tuning)
- Conceptual Captions 3M/12M (pretraining)
- Custom domain-specific datasets
- Automatic filtering and quality assessment

## Safety Features

- No face recognition or person identification
- PII detection and removal
- NSFW content filtering
- Bias mitigation and fairness testing
- Content moderation pipelines

## License

MIT License - see LICENSE file for details.
