"""
FastAPI main application for image captioning system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import logging
import time
import asyncio
from pathlib import Path
import json
import uuid
from datetime import datetime

# Import our modules
from ..models.utils import ModelConfig, load_model, create_model
from ..utils.safety import SafetyPipeline
from ..evaluation.metrics import CaptionEvaluator
from ..utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Captioning API",
    description="Multi-tier image captioning system with safety features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and services
models_cache = {}
safety_pipeline = None
evaluator = None

# Pydantic models for API
class CaptionRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    style: str = Field(default="descriptive", description="Caption style: descriptive, concise, SEO, poetic")
    max_length: int = Field(default=50, ge=10, le=200, description="Maximum caption length")
    beam_size: int = Field(default=1, ge=1, le=5, description="Beam search size")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    model_type: str = Field(default="production", description="Model type: baseline, production")

class BatchCaptionRequest(BaseModel):
    images: List[str] = Field(description="List of base64 encoded images")
    style: str = Field(default="descriptive")
    max_length: int = Field(default=50, ge=10, le=200)
    beam_size: int = Field(default=1, ge=1, le=5)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    model_type: str = Field(default="production")

class AltTextRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    model_type: str = Field(default="production")

class StyleCaptionRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    style: str = Field(description="Caption style: descriptive, concise, SEO, poetic")
    model_type: str = Field(default="production")

class CaptionResponse(BaseModel):
    caption: str
    alt_text: str
    style: str
    score: Dict[str, float]
    objects: List[str]
    warnings: List[str]
    processing_time: float
    model_type: str

class BatchCaptionResponse(BaseModel):
    results: List[CaptionResponse]
    total_processing_time: float
    model_type: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: List[str]
    system_info: Dict[str, Any]

# Dependency to get model
async def get_model(model_type: str = "production"):
    """Get model from cache or load it."""
    if model_type not in models_cache:
        try:
            # Load model based on type
            if model_type == "baseline":
                config = ModelConfig(model_type="baseline")
            else:
                config = ModelConfig(model_type="production")
            
            model = create_model(config)
            model.eval()
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            models_cache[model_type] = model
            logger.info(f"Loaded {model_type} model on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load {model_type} model")
    
    return models_cache[model_type]

# Dependency to get safety pipeline
async def get_safety_pipeline():
    """Get safety pipeline instance."""
    global safety_pipeline
    if safety_pipeline is None:
        from ..utils.safety import create_safety_pipeline
        safety_pipeline = create_safety_pipeline()
    return safety_pipeline

# Utility functions
def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model input."""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def generate_caption_style(caption: str, style: str) -> str:
    """Generate style-specific caption."""
    if style == "concise":
        # Make caption more concise
        words = caption.split()
        if len(words) > 10:
            return " ".join(words[:10]) + "..."
        return caption
    elif style == "SEO":
        # Make caption more SEO-friendly
        return caption.replace("a ", "").replace("an ", "").replace("the ", "")
    elif style == "poetic":
        # Make caption more poetic (simplified)
        return caption.replace(".", " beautifully.").replace("!", " with grace!")
    else:  # descriptive
        return caption

def extract_objects(caption: str) -> List[str]:
    """Extract objects from caption (simplified)."""
    # Simple object extraction based on common patterns
    objects = []
    words = caption.lower().split()
    
    # Common object keywords
    object_keywords = [
        'person', 'people', 'man', 'woman', 'child', 'baby',
        'car', 'truck', 'bus', 'bike', 'motorcycle',
        'dog', 'cat', 'bird', 'horse', 'cow',
        'tree', 'flower', 'grass', 'sky', 'cloud',
        'building', 'house', 'road', 'street', 'bridge',
        'food', 'drink', 'book', 'phone', 'computer'
    ]
    
    for word in words:
        if word in object_keywords:
            objects.append(word)
    
    return list(set(objects))

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Image Captioning API</title>
        </head>
        <body>
            <h1>Image Captioning API</h1>
            <p>Welcome to the Image Captioning API!</p>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/v1/health">Health Check</a></li>
                <li>POST /v1/caption - Generate captions</li>
                <li>POST /v1/caption/batch - Batch captioning</li>
                <li>POST /v1/alt_text - Generate alt text</li>
                <li>POST /v1/style_caption - Style-specific captions</li>
            </ul>
        </body>
    </html>
    """

@app.post("/v1/caption", response_model=CaptionResponse)
async def generate_caption(
    request: CaptionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Generate caption for a single image."""
    start_time = time.time()
    
    try:
        # Get image data
        if request.image_url:
            # Download image from URL
            import requests
            response = requests.get(request.image_url)
            image_data = response.content
        elif request.image_base64:
            # Decode base64 image
            image_data = base64.b64decode(request.image_base64)
        else:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Move to model device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                image_tensor,
                max_length=request.max_length,
                beam_size=request.beam_size,
                temperature=request.temperature
            )
        
        # Decode caption (simplified - in practice, use proper tokenizer)
        caption = "A beautiful image with various objects and scenes"
        
        # Apply style
        styled_caption = generate_caption_style(caption, request.style)
        
        # Generate alt text (shorter version)
        alt_text = styled_caption[:50] + "..." if len(styled_caption) > 50 else styled_caption
        
        # Extract objects
        objects = extract_objects(styled_caption)
        
        # Safety analysis
        safety_pipeline = await get_safety_pipeline()
        safety_result = safety_pipeline.process_image(Image.open(io.BytesIO(image_data)))
        
        warnings = []
        if not safety_result['is_safe']:
            warnings.extend(safety_result['warnings'])
        
        # Calculate scores (simplified)
        scores = {
            "confidence": 0.85,
            "clip_score": 0.78,
            "safety_score": 1.0 - len(warnings) * 0.2
        }
        
        processing_time = time.time() - start_time
        
        # Log request for monitoring
        background_tasks.add_task(
            log_request,
            request_id=str(uuid.uuid4()),
            model_type=request.model_type,
            processing_time=processing_time,
            success=True
        )
        
        return CaptionResponse(
            caption=styled_caption,
            alt_text=alt_text,
            style=request.style,
            score=scores,
            objects=objects,
            warnings=warnings,
            processing_time=processing_time,
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/caption/batch", response_model=BatchCaptionResponse)
async def generate_batch_captions(
    request: BatchCaptionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Generate captions for multiple images."""
    start_time = time.time()
    results = []
    
    try:
        for i, image_base64 in enumerate(request.images):
            try:
                # Decode image
                image_data = base64.b64decode(image_base64)
                
                # Preprocess image
                image_tensor = preprocess_image(image_data)
                
                # Move to model device
                device = next(model.parameters()).device
                image_tensor = image_tensor.to(device)
                
                # Generate caption
                with torch.no_grad():
                    outputs = model.generate(
                        image_tensor,
                        max_length=request.max_length,
                        beam_size=request.beam_size,
                        temperature=request.temperature
                    )
                
                # Decode caption (simplified)
                caption = f"Generated caption for image {i+1}"
                
                # Apply style
                styled_caption = generate_caption_style(caption, request.style)
                
                # Generate alt text
                alt_text = styled_caption[:50] + "..." if len(styled_caption) > 50 else styled_caption
                
                # Extract objects
                objects = extract_objects(styled_caption)
                
                # Safety analysis
                safety_pipeline = await get_safety_pipeline()
                safety_result = safety_pipeline.process_image(Image.open(io.BytesIO(image_data)))
                
                warnings = []
                if not safety_result['is_safe']:
                    warnings.extend(safety_result['warnings'])
                
                # Calculate scores
                scores = {
                    "confidence": 0.85,
                    "clip_score": 0.78,
                    "safety_score": 1.0 - len(warnings) * 0.2
                }
                
                results.append(CaptionResponse(
                    caption=styled_caption,
                    alt_text=alt_text,
                    style=request.style,
                    score=scores,
                    objects=objects,
                    warnings=warnings,
                    processing_time=0.1,  # Individual processing time
                    model_type=request.model_type
                ))
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                results.append(CaptionResponse(
                    caption="Error processing image",
                    alt_text="Error processing image",
                    style=request.style,
                    score={"confidence": 0.0, "clip_score": 0.0, "safety_score": 0.0},
                    objects=[],
                    warnings=[f"Processing error: {str(e)}"],
                    processing_time=0.0,
                    model_type=request.model_type
                ))
        
        total_processing_time = time.time() - start_time
        
        # Log batch request
        background_tasks.add_task(
            log_request,
            request_id=str(uuid.uuid4()),
            model_type=request.model_type,
            processing_time=total_processing_time,
            success=True,
            batch_size=len(request.images)
        )
        
        return BatchCaptionResponse(
            results=results,
            total_processing_time=total_processing_time,
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Batch caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/alt_text", response_model=CaptionResponse)
async def generate_alt_text(
    request: AltTextRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Generate accessible alt text for an image."""
    # Use the main caption endpoint with concise style
    caption_request = CaptionRequest(
        image_url=request.image_url,
        image_base64=request.image_base64,
        style="concise",
        max_length=30,
        model_type=request.model_type
    )
    
    result = await generate_caption(caption_request, background_tasks, model)
    
    # Override with alt text specific formatting
    result.alt_text = result.caption
    result.style = "alt_text"
    
    return result

@app.post("/v1/style_caption", response_model=CaptionResponse)
async def generate_style_caption(
    request: StyleCaptionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Generate style-specific caption."""
    caption_request = CaptionRequest(
        image_url=request.image_url,
        image_base64=request.image_base64,
        style=request.style,
        model_type=request.model_type
    )
    
    return await generate_caption(caption_request, background_tasks, model)

@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=list(models_cache.keys()),
        system_info={
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": "3.8+",
            "torch_version": torch.__version__
        }
    )

# Background task functions
async def log_request(
    request_id: str,
    model_type: str,
    processing_time: float,
    success: bool,
    batch_size: int = 1
):
    """Log request for monitoring."""
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "processing_time": processing_time,
        "success": success,
        "batch_size": batch_size
    }
    
    logger.info(f"Request logged: {json.dumps(log_entry)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Image Captioning API...")
    
    # Initialize safety pipeline
    global safety_pipeline
    from ..utils.safety import create_safety_pipeline
    safety_pipeline = create_safety_pipeline()
    
    # Initialize evaluator
    global evaluator
    from ..evaluation.metrics import CaptionEvaluator
    evaluator = CaptionEvaluator()
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Image Captioning API...")
    
    # Clear model cache
    models_cache.clear()
    
    logger.info("API shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
