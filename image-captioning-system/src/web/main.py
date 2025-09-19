"""
Web UI for image captioning system.
"""

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import io
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Image Captioning Web UI")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# API configuration
API_BASE_URL = "http://localhost:8000"  # Configure based on your setup


class CaptionRequest(BaseModel):
    image_base64: str
    style: str = "descriptive"
    max_length: int = 50
    beam_size: int = 1
    temperature: float = 1.0
    model_type: str = "production"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with image upload interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("descriptive"),
    max_length: int = Form(50),
    beam_size: int = Form(1),
    temperature: float = Form(1.0),
    model_type: str = Form("production")
):
    """Handle image upload and caption generation."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare API request
        api_request = CaptionRequest(
            image_base64=image_base64,
            style=style,
            max_length=max_length,
            beam_size=beam_size,
            temperature=temperature,
            model_type=model_type
        )
        
        # Call API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/v1/caption",
                json=api_request.dict(),
                timeout=30.0
            )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        
        # Return result with image data for display
        return JSONResponse({
            "success": True,
            "result": result,
            "image_base64": image_base64
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/batch-upload")
async def batch_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    style: str = Form("descriptive"),
    max_length: int = Form(50),
    beam_size: int = Form(1),
    temperature: float = Form(1.0),
    model_type: str = Form("production")
):
    """Handle batch image upload and caption generation."""
    try:
        # Validate files
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
        
        # Process images
        images_base64 = []
        for file in files:
            image_data = await file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            images_base64.append(image_base64)
        
        # Prepare API request
        api_request = {
            "images": images_base64,
            "style": style,
            "max_length": max_length,
            "beam_size": beam_size,
            "temperature": temperature,
            "model_type": model_type
        }
        
        # Call API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/v1/caption/batch",
                json=api_request,
                timeout=60.0
            )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        
        # Return result with image data for display
        return JSONResponse({
            "success": True,
            "result": result,
            "images_base64": images_base64
        })
        
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/v1/health", timeout=5.0)
        
        if response.status_code == 200:
            return JSONResponse({"status": "healthy", "api_status": "connected"})
        else:
            return JSONResponse({"status": "unhealthy", "api_status": "disconnected"})
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "api_status": "error", "error": str(e)})


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel for monitoring and management."""
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/api/status")
async def get_api_status():
    """Get API status and metrics."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/v1/health", timeout=5.0)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": "API not responding"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
