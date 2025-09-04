# deepfake_detector/app/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
import sys
from typing import List, Optional
from pydantic import BaseModel
import logging
import hashlib
import json
from datetime import datetime
import asyncio

sys.path.append('../src')
from inference import DeepfakeInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    processing_time: float
    file_hash: Optional[str] = None
    timestamp: str

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_files: int
    successful_predictions: int
    failed_predictions: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered deepfake detection system with REST API",
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

# Global detector instance
detector = None
model_loaded = False

def load_model():
    """Load the deepfake detection model"""
    global detector, model_loaded
    try:
        model_path = "../models/checkpoint_best.pth"
        if os.path.exists(model_path):
            detector = DeepfakeInference(model_path)
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    content = await file.read()
    
    # Create temporary file
    suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        temp_path = tmp_file.name
    
    return temp_path, content

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), include_heatmap: bool = False):
    """Predict deepfake for uploaded image"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Save temporary file
        temp_path, content = await save_temp_file(file)
        file_hash = calculate_file_hash(content)
        
        # Make prediction
        result = detector.predict_image(temp_path, return_heatmap=include_heatmap)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Prepare response
        response = PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=processing_time,
            file_hash=file_hash,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Image prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...), 
    max_frames: int = 30,
    return_frame_results: bool = False
):
    """Predict deepfake for uploaded video"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    temp_path = None
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Save temporary file
        temp_path, content = await save_temp_file(file)
        file_hash = calculate_file_hash(content)
        
        # Make prediction
        result = detector.predict_video(
            temp_path, 
            max_frames=max_frames,
            return_frame_results=return_frame_results
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Prepare response
        response_data = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['avg_probabilities'],
            "processing_time": processing_time,
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add video-specific fields
        if return_frame_results and 'frame_results' in result:
            response_data['frame_results'] = result['frame_results']
        
        response_data['frames_analyzed'] = result['frames_analyzed']
        response_data['total_frames'] = result['total_frames']
        
        logger.info(f"Video prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict deepfake for multiple files"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        temp_path = None
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Save temporary file
            temp_path, content = await save_temp_file(file)
            file_hash = calculate_file_hash(content)
            
            # Determine file type and predict accordingly
            if file.content_type.startswith('image/'):
                result = detector.predict_image(temp_path, return_heatmap=False)
            elif file.content_type.startswith('video/'):
                result = detector.predict_video(temp_path, max_frames=10)
                # Adjust result format for consistency
                result['probabilities'] = result['avg_probabilities']
            else:
                raise ValueError("Unsupported file type")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            response = PredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                processing_time=processing_time,
                file_hash=file_hash,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(response)
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            # Add error result
            error_response = PredictionResponse(
                prediction="Error",
                confidence=0.0,
                probabilities={"real": 0.0, "fake": 0.0},
                processing_time=0.0,
                timestamp=datetime.now().isoformat()
            )
            results.append(error_response)
            failed += 1
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    return BatchPredictionResponse(
        results=results,
        total_files=len(files),
        successful_predictions=successful,
        failed_predictions=failed
    )

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": "EfficientNet-B0",
        "input_size": "224x224",
        "classes": ["Real", "Fake"],
        "device": str(detector.device),
        "loaded": True
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    try:
        load_model()
        if model_loaded:
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Reload error: {str(e)}")

# Blockchain integration endpoints (optional)
@app.post("/blockchain/verify")
async def verify_authenticity(file_hash: str):
    """Verify file authenticity using blockchain (placeholder)"""
    # This is a placeholder for blockchain integration
    # In a real implementation, this would check a blockchain for file hash
    return {
        "file_hash": file_hash,
        "verified": False,
        "blockchain_record": None,
        "message": "Blockchain verification not implemented"
    }

@app.post("/blockchain/register")
async def register_authentic_content(file_hash: str, metadata: dict = None):
    """Register authentic content on blockchain (placeholder)"""
    # This is a placeholder for blockchain integration
    return {
        "file_hash": file_hash,
        "registered": False,
        "transaction_id": None,
        "message": "Blockchain registration not implemented"
    }

# WebSocket endpoint for real-time processing
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    await websocket.accept()
    
    if not model_loaded:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return
    
    try:
        while True:
            # Wait for file data
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # Process prediction request
            # Note: This is a simplified example
            # In practice, you'd handle binary file data differently
            await websocket.send_json({
                "type": "prediction",
                "status": "processing",
                "message": "WebSocket prediction processing not fully implemented"
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )