import io
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, load_dataset
import huggingface_hub
from fastapi.staticfiles import StaticFiles

# Configuration
MODEL_NAME = "google/gemma-2b-it"
MAX_SEQ_LENGTH = 512
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize FastAPI
app = FastAPI(
    title="Gemma Fine-Tuning API",
    description="Backend for fine-tuning Gemma models through a web interface",
    version="0.1.0"
)

# # Serve React frontend
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
app.mount("/static", StaticFiles(directory="frontend/assets"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"message": "Not Found", "detail": exc.detail}
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables (in-memory storage for demo purposes)
active_training_processes = {}
trained_models = {}
uploaded_datasets = {}

# Pydantic models for request/response validation
class Hyperparameters(BaseModel):
    learning_rate: float = Field(5e-5, gt=0)
    batch_size: int = Field(4, gt=0)
    num_epochs: int = Field(3, gt=0)
    max_seq_length: int = Field(MAX_SEQ_LENGTH, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    warmup_steps: int = Field(0, ge=0)
    logging_steps: int = Field(100, gt=0)
    save_steps: int = Field(500, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)

class TrainingRequest(BaseModel):
    dataset_id: str
    hyperparameters: Hyperparameters
    base_model: str = MODEL_NAME

class DatasetInfo(BaseModel):
    id: str
    name: str
    size: int
    created_at: str
    num_samples: int
    columns: list[str]

# Utility functions
def generate_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def validate_dataset(file: UploadFile) -> pd.DataFrame:
    try:
        # DEBUG: Full file inspection
        print(f"\n🔍 File Inspection:")
        print(f"Filename: {file.filename}")
        print(f"Content-type: {file.content_type}")
        print(f"Headers: {file.headers}")

        # Read raw bytes
        content = file.file.read()
        print(f"Raw size: {len(content)} bytes")

        # Extension detection (bulletproof)
        filename = file.filename.lower()
        is_txt = filename.endswith('.txt')
        is_csv = filename.endswith('.csv')
        is_json = filename.endswith(('.json', '.jsonl'))
        
        print(f"Extensions - TXT:{is_txt} CSV:{is_csv} JSON:{is_json}")

        if not (is_txt or is_csv or is_json):
            raise HTTPException(400, "Unsupported file format")

        # Decode content
        try:
            text_content = content.decode('utf-8-sig')  # Handles BOM
        except UnicodeDecodeError:
            text_content = content.decode('latin-1')
            print("⚠ Used latin-1 fallback decoding")

        print(f"First 100 chars:\n{text_content[:100]}")

        if is_txt:
            if not text_content.strip():
                raise HTTPException(400, "Text file is empty")
            return pd.DataFrame({'text': [text_content]})  # Full content as one sample

        elif is_csv:
            return pd.read_csv(io.BytesIO(content))
            
        elif is_json:
            return pd.read_json(io.BytesIO(content), lines=True)

    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {str(e)}")
        raise HTTPException(400, f"Processing failed: {str(e)}")
    finally:
        file.file.seek(0) 

def preprocess_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    try:
        # Convert pandas DataFrame to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(df)
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
        # Apply tokenization
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"] if "text" in hf_dataset.column_names else []
        )
        
        return tokenized_dataset
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing dataset: {str(e)}")

# API Endpoints
@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), name: str = Form(...)):
    try:
        # Debug: Log received file metadata
        logger.info(f"Received file: {file.filename} (Size: {file.size} bytes)")
        logger.info(f"Content type: {file.content_type}")
        
        # Debug: Save a temporary copy (remove in production)
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info(f"Saved temp copy to: {temp_path}")
        
        # Reset file pointer for validation
        await file.seek(0)
        
        # Enhanced CSV validation
        if file.filename.lower().endswith('.csv'):
            try:
                # First try standard CSV parsing
                content = await file.read()
                df = pd.read_csv(io.BytesIO(content))
                
                # Check for required 'text' column
                if 'text' not in df.columns:
                    logger.warning("CSV missing 'text' column, attempting fallback parsing")
                    df = pd.DataFrame({'text': pd.read_csv(io.BytesIO(content), header=None)[0].tolist()})
                    
            except Exception as csv_error:
                logger.warning(f"CSV parsing failed, trying text fallback: {str(csv_error)}")
                await file.seek(0)
                content = await file.read()
                df = pd.DataFrame({'text': [content.decode('utf-8')]})
        else:
            # Original validation for other file types
            df = validate_dataset(file)
        
        dataset_id = generate_id("dataset")
        
        uploaded_datasets[dataset_id] = {
            "df": df,
            "name": name,
            "size": file.size,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Processed dataset. Samples: {len(df)} | Columns: {list(df.columns)}")
        
        return {
            "id": dataset_id,
            "name": name,
            "size": file.size,
            "created_at": uploaded_datasets[dataset_id]["created_at"],
            "num_samples": len(df),
            "columns": list(df.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
    
@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets():
    """List all uploaded datasets"""
    return [
        {
            "id": dataset_id,
            "name": details["name"],
            "size": details["size"],
            "created_at": details["created_at"],
            "num_samples": len(details["df"]),
            "columns": list(details["df"].columns)
        }
        for dataset_id, details in uploaded_datasets.items()
    ]

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview samples from a dataset"""
    if dataset_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = uploaded_datasets[dataset_id]["df"]
    return JSONResponse(content=df.head(limit).to_dict(orient="records"))



@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a specific dataset by ID"""
    if dataset_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = uploaded_datasets[dataset_id]
    return {
        "id": dataset_id,
        "name": dataset["name"],
        "size": dataset["size"],
        "created_at": dataset["created_at"],
        "num_samples": len(dataset["df"]),
        "columns": list(dataset["df"].columns)
    }



@app.post("/api/train")
async def start_training(request: TrainingRequest):

    logger.info(f"Training request received: {request.json()}")
    logger.info(f"Current datasets: {list(uploaded_datasets.keys())}")
    try:
        # Enhanced dataset verification
        if request.dataset_id not in uploaded_datasets:
            logger.error(f"Dataset not found: {request.dataset_id}")
            raise HTTPException(status_code=404, detail="Dataset not found")
            
        dataset = uploaded_datasets[request.dataset_id]
        if len(dataset["df"]) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty")
            
        logger.info(f"Starting training with dataset: {request.dataset_id}")
        
        # Create mock training job
        job_id = f"job_{datetime.now().timestamp()}"
        active_training_processes[job_id] = {
            "status": "queued",
            "progress": 0,
            "dataset_id": request.dataset_id,
            "hyperparameters": {
                "base_model": request.base_model,
                "learning_rate": request.hyperparameters.learning_rate,
                "batch_size": request.hyperparameters.batch_size,
                "num_epochs": request.hyperparameters.num_epochs
            }
        }
        
        return {
            "job_id": job_id,
            "status": "queued",
            "dataset_id": request.dataset_id,
            "model": request.base_model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training initialization failed: {str(e)}")


@app.get("/api/train/{job_id}/status")
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in active_training_processes:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # In a real implementation, you would check the actual progress
    # For demo, we'll simulate progress
    status = active_training_processes[job_id]
    
    # Simulate progress updates
    if status["status"] == "queued":
        status["status"] = "running"
        status["progress"] = 5
    elif status["status"] == "running" and status["progress"] < 100:
        status["progress"] = min(status["progress"] + 5, 100)
        
        # Add some mock metrics
        status["metrics"] = {
            "loss": 1.5 - (status["progress"] * 0.01),
            "accuracy": status["progress"] * 0.008,
            "learning_rate": status["hyperparameters"]["learning_rate"]
        }
    
    if status["progress"] >= 100:
        status["status"] = "completed"
        
        # Generate a mock model ID
        model_id = generate_id("model")
        trained_models[model_id] = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "base_model": status["hyperparameters"]["base_model"],
            "metrics": status["metrics"]
        }
        status["model_id"] = model_id
    
    return status

@app.get("/api/models", response_model=list[dict])
async def list_trained_models():
    """List all trained models"""
    return [
        {
            "id": model_id,
            "job_id": details["job_id"],
            "created_at": details["created_at"],
            "base_model": details["base_model"],
            "metrics": details["metrics"]
        }
        for model_id, details in trained_models.items()
    ]

@app.get("/api/models/{model_id}/download")
async def download_model(model_id: str, format: str = "pytorch"):
    """Download a trained model in specified format"""
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # In a real implementation, you would:
    # 1. Locate the actual model files
    # 2. Convert to requested format if needed
    # 3. Stream the files back
    
    # For demo purposes, we'll return a mock response
    return JSONResponse(
        content={
            "message": f"Model {model_id} would be downloaded in {format} format in a real implementation",
            "status": "success"
        }
    )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    print(f"Starting server on port {port}...")  # Debug log
    uvicorn.run(app, host="0.0.0.0", port=port)