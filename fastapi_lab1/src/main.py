from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Create FastAPI app
app = FastAPI(
    title="🐧 Penguin Species Classification API",
    description="Machine Learning API for classifying Palmer Archipelago Penguins",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
species_encoder = None
numerical_imputer = None
island_encoder = None
sex_encoder = None

def load_models():
    """Load all models at startup"""
    global model, scaler, species_encoder, numerical_imputer, island_encoder, sex_encoder
    try:
        # Get correct path to model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, 'model')
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"⚠️ Model directory not found at: {model_dir}")
            print("Creating model directory...")
            os.makedirs(model_dir, exist_ok=True)
            return False
        
        # Load all required models
        model_path = os.path.join(model_dir, 'penguin_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        species_encoder_path = os.path.join(model_dir, 'species_encoder.pkl')
        numerical_imputer_path = os.path.join(model_dir, 'numerical_imputer.pkl')
        island_encoder_path = os.path.join(model_dir, 'island_encoder.pkl')
        sex_encoder_path = os.path.join(model_dir, 'sex_encoder.pkl')
        
        # Check if all model files exist
        required_files = [
            ('penguin_model.pkl', model_path),
            ('scaler.pkl', scaler_path),
            ('species_encoder.pkl', species_encoder_path),
            ('numerical_imputer.pkl', numerical_imputer_path),
            ('island_encoder.pkl', island_encoder_path),
            ('sex_encoder.pkl', sex_encoder_path)
        ]
        
        missing_files = []
        for name, path in required_files:
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            print(f"⚠️ Missing model files: {', '.join(missing_files)}")
            print("Please train the model first by:")
            print("1. Running: python train.py")
            print("2. Or use the API endpoint: POST /train")
            return False
        
        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        species_encoder = joblib.load(species_encoder_path)
        numerical_imputer = joblib.load(numerical_imputer_path)
        island_encoder = joblib.load(island_encoder_path)
        sex_encoder = joblib.load(sex_encoder_path)
        
        print("✅ All models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("Please train the model first")
        return False

# Load models on startup
models_loaded = load_models()

# Pydantic Models
class PenguinFeatures(BaseModel):
    """Complete penguin features including measurements and metadata"""
    bill_length_mm: Optional[float] = Field(None, ge=30, le=60, description="Bill length (30-60 mm)")
    bill_depth_mm: Optional[float] = Field(None, ge=13, le=22, description="Bill depth (13-22 mm)")
    flipper_length_mm: Optional[float] = Field(None, ge=170, le=235, description="Flipper length (170-235 mm)")
    body_mass_g: Optional[float] = Field(None, ge=2700, le=6300, description="Body mass (2700-6300 g)")
    island: str = Field("Biscoe", description="Island: Torgersen, Biscoe, or Dream")
    sex: Optional[str] = Field(None, description="Sex: male, female, or unknown")
    year: int = Field(2008, ge=2007, le=2009, description="Year of observation")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v is None or v == '' or (isinstance(v, str) and v.lower() == 'unknown'):
            return 'unknown'
        if isinstance(v, str) and v.lower() not in ['male', 'female']:
            return 'unknown'
        return v.lower() if isinstance(v, str) else 'unknown'
    
    @field_validator('island')
    @classmethod
    def validate_island(cls, v):
        valid_islands = ['Torgersen', 'Biscoe', 'Dream']
        if v not in valid_islands:
            return 'Biscoe'
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bill_length_mm": 39.1,
                    "bill_depth_mm": 18.7,
                    "flipper_length_mm": 181.0,
                    "body_mass_g": 3750.0,
                    "island": "Torgersen",
                    "sex": "male",
                    "year": 2007
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    species: str
    confidence: float
    probabilities: Dict[str, float]
    confidence_level: str
    fun_fact: str
    species_info: Dict[str, Any]
    input_summary: Dict[str, Any]
    data_quality: Dict[str, bool]

class TrainingResponse(BaseModel):
    """Response model for training"""
    status: str
    steps: List[str]
    best_model: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# API Endpoints
@app.get("/", tags=["General"])
async def home():
    """Welcome endpoint with API information"""
    return {
        "name": "🐧 Penguin Species Classification API",
        "version": "2.0.0",
        "description": "Classify Palmer Archipelago penguins",
        "status": "🟢 Ready" if models_loaded else "🔴 Models not loaded",
        "models_loaded": models_loaded,
        "endpoints": {
            "GET /": "This message",
            "POST /predict": "Classify a penguin",
            "GET /health": "API health status",
            "POST /train": "Train the model",
            "GET /metrics": "Get model metrics",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "API documentation (ReDoc)"
        },
        "instructions": {
            "train_first": "If models are not loaded, use POST /train endpoint first",
            "documentation": "Visit /docs for interactive API testing"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy" if models_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "api_version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(penguin: PenguinFeatures):
    """Predict penguin species based on features"""
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train the model first using POST /train endpoint."
        )
    
    try:
        # Import prediction module from same directory
        from predict import make_prediction
        
        # Convert Pydantic model to dictionary
        features_dict = {
            'bill_length_mm': penguin.bill_length_mm,
            'bill_depth_mm': penguin.bill_depth_mm,
            'flipper_length_mm': penguin.flipper_length_mm,
            'body_mass_g': penguin.body_mass_g,
            'island': penguin.island,
            'sex': penguin.sex if penguin.sex else 'unknown',
            'year': penguin.year
        }
        
        # Make prediction
        result = make_prediction(features_dict)
        
        # Ensure all required fields are present
        if not isinstance(result, dict):
            raise ValueError("Invalid prediction result format")
        
        return PredictionResponse(**result)
        
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import prediction module: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model():
    """Train the machine learning model"""
    try:
        # Import training modules from same directory
        from train import train_multiple_models, train_best_model
        from data import load_data, split_data, preprocess_data
        
        response_steps = []
        
        # Step 1: Load data
        print("Loading data...")
        X, y, feature_names, target_names = load_data()
        response_steps.append(f"✅ Data loaded: {len(X)} samples with {len(feature_names)} features")
        
        # Step 2: Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        response_steps.append(f"✅ Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Step 3: Preprocess data
        print("Preprocessing data...")
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        response_steps.append("✅ Data preprocessed and scaled")
        
        # Step 4: Train multiple models and find best
        print("Training multiple models...")
        best_model, best_name, results = train_multiple_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        response_steps.append(f"✅ Best model identified: {best_name}")
        
        # Step 5: Train final model
        print(f"Training final {best_name} model...")
        final_model, metrics = train_best_model(
            X_train_scaled, y_train, X_test_scaled, y_test, best_name
        )
        response_steps.append(f"✅ Final model trained with accuracy: {metrics.get('accuracy', 0):.2%}")
        
        # Step 6: Save metrics
        response_steps.append("✅ Model and metrics saved")
        
        # Reload models
        global models_loaded
        models_loaded = load_models()
        
        if models_loaded:
            response_steps.append("✅ Models reloaded successfully")
        
        return TrainingResponse(
            status="success",
            steps=response_steps,
            best_model=best_name,
            metrics=metrics
        )
        
    except ImportError as e:
        return TrainingResponse(
            status="error",
            steps=[],
            error=f"Import error: {str(e)}. Make sure all required modules (train.py, data.py) are in the same directory."
        )
    except Exception as e:
        return TrainingResponse(
            status="error",
            steps=[],
            error=f"Training failed: {str(e)}"
        )

@app.get("/metrics", tags=["Performance"])
async def get_metrics():
    """Get model performance metrics"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        metrics_path = os.path.join(project_root, 'model', 'metrics.json')
        
        if not os.path.exists(metrics_path):
            return {
                "message": "No metrics available. Please train the model first.",
                "status": "not_found"
            }
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add additional context
        metrics['status'] = 'available'
        metrics['timestamp'] = os.path.getmtime(metrics_path)
        
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metrics: {str(e)}"
        )

@app.get("/feature-importance", tags=["Performance"])
async def get_feature_importance():
    """Get feature importance from the trained model"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        importance_path = os.path.join(project_root, 'model', 'feature_importance.json')
        
        if not os.path.exists(importance_path):
            return {
                "message": "No feature importance data available.",
                "status": "not_found"
            }
        
        with open(importance_path, 'r') as f:
            importance_data = json.load(f)
        
        return importance_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load feature importance: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Get the host and port from environment variables or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting API server on {host}:{port}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    
    # Run the application
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )