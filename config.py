import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Application configuration"""
    
    # Project settings
    PROJECT_NAME: str = "black-friday-mlops"
    VERSION: str = "1.0.0"
    
    # Data settings
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    
    # Model settings
    MODEL_DIR: str = "models"
    EXPERIMENTS_DIR: str = os.path.join(MODEL_DIR, "experiments")
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    API_DEBUG: bool = False
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "blackfriday-sales-prediction"
    
    # AWS settings
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "blackfriday-mlops-data"
    
    # Model parameters
    MODEL_PARAMS: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.MODEL_PARAMS is None:
            self.MODEL_PARAMS = {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }

# Create config instance
config = Config()
