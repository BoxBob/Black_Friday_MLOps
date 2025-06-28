import pandas as pd
import boto3
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os

from .validation import DataValidator, ValidationResult
from .feature_engineering import FeatureEngineer
from ...data.schemas.black_friday_schema import DataSchema

class DataPipeline:
    """
    Main orchestrator for the data processing pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.logger = self._setup_logging()
        
        # Initialize components
        self.schema = DataSchema.get_black_friday_schema()
        self.validator = DataValidator(self.schema)
        self.feature_engineer = FeatureEngineer()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('data_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_pipeline(self, input_key: str, output_key: str) -> Dict[str, Any]:
        """
        Execute the complete data processing pipeline
        """
        try:
            # Step 1: Load data
            self.logger.info(f"Loading data from {input_key}")
            df = self._load_data_from_s3(input_key)
            
            # Step 2: Validate data
            self.logger.info("Validating data quality")
            validation_result = self.validator.validate_dataset(df)
            
            if not validation_result.is_valid:
                self.logger.error(f"Data validation failed: {validation_result.errors}")
                return {
                    'status': 'failed',
                    'reason': 'validation_failed',
                    'errors': validation_result.errors
                }
            
            # Step 3: Feature engineering
            self.logger.info("Performing feature engineering")
            df_features = self.feature_engineer.fit_transform(df)
            
            # Step 4: Save processed data
            self.logger.info(f"Saving processed data to {output_key}")
            self._save_data_to_s3(df_features, output_key)
            
            # Step 5: Save pipeline metadata
            metadata = {
                'pipeline_run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'input_key': input_key,
                'output_key': output_key,
                'validation_result': {
                    'is_valid': validation_result.is_valid,
                    'warnings': validation_result.warnings,
                    'metrics': validation_result.metrics
                },
                'feature_count': len(df_features.columns),
                'record_count': len(df_features),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            self._save_metadata(metadata, f"{output_key}_metadata.json")
            
            self.logger.info("Pipeline completed successfully")
            return {
                'status': 'success',
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'reason': 'processing_error',
                'error': str(e)
            }
    
    def _load_data_from_s3(self, s3_key: str) -> pd.DataFrame:
        """Load data from S3 bucket"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config['ingestion_bucket'],
                Key=s3_key
            )
            
            df = pd.read_csv(response['Body'])
            self.logger.info(f"Loaded {len(df)} records from {s3_key}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from S3: {str(e)}")
            raise
    
    def _save_data_to_s3(self, df: pd.DataFrame, s3_key: str) -> None:
        """Save processed data to S3"""
        try:
            csv_buffer = df.to_csv(index=False)
            
            self.s3_client.put_object(
                Bucket=self.config['processed_bucket'],
                Key=s3_key,
                Body=csv_buffer
            )
            
            self.logger.info(f"Saved {len(df)} records to {s3_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to S3: {str(e)}")
            raise
    
    def _save_metadata(self, metadata: Dict[str, Any], s3_key: str) -> None:
        """Save pipeline metadata to S3"""
        try:
            metadata_json = json.dumps(metadata, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.config['processed_bucket'],
                Key=s3_key,
                Body=metadata_json,
                ContentType='application/json'
            )
            
            self.logger.info(f"Saved metadata to {s3_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise
