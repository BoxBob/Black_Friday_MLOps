import boto3
import pandas as pd
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
import requests
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration class for the data ingestion pipeline"""
    ingestion_bucket: str
    processed_bucket: str
    aws_region: str = 'ap-south-1'
    max_retries: int = 3
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.ingestion_bucket or not self.processed_bucket:
            raise ValueError("Both ingestion_bucket and processed_bucket must be provided")


class DataIngestionPipeline:
    """
    Automated data ingestion pipeline for Black Friday sales data
    
    This class handles ingesting data from various sources (local files, APIs)
    and storing them in S3 for further processing.
    """
    
    # Constants
    RAW_DATA_PREFIX = "raw_data"
    SUPPORTED_FILE_EXTENSIONS = {'.csv', '.json', '.xlsx', '.parquet'}
    MAX_FILE_SIZE_MB = 100
    
    def __init__(self, config: Union[Dict, PipelineConfig]):
        """
        Initialize the data ingestion pipeline
        
        Args:
            config: Configuration dictionary or PipelineConfig object
        """
        # Handle both dict and PipelineConfig
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config
            
        self.logger = self._setup_logging()
        self.s3_client = self._initialize_s3_client()
        self._validate_buckets()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the data pipeline"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_s3_client(self) -> boto3.client:
        """Initialize S3 client with proper error handling"""
        try:
            s3_client = boto3.client(
                's3',
                region_name=self.config.aws_region
            )
            # Test connection
            s3_client.list_buckets()
            return s3_client
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure your credentials.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _validate_buckets(self) -> None:
        """Validate that S3 buckets exist and are accessible"""
        for bucket_name in [self.config.ingestion_bucket, self.config.processed_bucket]:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                self.logger.info(f"Validated access to bucket: {bucket_name}")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    raise ValueError(f"Bucket {bucket_name} does not exist")
                elif error_code == '403':
                    raise ValueError(f"Access denied to bucket {bucket_name}")
                else:
                    raise ValueError(f"Error accessing bucket {bucket_name}: {str(e)}")
    
    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and sanitize file path"""
        path = Path(file_path).resolve()
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max: {self.MAX_FILE_SIZE_MB}MB)")
        
        return path
    
    def _generate_s3_key(self, file_name: str, prefix: str = None) -> str:
        """Generate a unique S3 key for the file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_prefix = prefix or self.RAW_DATA_PREFIX
        
        # Sanitize filename
        safe_filename = "".join(c for c in file_name if c.isalnum() or c in "._-")
        
        return f"{base_prefix}/{timestamp}_{safe_filename}"
    
    def ingest_raw_data(self, data_source: str, source_path: str, 
                       metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Ingest raw data from various sources into S3
        
        Args:
            data_source: Source type ('local' or 'api')
            source_path: File path for local files or endpoint for APIs
            metadata: Optional metadata to attach to the object
            
        Returns:
            S3 key of uploaded object or None if failed
        """
        try:
            if data_source == 'local':
                return self._upload_local_file(source_path, metadata)
            elif data_source == 'api':
                return self._fetch_from_api(source_path, metadata)
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
                
        except Exception as e:
            self.logger.error(f"Data ingestion failed for {source_path}: {str(e)}")
            return None
    
    def _upload_local_file(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Upload local file to S3 ingestion bucket"""
        validated_path = self._validate_file_path(file_path)
        s3_key = self._generate_s3_key(validated_path.name)
        
        # Prepare metadata
        s3_metadata = {
            'source': 'local',
            'original_filename': validated_path.name,
            'upload_timestamp': datetime.now().isoformat(),
            'file_size': str(validated_path.stat().st_size)
        }
        
        if metadata:
            s3_metadata.update(metadata)
        
        # Upload with retry logic
        for attempt in range(self.config.max_retries):
            try:
                self.s3_client.upload_file(
                    str(validated_path),
                    self.config.ingestion_bucket,
                    s3_key,
                    ExtraArgs={
                        'Metadata': s3_metadata,
                        'ServerSideEncryption': 'AES256'
                    }
                )
                
                self.logger.info(f"Successfully uploaded {validated_path.name} to {s3_key}")
                return s3_key
                
            except ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                self.logger.warning(f"Upload attempt {attempt + 1} failed, retrying...")
                
        raise Exception("Max retries exceeded")
    
    def _fetch_from_api(self, endpoint: str, metadata: Optional[Dict] = None) -> str:
        """Fetch data from API and store in S3"""
        try:
            # Validate URL
            if not endpoint.startswith(('http://', 'https://')):
                raise ValueError("Invalid API endpoint URL")
            
            # Fetch data with timeout
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            
            # Generate filename based on endpoint
            filename = f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            s3_key = self._generate_s3_key(filename)
            
            # Prepare metadata
            s3_metadata = {
                'source': 'api',
                'endpoint': endpoint,
                'fetch_timestamp': datetime.now().isoformat(),
                'content_type': response.headers.get('content-type', 'application/json')
            }
            
            if metadata:
                s3_metadata.update(metadata)
            
            # Upload response data
            self.s3_client.put_object(
                Bucket=self.config.ingestion_bucket,
                Key=s3_key,
                Body=response.content,
                Metadata=s3_metadata,
                ServerSideEncryption='AES256'
            )
            
            self.logger.info(f"Successfully fetched and stored API data to {s3_key}")
            return s3_key
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch from API: {str(e)}")
            raise
    
    def list_ingested_files(self, prefix: str = None) -> List[Dict[str, Union[str, int]]]:
        """
        List all files in the ingestion bucket with metadata
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of dictionaries containing file information
        """
        try:
            list_prefix = prefix or self.RAW_DATA_PREFIX
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.ingestion_bucket,
                Prefix=f"{list_prefix}/"
            )
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag'].strip('"')
                        })
            
            self.logger.info(f"Found {len(files)} files in ingestion bucket")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {str(e)}")
            return []
    
    def get_file_metadata(self, s3_key: str) -> Optional[Dict]:
        """Get metadata for a specific file in S3"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.config.ingestion_bucket,
                Key=s3_key
            )
            
            return {
                'metadata': response.get('Metadata', {}),
                'content_type': response.get('ContentType'),
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self.logger.warning(f"File not found: {s3_key}")
            else:
                self.logger.error(f"Error getting metadata for {s3_key}: {str(e)}")
            return None
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on the pipeline components"""
        health_status = {
            's3_connection': False,
            'ingestion_bucket_access': False,
            'processed_bucket_access': False
        }
        
        try:
            # Test S3 connection
            self.s3_client.list_buckets()
            health_status['s3_connection'] = True
            
            # Test bucket access
            for bucket_key, bucket_name in [
                ('ingestion_bucket_access', self.config.ingestion_bucket),
                ('processed_bucket_access', self.config.processed_bucket)
            ]:
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    health_status[bucket_key] = True
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
        
        return health_status


# Example usage
""" if __name__ == "__main__":
    # Configuration
    config = PipelineConfig(
        ingestion_bucket="my-ingestion-bucket",
        processed_bucket="my-processed-bucket",
        aws_region="us-west-2",
        log_level="INFO"
    )
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(config)
    
    # Health check
    health = pipeline.health_check()
    print(f"Pipeline health: {health}")
    
    # Upload a local file
    # s3_key = pipeline.ingest_raw_data(
    #     data_source='local',
    #     source_path='/path/to/your/file.csv',
    #     metadata={'department': 'sales', 'campaign': 'black_friday_2024'}
    # )
    
    # List ingested files
    files = pipeline.list_ingested_files()
    print(f"Ingested files: {len(files)}") """