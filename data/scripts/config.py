import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class ValidationConfig:
    max_missing_percentage: int = 50
    max_outlier_percentage: int = 5
    required_columns_threshold: float = 0.9
    enable_drift_detection: bool = False
    quality_score_threshold: float = 0.8

@dataclass
class RetryConfig:
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    s3_operation_timeout: int = 300
    validation_timeout: int = 600

@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_namespace: str = "DataPipeline"
    cloudwatch_log_group: str = "/aws/lambda/data-pipeline"
    alert_on_validation_failure: bool = True
    processing_time_alert_threshold: int = 1800
    sns_topic_arn: Optional[str] = None

def get_pipeline_config() -> Dict[str, Any]:
    """
    Get comprehensive configuration for data pipeline
    """
    
    # Environment detection
    env = os.getenv('ENVIRONMENT', 'development')
    is_production = env.lower() == 'production'
    
    config = {
        # Core AWS Configuration
        'aws': {
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            'ingestion_bucket': os.getenv('DATA_INGESTION_BUCKET', 'black-friday-data-ingestion'),
            'processed_bucket': os.getenv('PROCESSED_DATA_BUCKET', 'black-friday-processed-data'),
            'kms_key_id': os.getenv('KMS_KEY_ID'),
            'iam_role_arn': os.getenv('IAM_ROLE_ARN'),
        },
        
        # Security Configuration
        'security': {
            'encrypt_at_rest': os.getenv('ENCRYPT_AT_REST', 'true').lower() == 'true',
            'encrypt_in_transit': True,
            'enable_access_logging': is_production,
            'data_classification': os.getenv('DATA_CLASSIFICATION', 'internal'),
        },
        
        # Logging Configuration
        'logging': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': 'json' if is_production else 'text',
            'enable_structured_logging': is_production,
            'log_sensitive_data': not is_production,
        },
        
        # Retry and Timeout Configuration
        'retry': {
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'retry_backoff_factor': float(os.getenv('RETRY_BACKOFF', '2.0')),
            's3_operation_timeout': int(os.getenv('S3_TIMEOUT', '300')),
            'validation_timeout': int(os.getenv('VALIDATION_TIMEOUT', '600')),
        },
        
        # Monitoring and Alerting
        'monitoring': {
            'enable_metrics': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'metrics_namespace': os.getenv('METRICS_NAMESPACE', 'DataPipeline'),
            'cloudwatch_log_group': os.getenv('CLOUDWATCH_LOG_GROUP', '/aws/lambda/data-pipeline'),
            'alert_on_validation_failure': True,
            'processing_time_alert_threshold': int(os.getenv('PROCESSING_TIME_ALERT', '1800')),
            'sns_topic_arn': os.getenv('SNS_TOPIC_ARN'),
            'enable_custom_metrics': is_production,
        },
        
        # Data Quality and Validation
        'validation_rules': {
            'max_missing_percentage': int(os.getenv('MAX_MISSING_PCT', '50')),
            'max_outlier_percentage': int(os.getenv('MAX_OUTLIER_PCT', '5')),
            'required_columns_threshold': float(os.getenv('REQUIRED_COLS_THRESHOLD', '0.9')),
            'enable_drift_detection': os.getenv('ENABLE_DRIFT_DETECTION', 'false').lower() == 'true',
            'quality_score_threshold': float(os.getenv('QUALITY_THRESHOLD', '0.8')),
            'baseline_data_path': os.getenv('BASELINE_DATA_PATH'),
        },
        
        # Feature Engineering Configuration
        'feature_engineering': {
            'create_interaction_features': os.getenv('CREATE_INTERACTIONS', 'true').lower() == 'true',
            'create_behavioral_features': os.getenv('CREATE_BEHAVIORAL', 'true').lower() == 'true',
            'create_seasonal_features': os.getenv('CREATE_SEASONAL', 'true').lower() == 'true',
            'normalize_features': os.getenv('NORMALIZE_FEATURES', 'true').lower() == 'true',
            'feature_selection_method': os.getenv('FEATURE_SELECTION', 'correlation'),
            'max_features': int(os.getenv('MAX_FEATURES', '1000')),
            'encoding': {
                'categorical_method': os.getenv('CATEGORICAL_ENCODING', 'onehot'),
                'handle_unknown': 'ignore',
                'max_categories': int(os.getenv('MAX_CATEGORIES', '50'))
            }
        },
        
        # Performance Configuration
        'performance': {
            'chunk_size': int(os.getenv('CHUNK_SIZE', '10000')),
            'parallel_processing': os.getenv('PARALLEL_PROCESSING', 'false').lower() == 'true',
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            'memory_limit_mb': int(os.getenv('MEMORY_LIMIT_MB', '4096')),
            'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
        },
        
        # Environment-Specific Settings
        'environment': {
            'name': env,
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'enable_sampling': os.getenv('ENABLE_SAMPLING', str(not is_production)).lower() == 'true',
            'sample_percentage': float(os.getenv('SAMPLE_PERCENTAGE', '0.1')),
            'is_production': is_production,
        },
        
        # Metadata and Lineage
        'metadata': {
            'enable_lineage_tracking': os.getenv('ENABLE_LINEAGE', 'true').lower() == 'true',
            'metadata_store': os.getenv('METADATA_STORE', 'dynamodb'),
            'lineage_table': os.getenv('LINEAGE_TABLE', 'data-pipeline-lineage'),
            'enable_versioning': True,
            'version_prefix': os.getenv('VERSION_PREFIX', 'v'),
        },
        
        # Data Governance
        'governance': {
            'enable_pii_detection': is_production,
            'enable_data_masking': is_production,
            'retention_policy_days': int(os.getenv('RETENTION_DAYS', '90')),
            'compliance_mode': os.getenv('COMPLIANCE_MODE', 'none'),  # gdpr, ccpa, none
        }
    }
    
    # Load additional config from file if specified
    config_file = os.getenv('PIPELINE_CONFIG_FILE')
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration settings"""
    required_keys = ['aws', 'validation_rules', 'feature_engineering']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate AWS configuration
    aws_config = config['aws']
    if not aws_config.get('ingestion_bucket') or not aws_config.get('processed_bucket'):
        raise ValueError("AWS bucket names must be specified")
    
    # Validate numeric ranges
    validation_rules = config['validation_rules']
    if not (0 <= validation_rules['max_missing_percentage'] <= 100):
        raise ValueError("max_missing_percentage must be between 0 and 100")
    
    if not (0 <= validation_rules['quality_score_threshold'] <= 1):
        raise ValueError("quality_score_threshold must be between 0 and 1")

# Usage example
if __name__ == "__main__":
    config = get_pipeline_config()
    validate_config(config)
    print("Configuration loaded and validated successfully")