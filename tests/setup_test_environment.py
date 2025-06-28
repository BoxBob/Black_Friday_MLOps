import os
import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging

def setup_test_environment():
    """Setup local testing environment for data pipeline"""
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test AWS connectivity
    try:
        # Test S3 connection
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        logger.info("✅ AWS S3 connection successful")
        
        # Test CloudWatch connection
        cloudwatch = boto3.client('cloudwatch')
        metrics = cloudwatch.list_metrics(Namespace='AWS/EC2')
        logger.info("✅ AWS CloudWatch connection successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AWS connection failed: {str(e)}")
        return False

def create_test_data():
    """Create synthetic test data for pipeline testing"""
    
    np.random.seed(42)
    
    # Generate synthetic Black Friday data
    n_records = 1000
    
    test_data = {
        'User_ID': np.random.randint(1000000, 1010000, n_records),
        'Product_ID': [f'P00{np.random.randint(100000, 400000)}' for _ in range(n_records)],
        'Gender': np.random.choice(['M', 'F'], n_records),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], n_records),
        'Occupation': np.random.randint(0, 21, n_records),
        'City_Category': np.random.choice(['A', 'B', 'C'], n_records),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], n_records),
        'Marital_Status': np.random.choice([0, 1], n_records),
        'Product_Category_1': np.random.randint(1, 21, n_records),
        'Product_Category_2': np.random.choice([np.nan] + list(range(2, 18)), n_records),
        'Product_Category_3': np.random.choice([np.nan] + list(range(3, 19)), n_records),
        'Purchase': np.random.randint(185, 23961, n_records)
    }
    
    df = pd.DataFrame(test_data)
    
    # Save test data
    os.makedirs('data/test', exist_ok=True)
    df.to_csv('data/test/test_black_friday_data.csv', index=False)
    
    print(f"✅ Created test dataset with {len(df)} records")

    try:
        s3_client = boto3.client('s3')
        bucket_name = os.getenv('TEST_S3_BUCKET') or 'my-ingestion-bucket'
        s3_key = 'test/test_black_friday_data.csv'
        s3_client.upload_file('data/test/test_black_friday_data.csv', bucket_name, s3_key)
        print(f"✅ Uploaded test dataset to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"⚠️  Failed to upload test dataset to S3: {e}")
    return df

if __name__ == "__main__":
    print("Setting up test environment...")
    
    # Test AWS connectivity
    aws_connected = setup_test_environment()
    
    if aws_connected:
        # Create test data
        test_df = create_test_data()
        print("🎉 Test environment setup complete!")
    else:
        print("❌ Please check your AWS credentials and try again")
