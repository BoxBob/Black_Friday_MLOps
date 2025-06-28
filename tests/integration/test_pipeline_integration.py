import unittest
import boto3
import pandas as pd
import os
import sys
from moto import mock_s3
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data_pipeline.pipeline import DataPipeline

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete data pipeline"""
    
    @mock_s3
    def setUp(self):
        """Setup test environment with mocked AWS services"""
        # Create mock S3 buckets
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        self.ingestion_bucket = 'test-ingestion-bucket'
        self.processed_bucket = 'test-processed-bucket'
        
        self.s3_client.create_bucket(Bucket=self.ingestion_bucket)
        self.s3_client.create_bucket(Bucket=self.processed_bucket)
        
        # Create test configuration
        self.config = {
            'ingestion_bucket': self.ingestion_bucket,
            'processed_bucket': self.processed_bucket,
            'aws_region': 'us-east-1'
        }
        
        # Create test data
        self.test_data = pd.DataFrame({
            'User_ID': [1001, 1002, 1003],
            'Product_ID': ['P001', 'P002', 'P003'],
            'Gender': ['M', 'F', 'M'],
            'Age': ['26-35', '18-25', '36-45'],
            'Occupation': [1, 2, 3],
            'City_Category': ['A', 'B', 'C'],
            'Stay_In_Current_City_Years': ['1', '2', '3'],
            'Marital_Status': [0, 1, 0],
            'Product_Category_1': [1.0, 2.0, 3.0],
            'Product_Category_2': [2.0, None, 4.0],
            'Product_Category_3': [None, None, 5.0],
            'Purchase': [1000, 2000, 1500]
        })
        
        # Upload test data to S3
        csv_buffer = self.test_data.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=self.ingestion_bucket,
            Key='test_data.csv',
            Body=csv_buffer
        )
    
    @mock_s3
    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution"""
        pipeline = DataPipeline(self.config)
        
        result = pipeline.run_pipeline('test_data.csv', 'processed_data.csv')
        
        # Check pipeline execution result
        self.assertEqual(result['status'], 'success')
        self.assertIn('metadata', result)
        
        # Verify processed data exists in S3
        response = self.s3_client.list_objects_v2(Bucket=self.processed_bucket)
        self.assertIn('Contents', response)
        
        # Check that processed data has more columns (features)
        processed_response = self.s3_client.get_object(
            Bucket=self.processed_bucket,
            Key='processed_data.csv'
        )
        processed_df = pd.read_csv(processed_response['Body'])
        self.assertGreater(len(processed_df.columns), len(self.test_data.columns))

if __name__ == '__main__':
    unittest.main()
