import unittest
import boto3
import pandas as pd
import os
import sys
from moto import mock_aws
import json
from src.data_pipeline.pipeline import DataPipeline

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete data pipeline"""
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution"""
        with mock_aws():
            # Setup test environment with mocked AWS services
            s3_client = boto3.client('s3', region_name='ap-south-1')
            ingestion_bucket = 'test-ingestion-bucket'
            processed_bucket = 'test-processed-bucket'
            
            # Create mock S3 buckets
            s3_client.create_bucket(
                Bucket=ingestion_bucket,
                CreateBucketConfiguration={'LocationConstraint': 'ap-south-1'}
            )
            s3_client.create_bucket(
                Bucket=processed_bucket,
                CreateBucketConfiguration={'LocationConstraint': 'ap-south-1'}
            )
            
            # Create test configuration
            config = {
                'ingestion_bucket': ingestion_bucket,
                'processed_bucket': processed_bucket,
                'aws_region': 'ap-south-1'
            }
            
            # Create test data
            test_data = pd.DataFrame({
                'User_ID': [1000001, 1000002, 1000003, 1000004, 1000005],
                'Product_ID': ['P100001', 'P100002', 'P100003', 'P100004', 'P100005'],
                'Gender': ['M', 'F', 'M', 'F', 'M'],
                'Age': ['26-35', '18-25', '36-45', '46-50', '51-55'],
                'Occupation': [1, 2, 3, 4, 5],
                'City_Category': ['A', 'B', 'C', 'A', 'B'],
                'Stay_In_Current_City_Years': ['0', '1', '2', '3', '4+'],
                'Marital_Status': [0, 1, 0, 1, 0],
                'Product_Category_1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'Product_Category_2': [2.0, 5.0, 4.0, 3.0, 2.0],
                'Product_Category_3': [6.0, 7.0, 5.0, 4.0, 3.0],
                'Purchase': [1000, 2000, 1500, 1800, 2200]
            })
            
            # Upload test data to S3
            csv_buffer = test_data.to_csv(index=False)
            s3_client.put_object(
                Bucket=ingestion_bucket,
                Key='test_data.csv',
                Body=csv_buffer
            )
            
            # Run the pipeline
            pipeline = DataPipeline(config)
            result = pipeline.run_pipeline('test_data.csv', 'processed_data.csv')
            
            # Check pipeline execution result
            self.assertEqual(result['status'], 'success')
            self.assertIn('metadata', result)
            
            # Verify processed data exists in S3
            response = s3_client.list_objects_v2(Bucket=processed_bucket)
            self.assertIn('Contents', response)
            
            # Check that processed data has more columns (features)
            processed_response = s3_client.get_object(
                Bucket=processed_bucket,
                Key='processed_data.csv'
            )
            processed_df = pd.read_csv(processed_response['Body'])
            self.assertGreater(len(processed_df.columns), len(test_data.columns))

if __name__ == '__main__':
    unittest.main()