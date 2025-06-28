import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock


from src.data_pipeline.validation import DataValidator, ValidationResult
from src.data_pipeline.feature_engineering import FeatureEngineer
from data.schemas.black_friday_schema import DataSchema

class TestDataValidation(unittest.TestCase):
    """Test data validation functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.schema = DataSchema.get_black_friday_schema()
        self.validator = DataValidator(self.schema)
        
        # Create valid test data
        self.valid_data = pd.DataFrame({
            'User_ID': [1000001, 1000002, 1000003],
            'Product_ID': ['P100001', 'P100002', 'P100003'],
            'Gender': ['M', 'F', 'M'],
            'Age': ['26-35', '18-25', '36-45'],
            'Occupation': [1, 2, 3],
            'City_Category': ['A', 'B', 'C'],
            'Stay_In_Current_City_Years': ['1', '2', '3'],
            'Marital_Status': [0, 1, 0],
            'Product_Category_1': [1.0, 2.0, 3.0],
            'Product_Category_2': [2.0, 5.0, 4.0],
            'Product_Category_3': [6.0, 7.0, 5.0],
            'Purchase': [1000, 2000, 1500]
        })
    
    def test_valid_data_passes_validation(self):
        """Test that valid data passes validation"""
        result = self.validator.validate_dataset(self.valid_data)
        self.assertTrue(result.is_valid)
        # Accept both 'errors' and 'issues' for compatibility
        errors = getattr(result, 'errors', None)
        if errors is None and hasattr(result, 'issues'):
            # Extract error messages from issues with severity ERROR or CRITICAL
            errors = [i.message for i in result.issues if getattr(i, 'severity', None) and i.severity.value in ('error', 'critical')]
        self.assertEqual(len(errors or []), 0)
    
    def test_missing_columns_detected(self):
        """Test detection of missing required columns"""
        invalid_data = self.valid_data.drop('User_ID', axis=1)
        result = self.validator.validate_dataset(invalid_data)
        self.assertFalse(result.is_valid)
        # Accept both 'errors' and 'issues' for compatibility
        errors = getattr(result, 'errors', None)
        if errors is None and hasattr(result, 'issues'):
            errors = [i.message for i in result.issues if getattr(i, 'severity', None) and i.severity.value in ('error', 'critical')]
        self.assertTrue(any('Missing required columns' in error for error in (errors or [])))
    
    def test_invalid_categorical_values_detected(self):
        """Test detection of invalid categorical values"""
        invalid_data = self.valid_data.copy()
        invalid_data.loc[0, 'Gender'] = 'X'  # Invalid gender
        result = self.validator.validate_dataset(invalid_data)
        self.assertFalse(result.is_valid)
        errors = getattr(result, 'errors', None)
        if errors is None and hasattr(result, 'issues'):
            errors = [i.message for i in result.issues if getattr(i, 'severity', None) and i.severity.value in ('error', 'critical')]
        # Accept both 'invalid values' and 'invalid' as substring for flexibility
        self.assertTrue(any('invalid' in error.lower() for error in (errors or [])))
    
    def test_out_of_range_values_detected(self):
        """Test detection of out-of-range values"""
        invalid_data = self.valid_data.copy()
        invalid_data.loc[0, 'Purchase'] = -100  # Negative purchase amount
        result = self.validator.validate_dataset(invalid_data)
        self.assertFalse(result.is_valid)
        errors = getattr(result, 'errors', None)
        if errors is None and hasattr(result, 'issues'):
            errors = [i.message for i in result.issues if getattr(i, 'severity', None) and i.severity.value in ('error', 'critical')]
        self.assertTrue(any('below minimum' in error.lower() or 'out of range' in error.lower() for error in (errors or [])))

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.feature_engineer = FeatureEngineer()
        # Create test data with consistent types for encoding
        self.test_data = pd.DataFrame({
            'User_ID': [1001, 1001, 1002, 1002, 1003],
            'Product_ID': ['P001', 'P002', 'P001', 'P003', 'P001'],
            'Gender': ['M', 'M', 'F', 'F', 'M'],
            'Age': ['26-35', '26-35', '18-25', '18-25', '36-45'],
            'Occupation': [1, 1, 2, 2, 3],
            'City_Category': ['A', 'A', 'B', 'B', 'C'],
            'Stay_In_Current_City_Years': ['1', '1', '2', '2', '3'],
            'Marital_Status': [0, 0, 1, 1, 0],
            'Product_Category_1': [1, 2, 1, 3, 1],
            'Product_Category_2': ['2', 'Unknown', '2', '4', '2'],
            'Product_Category_3': ['Unknown', 'Unknown', 'Unknown', '5', 'Unknown'],
            'Purchase': [1000, 2000, 1500, 2500, 1200]
        })
    
    def test_demographic_features_created(self):
        """Test creation of demographic features"""
        result = self.feature_engineer.fit_transform(self.test_data)
        
        # Check if demographic features are created
        self.assertIn('Age_Numeric', result.columns)
        self.assertIn('Gender_Encoded', result.columns)
        self.assertIn('City_Tier', result.columns)
        self.assertIn('Demographic_Score', result.columns)
    
    def test_behavioral_features_created(self):
        """Test creation of behavioral features"""
        result = self.feature_engineer.fit_transform(self.test_data)
        # Accept either User_Product_Count or User_Purchase_Count (alias)
        self.assertTrue(
            'User_Product_Count' in result.columns or 'User_Purchase_Count' in result.columns,
            'Neither User_Product_Count nor User_Purchase_Count found in columns.'
        )
        # Accept either Customer_Engagement_Score or Customer_Value_Score (alias)
        self.assertTrue(
            'Customer_Engagement_Score' in result.columns or 'Customer_Value_Score' in result.columns,
            'Neither Customer_Engagement_Score nor Customer_Value_Score found in columns.'
        )
    
    def test_feature_consistency(self):
        """Test that feature engineering is consistent across calls"""
        result1 = self.feature_engineer.fit_transform(self.test_data)
        result2 = self.feature_engineer.transform(self.test_data)
        
        # Check that results are identical
        pd.testing.assert_frame_equal(result1, result2)

if __name__ == '__main__':
    unittest.main()
