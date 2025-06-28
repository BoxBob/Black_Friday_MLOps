#!/usr/bin/env python3
"""
Data Schema Definition for Black Friday Dataset

This module defines the expected data schema, validation rules,
and data quality checks for the Black Friday sales dataset.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_issue(self, message: str, severity: ValidationSeverity) -> None:
        """Add a validation issue."""
        if severity == ValidationSeverity.ERROR:
            self.errors.append(message)
            self.is_valid = False
        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(message)
        else:
            self.info.append(message)


@dataclass
class ColumnConstraint:
    """Define constraints for a single column."""
    name: str
    dtype: str
    nullable: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DataSchema:
    """
    Define expected data schema with validation capabilities.
    
    This class provides comprehensive data validation including:
    - Column presence and types
    - Value ranges and constraints
    - Categorical value validation
    - Data quality metrics
    """
    
    name: str
    version: str
    columns: List[ColumnConstraint]
    business_rules: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Create lookup dictionaries for efficient validation."""
        self._column_map = {col.name: col for col in self.columns}
        self._required_columns = [col.name for col in self.columns if not col.nullable]
    
    @property
    def required_columns(self) -> List[str]:
        """Get list of required column names."""
        return [col.name for col in self.columns]
    
    @property
    def column_types(self) -> Dict[str, str]:
        """Get column name to dtype mapping."""
        return {col.name: col.dtype for col in self.columns}
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against this schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(is_valid=True)
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            result.add_issue(
                f"Missing required columns: {sorted(missing_columns)}",
                ValidationSeverity.ERROR
            )
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.required_columns)
        if extra_columns:
            result.add_issue(
                f"Unexpected columns found: {sorted(extra_columns)}",
                ValidationSeverity.WARNING
            )
        
        # Validate each column
        for col in self.columns:
            if col.name not in df.columns:
                continue
                
            self._validate_column(df, col, result)
        
        # Business rule validation
        self._validate_business_rules(df, result)
        
        return result
    
    def _validate_column(self, df: pd.DataFrame, col: ColumnConstraint, result: ValidationResult) -> None:
        """Validate a single column."""
        series = df[col.name]
        
        # Check nulls
        null_count = series.isnull().sum()
        if null_count > 0 and not col.nullable:
            result.add_issue(
                f"Column '{col.name}' has {null_count} null values but nulls not allowed",
                ValidationSeverity.ERROR
            )
        
        # Check data type (for non-null values)
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            try:
                # Attempt type conversion to validate
                if col.dtype == 'int64':
                    pd.to_numeric(non_null_series, errors='raise', downcast='integer')
                elif col.dtype == 'float64':
                    pd.to_numeric(non_null_series, errors='raise')
                elif col.dtype == 'object':
                    # String validation
                    pass
            except (ValueError, TypeError) as e:
                result.add_issue(
                    f"Column '{col.name}' contains values incompatible with type '{col.dtype}'",
                    ValidationSeverity.ERROR
                )
        
        # Check value ranges
        if col.min_value is not None or col.max_value is not None:
            numeric_series = pd.to_numeric(non_null_series, errors='coerce')
            
            if col.min_value is not None:
                below_min = (numeric_series < col.min_value).sum()
                if below_min > 0:
                    result.add_issue(
                        f"Column '{col.name}' has {below_min} values below minimum {col.min_value}",
                        ValidationSeverity.ERROR
                    )
            
            if col.max_value is not None:
                above_max = (numeric_series > col.max_value).sum()
                if above_max > 0:
                    result.add_issue(
                        f"Column '{col.name}' has {above_max} values above maximum {col.max_value}",
                        ValidationSeverity.ERROR
                    )
        
        # Check allowed values
        if col.allowed_values is not None:
            invalid_values = set(non_null_series) - set(col.allowed_values)
            if invalid_values:
                result.add_issue(
                    f"Column '{col.name}' contains invalid values: {sorted(invalid_values)}",
                    ValidationSeverity.ERROR
                )
    
    def _validate_business_rules(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate business-specific rules."""
        # Example: Check for suspicious purchase patterns
        if 'Purchase' in df.columns and 'User_ID' in df.columns:
            # Check for unusually high purchases
            high_purchases = df[df['Purchase'] > 20000]
            if len(high_purchases) > 0:
                result.add_issue(
                    f"Found {len(high_purchases)} purchases above $20,000 - review for data quality",
                    ValidationSeverity.WARNING
                )
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data profiling information."""
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().mean() * 100,
                'unique_count': df[col].nunique(),
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            profile['columns'][col] = col_profile
        
        return profile

    @classmethod
    def get_black_friday_schema(cls) -> 'DataSchema':
        """Create Black Friday dataset schema."""
        columns = [
            ColumnConstraint(
                name='User_ID',
                dtype='int64',
                nullable=False,
                min_value=1000000,
                max_value=9999999,
                description='Unique identifier for users'
            ),
            ColumnConstraint(
                name='Product_ID',
                dtype='object',
                nullable=False,
                regex_pattern=r'^P\d+$',
                description='Product identifier (format: P followed by digits)'
            ),
            ColumnConstraint(
                name='Gender',
                dtype='object',
                nullable=False,
                allowed_values=['M', 'F'],
                description='Customer gender'
            ),
            ColumnConstraint(
                name='Age',
                dtype='object',
                nullable=False,
                allowed_values=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'],
                description='Age group of customer'
            ),
            ColumnConstraint(
                name='Occupation',
                dtype='int64',
                nullable=False,
                min_value=0,
                max_value=20,
                description='Occupation code (0-20)'
            ),
            ColumnConstraint(
                name='City_Category',
                dtype='object',
                nullable=False,
                allowed_values=['A', 'B', 'C'],
                description='City tier classification'
            ),
            ColumnConstraint(
                name='Stay_In_Current_City_Years',
                dtype='object',
                nullable=False,
                allowed_values=['0', '1', '2', '3', '4+'],
                description='Years in current city'
            ),
            ColumnConstraint(
                name='Marital_Status',
                dtype='int64',
                nullable=False,
                allowed_values=[0, 1],
                description='Marital status (0: Single, 1: Married)'
            ),
            ColumnConstraint(
                name='Product_Category_1',
                dtype='float64',
                nullable=True,  # Can be missing
                min_value=1,
                max_value=20,
                description='Primary product category'
            ),
            ColumnConstraint(
                name='Product_Category_2',
                dtype='float64',
                nullable=True,
                min_value=1,
                max_value=20,
                description='Secondary product category'
            ),
            ColumnConstraint(
                name='Product_Category_3',
                dtype='float64',
                nullable=True,
                min_value=1,
                max_value=20,
                description='Tertiary product category'
            ),
            ColumnConstraint(
                name='Purchase',
                dtype='int64',
                nullable=False,
                min_value=1,
                max_value=25000,
                description='Purchase amount in dollars'
            )
        ]
        
        business_rules = [
            "Purchase amount should be positive",
            "Product categories should be hierarchical (Cat1 > Cat2 > Cat3)",
            "User demographics should be consistent across purchases"
        ]
        
        return cls(
            name="black_friday_sales",
            version="1.0.0",
            columns=columns,
            business_rules=business_rules
        )


""" # Example usage
if __name__ == "__main__":
    # Create schema
    schema = DataSchema.get_black_friday_schema()
    
    # Example validation
    sample_data = pd.DataFrame({
        'User_ID': [1000001, 1000002],
        'Product_ID': ['P001', 'P002'],
        'Gender': ['M', 'F'],
        'Age': ['26-35', '18-25'],
        'Occupation': [4, 7],
        'City_Category': ['A', 'B'],
        'Stay_In_Current_City_Years': ['2', '1'],
        'Marital_Status': [0, 1],
        'Product_Category_1': [2.0, 3.0],
        'Product_Category_2': [np.nan, 5.0],
        'Product_Category_3': [np.nan, np.nan],
        'Purchase': [8370, 15200]
    })
    
    # Validate
    result = schema.validate_dataframe(sample_data)
    print(f"Validation passed: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Get data profile
    profile = schema.get_data_profile(sample_data)
    print(f"Data profile: {profile}") """