#!/usr/bin/env python3
"""
Enhanced Data Validation Pipeline - with Schema Integration

This module integrates the schema's validation method with the pipeline
validators to ensure comprehensive validation coverage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import traceback
from abc import ABC, abstractmethod


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Pipeline should stop
    ERROR = "error"       # Data issues that need attention
    WARNING = "warning"   # Issues to monitor
    INFO = "info"         # Informational messages


class ValidationCategory(Enum):
    """Categories of validation checks."""
    SCHEMA = "schema"
    DATA_QUALITY = "data_quality"
    STATISTICAL = "statistical"
    BUSINESS_RULES = "business_rules"
    PERFORMANCE = "performance"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_count: Optional[int] = None
    percentage: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    total_rows: int
    total_columns: int
    missing_values_count: int
    missing_values_percentage: float
    duplicate_rows: int
    unique_users: int
    unique_products: int
    memory_usage_mb: float
    validation_duration_seconds: float
    data_freshness_hours: Optional[float] = None
    
    # Business-specific metrics
    purchase_statistics: Dict[str, float] = field(default_factory=dict)
    categorical_distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'missing_values_count': self.missing_values_count,
            'missing_values_percentage': round(self.missing_values_percentage, 2),
            'duplicate_rows': self.duplicate_rows,
            'unique_users': self.unique_users,
            'unique_products': self.unique_products,
            'memory_usage_mb': round(self.memory_usage_mb, 2),
            'validation_duration_seconds': round(self.validation_duration_seconds, 3),
            'data_freshness_hours': self.data_freshness_hours,
            'purchase_statistics': self.purchase_statistics,
            'categorical_distributions': self.categorical_distributions
        }


@dataclass
class ValidationResult:
    """Comprehensive validation results."""
    is_valid: bool
    is_critical: bool
    issues: List[ValidationIssue]
    metrics: ValidationMetrics
    summary: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            'total_issues': len(self.issues),
            'critical_count': sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL),
            'error_count': sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
            'warning_count': sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
            'info_count': sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)
        }
        self.is_critical = self.summary['critical_count'] > 0
        # Update is_valid based on ERROR and CRITICAL issues
        self.is_valid = self.summary['critical_count'] == 0 and self.summary['error_count'] == 0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'is_critical': self.is_critical,
            'summary': self.summary,
            'metrics': self.metrics.to_dict(),
            'issues': [
                {
                    'category': issue.category.value,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'column': issue.column,
                    'row_count': issue.row_count,
                    'percentage': round(issue.percentage, 2) if issue.percentage else None,
                    'details': issue.details,
                    'timestamp': issue.timestamp.isoformat()
                }
                for issue in self.issues
            ]
        }


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Perform validation and return issues."""
        pass


class SchemaValidator(BaseValidator):
    """Enhanced schema validator that uses the schema's own validation method."""
    
    def __init__(self, schema):
        self.schema = schema
        self.logger = logging.getLogger(f'{__name__}.SchemaValidator')
    
    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate using the schema's built-in validation method."""
        issues = []
        
        try:
            # Use the schema's own validate_dataframe method
            schema_result = self.schema.validate_dataframe(df)
            
            # Convert schema validation results to our ValidationIssue format
            # Schema errors become ERROR severity
            for error_msg in schema_result.errors:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    message=error_msg,
                    details={'source': 'schema_validation'}
                ))
            
            # Schema warnings remain as warnings
            for warning_msg in schema_result.warnings:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    message=warning_msg,
                    details={'source': 'schema_validation'}
                ))
                
            # Schema info messages
            for info_msg in schema_result.info:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.INFO,
                    message=info_msg,
                    details={'source': 'schema_validation'}
                ))
                
        except Exception as e:
            self.logger.error(f"Error in schema validation: {str(e)}")
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                message=f"Schema validation failed: {str(e)}",
                details={'exception': str(e)}
            ))
        
        return issues


class DataQualityValidator(BaseValidator):
    """Enhanced data quality validator."""
    
    def __init__(self, schema, quality_thresholds: Optional[Dict[str, float]] = None):
        self.schema = schema
        self.quality_thresholds = quality_thresholds or {
            'missing_critical_threshold': 50.0,
            'missing_warning_threshold': 10.0,
            'outlier_warning_threshold': 5.0,
            'duplicate_warning_threshold': 1.0
        }
        self.logger = logging.getLogger(f'{__name__}.DataQualityValidator')
    
    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data quality."""
        issues = []
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        if duplicate_pct > self.quality_thresholds['duplicate_warning_threshold']:
            severity = ValidationSeverity.ERROR if duplicate_pct > 5.0 else ValidationSeverity.WARNING
            issues.append(ValidationIssue(
                category=ValidationCategory.DATA_QUALITY,
                severity=severity,
                message=f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                row_count=duplicate_count,
                percentage=duplicate_pct
            ))
        
        # Check missing values per column with schema context
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            # Get column info from schema if available
            col_constraint = None
            if hasattr(self.schema, '_column_map') and col in self.schema._column_map:
                col_constraint = self.schema._column_map[col]
            elif hasattr(self.schema, 'columns'):
                col_constraint = next((c for c in self.schema.columns if c.name == col), None)
            
            # Adjust thresholds based on whether column is nullable
            is_nullable = col_constraint.nullable if col_constraint else True
            
            if not is_nullable and missing_count > 0:
                # Non-nullable column with missing values is always an error
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' is not nullable but has {missing_count} missing values ({missing_pct:.1f}%)",
                    column=col,
                    row_count=missing_count,
                    percentage=missing_pct
                ))
            elif missing_pct > self.quality_thresholds['missing_critical_threshold']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' has {missing_pct:.1f}% missing values (critical level)",
                    column=col,
                    row_count=missing_count,
                    percentage=missing_pct
                ))
            elif missing_pct > self.quality_thresholds['missing_warning_threshold']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' has {missing_pct:.1f}% missing values",
                    column=col,
                    row_count=missing_count,
                    percentage=missing_pct
                ))
        
        return issues


class StatisticalValidator(BaseValidator):
    """Validates statistical properties of the data."""
    
    def __init__(self, outlier_threshold: float = 1.5):
        self.outlier_threshold = outlier_threshold
        self.logger = logging.getLogger(f'{__name__}.StatisticalValidator')
    
    def validate(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate statistical properties."""
        issues = []
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'Purchase':  # Special handling for purchase amounts
                outlier_info = self._detect_outliers_iqr(df[col])
                if outlier_info['count'] > 0:
                    outlier_pct = outlier_info['percentage']
                    severity = ValidationSeverity.WARNING if outlier_pct < 5 else ValidationSeverity.ERROR
                    
                    issues.append(ValidationIssue(
                        category=ValidationCategory.STATISTICAL,
                        severity=severity,
                        message=f"Column '{col}' has {outlier_info['count']} outliers ({outlier_pct:.1f}%)",
                        column=col,
                        row_count=outlier_info['count'],
                        percentage=outlier_pct,
                        details=outlier_info
                    ))
        
        # Check for unusual distributions
        if 'Purchase' in df.columns:
            purchase_stats = self._analyze_distribution(df['Purchase'])
            
            # Check for potential data entry errors (e.g., all values the same)
            if purchase_stats['std'] == 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.ERROR,
                    message="Purchase column has zero variance - all values are identical",
                    column='Purchase',
                    details=purchase_stats
                ))
            
            # Check for unrealistic mean/std ratios
            cv = purchase_stats['std'] / purchase_stats['mean'] if purchase_stats['mean'] > 0 else float('inf')
            if cv > 2.0:  # Coefficient of variation > 200%
                issues.append(ValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.WARNING,
                    message=f"Purchase data shows high variability (CV: {cv:.2f})",
                    column='Purchase',
                    details={'coefficient_of_variation': cv, **purchase_stats}
                ))
        
        return issues
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers_mask.sum()
        
        return {
            'count': int(outlier_count),
            'percentage': float((outlier_count / len(series)) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR)
        }
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, float]:
        """Analyze distribution statistics."""
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis())
        }


class DataValidator:
    """
    Enhanced enterprise-grade data validation pipeline.
    
    Now properly integrates with the schema's validation method
    to ensure comprehensive validation coverage.
    """
    
    def __init__(self, schema, config: Optional[Dict[str, Any]] = None):
        self.schema = schema
        self.config = config or {}
        self.logger = logging.getLogger('DataValidator')
        
        # Initialize validators - SchemaValidator now uses schema's own validation
        self.validators = [
            SchemaValidator(schema),  # This will now call schema.validate_dataframe()
            DataQualityValidator(schema, self.config.get('quality_thresholds')),
            StatisticalValidator(self.config.get('outlier_threshold', 1.5))
        ]
    
    def validate_dataset(self, df: pd.DataFrame, 
                        data_source: Optional[str] = None) -> ValidationResult:
        """
        Perform comprehensive validation on the dataset.
        
        Args:
            df: DataFrame to validate
            data_source: Optional source identifier for logging
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting validation for dataset with {len(df)} rows, {len(df.columns)} columns")
            if data_source:
                self.logger.info(f"Data source: {data_source}")
            
            # Collect all validation issues
            all_issues = []
            
            for validator in self.validators:
                try:
                    issues = validator.validate(df)
                    all_issues.extend(issues)
                    self.logger.debug(f"{validator.__class__.__name__} found {len(issues)} issues")
                except Exception as e:
                    self.logger.error(f"Validation error in {validator.__class__.__name__}: {str(e)}")
                    all_issues.append(ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validator {validator.__class__.__name__} failed: {str(e)}",
                        details={'exception': str(e), 'traceback': traceback.format_exc()}
                    ))
            
            # Calculate metrics
            end_time = datetime.now(timezone.utc)
            metrics = self._calculate_metrics(df, (end_time - start_time).total_seconds())
            
            # Create result - ValidationResult.__post_init__ will set is_valid correctly
            result = ValidationResult(
                is_valid=True,  # Will be updated in __post_init__ based on issues
                is_critical=False,  # Will be updated in __post_init__
                issues=all_issues,
                metrics=metrics
            )
            
            self.logger.info(f"Validation completed: {result.summary}")
            return result
            
        except Exception as e:
            self.logger.error(f"Critical validation failure: {str(e)}", exc_info=True)
            
            # Return a critical failure result
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=False,
                is_critical=True,
                issues=[ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Critical validation failure: {str(e)}",
                    details={'exception': str(e), 'traceback': traceback.format_exc()}
                )],
                metrics=ValidationMetrics(
                    total_rows=len(df) if df is not None else 0,
                    total_columns=len(df.columns) if df is not None else 0,
                    missing_values_count=0,
                    missing_values_percentage=0.0,
                    duplicate_rows=0,
                    unique_users=0,
                    unique_products=0,
                    memory_usage_mb=0.0,
                    validation_duration_seconds=duration
                )
            )
    
    def _calculate_metrics(self, df: pd.DataFrame, duration: float) -> ValidationMetrics:
        """Calculate comprehensive data metrics."""
        missing_count = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0.0
        
        # Memory usage in MB
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Business-specific metrics
        purchase_stats = {}
        if 'Purchase' in df.columns:
            purchase_stats = {
                'mean': float(df['Purchase'].mean()),
                'std': float(df['Purchase'].std()),
                'min': float(df['Purchase'].min()),
                'max': float(df['Purchase'].max()),
                'median': float(df['Purchase'].median())
            }
        
        # Categorical distributions
        categorical_distributions = {}
        categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
        for col in categorical_cols:
            if col in df.columns:
                categorical_distributions[col] = df[col].value_counts().to_dict()
        
        return ValidationMetrics(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values_count=int(missing_count),
            missing_values_percentage=missing_pct,
            duplicate_rows=int(df.duplicated().sum()),
            unique_users=int(df['User_ID'].nunique()) if 'User_ID' in df.columns else 0,
            unique_products=int(df['Product_ID'].nunique()) if 'Product_ID' in df.columns else 0,
            memory_usage_mb=memory_usage_mb,
            validation_duration_seconds=duration,
            purchase_statistics=purchase_stats,
            categorical_distributions=categorical_distributions
        )
    
    def save_validation_report(self, result: ValidationResult, 
                             output_path: Union[str, Path]) -> None:
        """Save validation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {output_path}")


# Example usage demonstrating the fix
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import your actual schema
    # from your_schema_module import DataSchema
    
    # For demonstration, create a mock schema with the same interface
    class MockSchema:
        def __init__(self):
            self.required_columns = ['User_ID', 'Product_ID', 'Gender', 'Purchase']
            self.column_types = {
                'User_ID': 'int64', 
                'Product_ID': 'object', 
                'Gender': 'object', 
                'Purchase': 'int64'
            }
            self.columns = []
            
        def validate_dataframe(self, df):
            # Mock validation result that mimics your schema's behavior
            from types import SimpleNamespace
            result = SimpleNamespace(errors=[], warnings=[], info=[])
            
            # Check for invalid categorical values
            if 'Gender' in df.columns:
                invalid_genders = set(df['Gender'].dropna()) - {'M', 'F'}
                if invalid_genders:
                    result.errors.append(f"Column 'Gender' contains invalid values: {list(invalid_genders)}")
            
            # Check for out-of-range values
            if 'Purchase' in df.columns:
                out_of_range = ((df['Purchase'] < 1) | (df['Purchase'] > 25000)).sum()
                if out_of_range > 0:
                    result.errors.append(f"Column 'Purchase' has {out_of_range} values out of range [1, 25000]")
            
            return result
    
    # Create validator with mock schema
    schema = MockSchema()
    validator = DataValidator(schema)
    
    # Test data with issues that should make is_valid = False
    test_data = pd.DataFrame({
        'User_ID': [1000001, 1000002, 1000001],
        'Product_ID': ['P001', 'P002', 'P001'],
        'Gender': ['M', 'X', 'M'],  # Invalid gender 'X'
        'Purchase': [8370, 25001, 8370]  # Value 25001 is out of range
    })
    
    # Validate
    result = validator.validate_dataset(test_data, data_source="test_data")
    
    # Print results
    print(f"Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Critical Issues: {result.is_critical}")
    print(f"Summary: {result.summary}")
    print()
    
    for issue in result.issues:
        print(f"[{issue.severity.value.upper()}] {issue.category.value}: {issue.message}")
    
    # This should now show is_valid = False due to the ERROR issues