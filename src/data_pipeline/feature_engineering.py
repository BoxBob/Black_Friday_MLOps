import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:
    """
    Advanced feature engineering for Black Friday sales prediction
    """
    
    def __init__(self):
        self.logger = logging.getLogger('feature_engineer')
        self.encoders = {}
        self.scalers = {}
        self.fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data"""
        self.fitted = True
        return self._create_features(df, fit=True)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        return self._create_features(df, fit=False)
    
    def _create_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df_features = df.copy()
        
        # Handle missing values first
        df_features = self._handle_missing_values(df_features)
        
        # Demographic features
        df_features = self._create_demographic_features(df_features, fit)
        
        # Behavioral features
        df_features = self._create_behavioral_features(df_features, fit)
        
        # Product features
        df_features = self._create_product_features(df_features, fit)
        
        # Interaction features
        df_features = self._create_interaction_features(df_features, fit)
        
        # Seasonal features
        df_features = self._create_seasonal_features(df_features, fit)
        
        # Scale numerical features
        df_features = self._scale_features(df_features, fit)
        
        return df_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Handle missing values in categorical product categories
        df['Product_Category_2'] = df['Product_Category_2'].fillna('Unknown')
        df['Product_Category_3'] = df['Product_Category_3'].fillna('Unknown')
        
        # Handle any other missing values that might exist
        df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].fillna('0')
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create demographic-based features"""
        
        # Consistent encoding using LabelEncoder for all categorical variables
        categorical_features = ['Age', 'Gender', 'City_Category', 'Stay_In_Current_City_Years']
        
        for feature in categorical_features:
            if fit:
                self.encoders[feature] = LabelEncoder()
                df[f'{feature}_Encoded'] = self.encoders[feature].fit_transform(df[feature])
            else:
                df[f'{feature}_Encoded'] = self.encoders[feature].transform(df[feature])
        
        # Create meaningful mappings for interpretable features
        age_order_mapping = {
            '0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3,
            '46-50': 4, '51-55': 5, '55+': 6
        }
        df['Age_Numeric'] = df['Age'].map(age_order_mapping)
        
        city_tier_mapping = {'A': 2, 'B': 1, 'C': 0}  # A is tier-1, C is tier-3
        df['City_Tier'] = df['City_Category'].map(city_tier_mapping)
        
        stay_duration_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
        df['Stay_Duration_Numeric'] = df['Stay_In_Current_City_Years'].map(stay_duration_mapping)
        
        # Create demographic score (composite feature)
        df['Demographic_Score'] = (
            df['Age_Numeric'] * 0.3 + 
            df['City_Tier'] * 0.4 + 
            df['Stay_Duration_Numeric'] * 0.3
        )
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create behavioral features based on user patterns (without target leakage)"""
        
        # User-level aggregations WITHOUT using the target variable
        user_stats = df.groupby('User_ID').agg({
            'Product_ID': ['count', 'nunique'],  # Number of purchases and unique products
            'Product_Category_1': 'nunique',     # Category diversity
            'Occupation': 'first',               # User's occupation
            'Marital_Status': 'first'            # User's marital status
        }).round(2)
        
        user_stats.columns = [
            'User_Product_Count', 'User_Unique_Products', 
            'User_Category_Diversity', 'User_Occupation', 'User_Marital_Status'
        ]
        
        # Merge back to main dataframe
        df = df.merge(user_stats, left_on='User_ID', right_index=True, how='left')
        
        # Customer engagement score (without using target)
        df['Customer_Engagement_Score'] = (
            df['User_Product_Count'] * 0.4 + 
            df['User_Unique_Products'] * 0.3 + 
            df['User_Category_Diversity'] * 0.3
        )
        
        # Purchase frequency categories
        df['Purchase_Frequency_Category'] = pd.cut(
            df['User_Product_Count'], 
            bins=[0, 1, 3, 5, float('inf')], 
            labels=[0, 1, 2, 3]  # Use numeric labels for consistency
        )
        
        return df
    
    def _create_product_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create product-based features (without target leakage)"""
        
        # Product popularity based on interaction counts, not purchase amounts
        product_stats = df.groupby('Product_ID').agg({
            'User_ID': ['count', 'nunique'],     # Total interactions and unique users
            'Product_Category_1': 'first'        # Product's primary category
        }).round(2)
        
        product_stats.columns = [
            'Product_Interaction_Count', 'Product_Unique_Users', 'Product_Primary_Category'
        ]
        
        df = df.merge(product_stats, left_on='Product_ID', right_index=True, how='left')
        
        # Product category features
        df['Product_Category_Count'] = (
            df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]
            .notna().sum(axis=1)
        )
        
        # Encode product categories consistently
        product_categories = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']
        for cat in product_categories:
            if fit:
                self.encoders[cat] = LabelEncoder()
                df[f'{cat}_Encoded'] = self.encoders[cat].fit_transform(df[cat])
            else:
                df[f'{cat}_Encoded'] = self.encoders[cat].transform(df[cat])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create interaction features between different variables"""
        
        # Age-Gender interaction
        df['Age_Gender_Interaction'] = df['Age_Encoded'] * df['Gender_Encoded']
        
        # City-Marital interaction
        df['City_Marital_Interaction'] = df['City_Tier'] * df['Marital_Status']
        
        # Occupation-Age interaction
        df['Occupation_Age_Interaction'] = df['Occupation'] * df['Age_Numeric']
        
        # Product category interactions
        df['Category_1_2_Interaction'] = df['Product_Category_1_Encoded'] * df['Product_Category_2_Encoded']
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create seasonal and contextual features"""
        
        # Black Friday indicator (contextual feature)
        df['Is_Black_Friday'] = 1  # All data is from Black Friday
        
        # Create synthetic time-based features for demonstration
        # In production, these would come from actual timestamp data
        np.random.seed(42)  # For reproducibility in demo
        df['Hour_of_Day'] = np.random.randint(9, 22, size=len(df))
        df['Is_Peak_Hour'] = ((df['Hour_of_Day'] >= 18) & (df['Hour_of_Day'] <= 21)).astype(int)
        df['Is_Weekend'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        # Day part categories
        df['Day_Part'] = pd.cut(
            df['Hour_of_Day'], 
            bins=[0, 12, 17, 21, 24], 
            labels=[0, 1, 2, 3],  # Morning, Afternoon, Evening, Night
            include_lowest=True
        )
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Scale numerical features"""
        
        # Define numerical features to scale
        numerical_features = [
            'Age_Numeric', 'Stay_Duration_Numeric', 'Demographic_Score',
            'User_Product_Count', 'User_Unique_Products', 'User_Category_Diversity',
            'Customer_Engagement_Score', 'Product_Interaction_Count', 
            'Product_Unique_Users', 'Product_Category_Count', 'Hour_of_Day',
            'Age_Gender_Interaction', 'City_Marital_Interaction', 
            'Occupation_Age_Interaction', 'Category_1_2_Interaction'
        ]
        
        # Only scale features that exist in the dataframe
        features_to_scale = [f for f in numerical_features if f in df.columns]
        
        if fit:
            self.scalers['numerical'] = StandardScaler()
            df[features_to_scale] = self.scalers['numerical'].fit_transform(df[features_to_scale])
        else:
            df[features_to_scale] = self.scalers['numerical'].transform(df[features_to_scale])
        
        return df
    
    def get_feature_importance_names(self) -> List[str]:
        """Return list of all created feature names"""
        return [
            # Demographic features
            'Age_Encoded', 'Gender_Encoded', 'City_Category_Encoded', 
            'Stay_In_Current_City_Years_Encoded', 'Age_Numeric', 'City_Tier', 
            'Stay_Duration_Numeric', 'Demographic_Score',
            
            # Behavioral features
            'User_Product_Count', 'User_Unique_Products', 'User_Category_Diversity',
            'Customer_Engagement_Score', 'Purchase_Frequency_Category',
            
            # Product features
            'Product_Interaction_Count', 'Product_Unique_Users', 
            'Product_Category_1_Encoded', 'Product_Category_2_Encoded', 
            'Product_Category_3_Encoded', 'Product_Category_Count',
            
            # Interaction features
            'Age_Gender_Interaction', 'City_Marital_Interaction',
            'Occupation_Age_Interaction', 'Category_1_2_Interaction',
            
            # Seasonal features
            'Is_Black_Friday', 'Hour_of_Day', 'Is_Peak_Hour', 'Is_Weekend', 'Day_Part'
        ]
    
    def get_feature_statistics(self) -> Dict:
        """Return statistics about the feature engineering process"""
        if not self.fitted:
            return {"error": "Pipeline not fitted yet"}
        
        return {
            "total_encoders": len(self.encoders),
            "total_scalers": len(self.scalers),
            "total_features": len(self.get_feature_importance_names()),
            "encoder_features": list(self.encoders.keys()),
            "is_fitted": self.fitted
        }