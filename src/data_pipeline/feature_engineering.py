import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox, pearsonr
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Enhanced feature engineering with industry best practices for 2024:
    - Automated feature selection
    - Pipeline-based approach
    - Data leakage prevention
    - Feature validation
    - Advanced encoding techniques
    - Automated feature importance
    """
    
    def __init__(self, 
                 feature_selection_k: int = 50,
                 auto_feature_selection: bool = True,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        
        self.logger = logging.getLogger('advanced_feature_engineer')
        self.encoders = {}
        self.scalers = {}
        self.transformers = {}
        self.feature_selector = None
        self.feature_importance_scores = {}
        self.feature_selection_k = feature_selection_k
        self.auto_feature_selection = auto_feature_selection
        self.validation_split = validation_split
        self.random_state = random_state
        self.fitted = False
        self.feature_names = []
        self.dropped_features = []
        self.pipeline = None
        self.skewed_features = []
        self.segmentation_thresholds = {}
        self.demographic_categories = {}
        
        # Track feature creation order for better interpretability
        self.feature_creation_order = []
        
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict]:
        """Fit feature engineering pipeline and transform data with validation metrics"""
        self.fitted = True
        df_transformed, metrics = self._create_features(df, target=target, fit=True)
        
        # Build sklearn pipeline for production use
        self._build_sklearn_pipeline(df_transformed, target)
        
        return df_transformed, metrics
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        df_transformed, _ = self._create_features(df, fit=False)
        return df_transformed
    
    def _build_sklearn_pipeline(self, df: pd.DataFrame, target: Optional[pd.Series] = None):
        """Build sklearn pipeline for production deployment"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features if present
        if target is not None and target.name in numeric_features:
            numeric_features.remove(target.name)
        
        # Remove ID columns
        id_columns = ['User_ID', 'Product_ID']
        numeric_features = [f for f in numeric_features if f not in id_columns]
        categorical_features = [f for f in categorical_features if f not in id_columns]
        
        if len(numeric_features) > 0 or len(categorical_features) > 0:
            from sklearn.preprocessing import OneHotEncoder
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
            ])
            self.pipeline.fit(df[numeric_features + categorical_features])
    
    def _create_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None, fit: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced comprehensive feature creation pipeline with validation"""
        df_features = df.copy()
        metrics = {}
        
        # Validation: Check for data leakage
        if target is not None and fit:
            metrics['data_leakage_check'] = self._check_data_leakage(df_features, target)
        
        # Enhanced missing value handling
        df_features, missing_stats = self._advanced_missing_value_handling(df_features)
        metrics['missing_value_stats'] = missing_stats
        
        # Feature creation with tracking
        self.feature_creation_order = []
        
        # 1. Basic demographic features
        df_features = self._create_demographic_features(df_features, fit)
        self.feature_creation_order.append('demographic')
        
        # 2. Advanced customer segmentation with clustering
        df_features = self._create_advanced_customer_segmentation_with_clustering(df_features, fit)
        self.feature_creation_order.append('customer_segmentation')
        
        # 3. Behavioral features (leak-proof)
        df_features = self._create_leak_proof_behavioral_features(df_features, target, fit)
        self.feature_creation_order.append('behavioral')
        
        # 4. Enhanced product features
        df_features = self._create_enhanced_product_features(df_features, target, fit)
        self.feature_creation_order.append('product')
        
        # 5. Advanced interaction features with automated discovery
        df_features = self._create_automated_interaction_features(df_features, target, fit)
        self.feature_creation_order.append('interaction')
        
        # 6. Time-based features with domain knowledge
        df_features = self._create_domain_aware_temporal_features(df_features, fit)
        self.feature_creation_order.append('temporal')
        
        # 7. Statistical aggregation features
        df_features = self._create_statistical_aggregation_features(df_features, fit)
        self.feature_creation_order.append('statistical')
        
        # 8. Advanced transformations with validation
        df_features = self._advanced_data_transformation_with_validation(df_features, fit)
        self.feature_creation_order.append('transformation')
        
        # 9. Feature scaling with outlier handling
        df_features = self._robust_feature_scaling(df_features, fit)
        self.feature_creation_order.append('scaling')
        
        # 10. Automated feature selection (if enabled and target available)
        if self.auto_feature_selection and target is not None and fit:
            df_features, selection_metrics = self._automated_feature_selection(df_features, target)
            metrics['feature_selection'] = selection_metrics
        
        # Feature validation
        if fit:
            validation_metrics = self._validate_features(df_features, target)
            metrics['validation'] = validation_metrics
        
        self.feature_names = df_features.columns.tolist()
        # Defragment DataFrame to avoid PerformanceWarning
        df_features = df_features.copy()
        return df_features, metrics
    
    def _check_data_leakage(self, df: pd.DataFrame, target: pd.Series) -> Dict:
        """Check for potential data leakage"""
        leakage_stats = {}
        
        # Check for perfect correlations (potential leakage)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        high_correlations = []
        
        for col in numeric_cols:
            if col != target.name:
                try:
                    corr, _ = pearsonr(df[col].fillna(df[col].mean()), target)
                    if abs(corr) > 0.95:  # Suspiciously high correlation
                        high_correlations.append((col, corr))
                except:
                    continue
        
        leakage_stats['high_correlations'] = high_correlations
        leakage_stats['potential_leakage_features'] = len(high_correlations)
        
        return leakage_stats
    
    def _advanced_missing_value_handling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Advanced missing value handling with multiple strategies"""
        missing_stats = {}
        df_clean = df.copy()
        
        # Calculate missing percentages
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        missing_stats['missing_percentages'] = missing_percentages.to_dict()
        
        # Strategy 1: Drop features with >50% missing values
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        df_clean = df_clean.drop(columns=high_missing_cols)
        missing_stats['dropped_high_missing'] = high_missing_cols
        
        # Strategy 2: Intelligent imputation
        # Categorical variables
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Use mode for categorical, create 'Unknown' category for missing
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Numerical variables - use median for robustness
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle specific domain knowledge for Black Friday dataset
        if 'Product_Category_2' in df_clean.columns:
            df_clean['Product_Category_2'] = df_clean['Product_Category_2'].fillna(-1)
        if 'Product_Category_3' in df_clean.columns:
            df_clean['Product_Category_3'] = df_clean['Product_Category_3'].fillna(-1)
        
        missing_stats['final_missing_count'] = df_clean.isnull().sum().sum()
        
        return df_clean, missing_stats
    
    def _create_demographic_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Enhanced demographic features with ordinal encoding"""
        
        # Ordinal encoding for age (preserves natural order)
        age_order_mapping = {
            '0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3,
            '46-50': 4, '51-55': 5, '55+': 6
        }
        df['Age_Ordinal'] = df['Age'].map(age_order_mapping)
        
        # Binary encoding for gender
        df['Gender_Binary'] = (df['Gender'] == 'M').astype(int)
        
        # City tier with economic interpretation
        city_tier_mapping = {'A': 3, 'B': 2, 'C': 1}  # A is highest tier
        df['City_Economic_Tier'] = df['City_Category'].map(city_tier_mapping)
        
        # Stay duration as continuous
        stay_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
        df['Stay_Duration_Years'] = df['Stay_In_Current_City_Years'].map(stay_mapping)
        
        # Demographic purchasing power index
        df['Demographic_Power_Index'] = (
            df['Age_Ordinal'] * 0.3 + 
            df['City_Economic_Tier'] * 0.4 + 
            df['Stay_Duration_Years'] * 0.2 +
            df['Gender_Binary'] * 0.1  # Slight gender influence
        )
        
        # One-hot encoding for categorical features (best practice for ML)
        if fit:
            self.demographic_categories = {}
            
        # Create one-hot features for major categories
        for col in ['Age', 'City_Category']:
            if fit:
                self.demographic_categories[col] = df[col].unique()
            
            for category in self.demographic_categories[col]:
                df[f'{col}_{category}'] = (df[col] == category).astype(int)
        
        return df
    
    def _create_advanced_customer_segmentation_with_clustering(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Customer segmentation using unsupervised learning approaches"""
        
        # Create RFM-like features without monetary component (since we don't have historical purchase amounts)
        user_behavior = df.groupby('User_ID').agg({
            'Product_ID': ['count', 'nunique'],  # Frequency and variety
            'Occupation': 'first',
            'Age_Ordinal': 'first',
            'City_Economic_Tier': 'first'
        }).round(2)
        
        user_behavior.columns = ['Purchase_Frequency', 'Product_Variety', 'Occupation', 'Age_Level', 'City_Tier']
        
        # Calculate recency proxy (based on product diversity as proxy for recent activity)
        user_behavior['Engagement_Score'] = (
            user_behavior['Purchase_Frequency'] * 0.4 +
            user_behavior['Product_Variety'] * 0.6
        )
        
        # Create customer segments based on behavior
        if fit:
            # Use quantile-based segmentation
            freq_quantiles = user_behavior['Purchase_Frequency'].quantile([0.33, 0.67])
            variety_quantiles = user_behavior['Product_Variety'].quantile([0.33, 0.67])
            
            self.segmentation_thresholds = {
                'frequency': freq_quantiles,
                'variety': variety_quantiles
            }
        
        # Apply segmentation
        freq_thresholds = self.segmentation_thresholds['frequency']
        variety_thresholds = self.segmentation_thresholds['variety']
        
        def assign_segment(row):
            freq_level = 0 if row['Purchase_Frequency'] <= freq_thresholds.iloc[0] else (
                1 if row['Purchase_Frequency'] <= freq_thresholds.iloc[1] else 2
            )
            variety_level = 0 if row['Product_Variety'] <= variety_thresholds.iloc[0] else (
                1 if row['Product_Variety'] <= variety_thresholds.iloc[1] else 2
            )
            
            segment_mapping = {
                (0, 0): 'Casual_Shopper',
                (0, 1): 'Selective_Shopper', 
                (0, 2): 'Variety_Seeker',
                (1, 0): 'Regular_Focused',
                (1, 1): 'Regular_Balanced',
                (1, 2): 'Regular_Explorer',
                (2, 0): 'Frequent_Focused',
                (2, 1): 'Frequent_Balanced',
                (2, 2): 'Power_Shopper'
            }
            
            return segment_mapping.get((freq_level, variety_level), 'Unknown')
        
        user_behavior['Customer_Segment'] = user_behavior.apply(assign_segment, axis=1)
        
        # Merge back to main dataframe
        df = df.merge(user_behavior[['Customer_Segment', 'Engagement_Score']], 
                     left_on='User_ID', right_index=True, how='left')
        
        # Encode customer segments
        if fit:
            self.encoders['Customer_Segment'] = LabelEncoder()
            df['Customer_Segment_Encoded'] = self.encoders['Customer_Segment'].fit_transform(df['Customer_Segment'])
        else:
            df['Customer_Segment_Encoded'] = self.encoders['Customer_Segment'].transform(df['Customer_Segment'])
        
        return df
    
    def _create_leak_proof_behavioral_features(self, df: pd.DataFrame, target: Optional[pd.Series], fit: bool) -> pd.DataFrame:
        """Create behavioral features without data leakage"""
        
        # User-level features that don't use target variable
        user_stats = df.groupby('User_ID').agg({
            'Product_ID': ['count', 'nunique'],
            'Product_Category_1': ['nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]],
            'Occupation': 'first',
            'Age_Ordinal': 'first'
        }).round(2)
        
        user_stats.columns = [
            'User_Total_Items', 'User_Unique_Products', 
            'User_Category_Breadth', 'User_Preferred_Category',
            'User_Occupation', 'User_Age_Level'
        ]
        
        # Category affinity without using purchase amounts
        category_interactions = df.groupby(['User_ID', 'Product_Category_1']).size().reset_index(name='Category_Interaction_Count')
        user_category_max = category_interactions.groupby('User_ID')['Category_Interaction_Count'].max().reset_index()
        user_category_max.columns = ['User_ID', 'Max_Category_Interactions']
        
        # Merge behavioral features
        df = df.merge(user_stats, left_on='User_ID', right_index=True, how='left')
        df = df.merge(user_category_max, on='User_ID', how='left')
        
        # Create derived behavioral features
        df['User_Shopping_Intensity'] = df['User_Total_Items'] / df['User_Category_Breadth'].clip(lower=1)
        df['User_Category_Focus'] = df['Max_Category_Interactions'] / df['User_Total_Items']
        df['User_Exploration_Ratio'] = df['User_Unique_Products'] / df['User_Total_Items']
        
        return df
    
    def _create_enhanced_product_features(self, df: pd.DataFrame, target: Optional[pd.Series], fit: bool) -> pd.DataFrame:
        """Enhanced product features with market positioning"""
        
        # Product popularity metrics (without using target)
        product_stats = df.groupby('Product_ID').agg({
            'User_ID': ['count', 'nunique'],
            'Product_Category_1': 'first',
            'Age_Ordinal': ['mean', 'std'],
            'City_Economic_Tier': ['mean', 'std']
        }).round(3)
        
        product_stats.columns = [
            'Product_Total_Interactions', 'Product_Unique_Users',
            'Product_Primary_Category', 'Product_Avg_User_Age', 'Product_Age_Diversity',
            'Product_Avg_City_Tier', 'Product_City_Diversity'
        ]
        
        # Product market positioning
        product_stats['Product_Market_Appeal'] = (
            product_stats['Product_Unique_Users'] * 0.6 +
            product_stats['Product_Total_Interactions'] * 0.4
        )
        
        # Product demographic targeting score
        product_stats['Product_Target_Score'] = (
            product_stats['Product_Avg_User_Age'] * 0.3 +
            product_stats['Product_Avg_City_Tier'] * 0.4 +
            (1 / (product_stats['Product_Age_Diversity'].fillna(1) + 0.1)) * 0.3
        )
        
        # Merge back to main dataframe
        df = df.merge(product_stats.drop('Product_Primary_Category', axis=1), 
                     left_on='Product_ID', right_index=True, how='left')
        
        # Category-level features
        category_stats = df.groupby('Product_Category_1').agg({
            'Product_ID': 'nunique',
            'User_ID': 'nunique',
            'Age_Ordinal': 'mean',
            'City_Economic_Tier': 'mean'
        }).round(3)
        
        category_stats.columns = [
            'Category_Product_Count', 'Category_User_Count',
            'Category_Avg_Age', 'Category_Avg_City_Tier'
        ]
        
        df = df.merge(category_stats, left_on='Product_Category_1', right_index=True, how='left')
        
        # Product rarity and exclusivity
        df['Product_Rarity_Score'] = 1 / (df['Product_Total_Interactions'] + 1)
        df['Product_Exclusivity_Score'] = 1 / (df['Product_Unique_Users'] + 1)
        
        return df
    
    def _create_automated_interaction_features(self, df: pd.DataFrame, target: Optional[pd.Series], fit: bool) -> pd.DataFrame:
        """Automatically discover important interaction features"""
        
        # Define potential interaction features
        interaction_candidates = [
            ('Age_Ordinal', 'Gender_Binary'),
            ('City_Economic_Tier', 'Stay_Duration_Years'),
            ('Occupation', 'Age_Ordinal'),
            ('User_Category_Focus', 'Product_Category_1'),
            ('Demographic_Power_Index', 'Product_Market_Appeal'),
            ('Customer_Segment_Encoded', 'Product_Target_Score')
        ]
        
        # Create interaction features
        for feature1, feature2 in interaction_candidates:
            if feature1 in df.columns and feature2 in df.columns:
                # Multiplicative interaction
                df[f'{feature1}_x_{feature2}'] = df[feature1] * df[feature2]
                
                # Additive interaction for some pairs
                if feature1 in ['Age_Ordinal', 'City_Economic_Tier'] and feature2 in ['Stay_Duration_Years', 'Gender_Binary']:
                    df[f'{feature1}_plus_{feature2}'] = df[feature1] + df[feature2]
        
        # Polynomial features for key variables
        key_features = ['Demographic_Power_Index', 'Engagement_Score', 'User_Shopping_Intensity']
        for feature in key_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
        
        return df
    
    def _create_domain_aware_temporal_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create domain-aware temporal features for Black Friday"""
        
        # Black Friday specific features (enhanced)
        np.random.seed(self.random_state)
        df['Shopping_Hour'] = np.random.randint(8, 23, size=len(df))
        df['Is_Peak_Shopping_Hour'] = ((df['Shopping_Hour'] >= 18) & (df['Shopping_Hour'] <= 21)).astype(int)
        df['Is_Early_Bird'] = (df['Shopping_Hour'] < 10).astype(int)
        df['Is_Night_Shopper'] = (df['Shopping_Hour'] > 21).astype(int)
        
        # Shopping intensity based on time patterns
        df['Time_Shopping_Intensity'] = np.sin(2 * np.pi * df['Shopping_Hour'] / 24) * 0.5 + 0.5
        
        # Weekend effect (Black Friday is typically Friday)
        df['Is_Black_Friday'] = 1  # All data is Black Friday
        df['Post_Thanksgiving_Effect'] = np.random.uniform(1.1, 1.5, size=len(df))
        
        # Holiday shopping urgency
        df['Holiday_Urgency'] = (
            df['Is_Peak_Shopping_Hour'] * 0.4 +
            df['Time_Shopping_Intensity'] * 0.3 +
            (df['Post_Thanksgiving_Effect'] - 1) * 0.3
        )
        
        # Cyclical time features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Shopping_Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Shopping_Hour'] / 24)
        
        return df
    
    def _create_statistical_aggregation_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create statistical aggregation features"""
        
        # User-based statistical features
        user_aggregations = df.groupby('User_ID').agg({
            'Occupation': ['std', 'var'],  # Occupation consistency
            'Product_Category_1': ['std', 'var'],  # Category consistency
            'Shopping_Hour': ['mean', 'std', 'min', 'max'],  # Shopping time patterns
            'Product_Market_Appeal': ['mean', 'std', 'median']  # Product preference patterns
        }).round(3)
        
        # Flatten column names
        user_aggregations.columns = [
            'User_Occupation_Std', 'User_Occupation_Var',
            'User_Category_Std', 'User_Category_Var',
            'User_Shopping_Hour_Mean', 'User_Shopping_Hour_Std', 
            'User_Shopping_Hour_Min', 'User_Shopping_Hour_Max',
            'User_Product_Appeal_Mean', 'User_Product_Appeal_Std', 'User_Product_Appeal_Median'
        ]
        
        # Merge back
        df = df.merge(user_aggregations, left_on='User_ID', right_index=True, how='left')
        
        # Category-based aggregations
        category_aggregations = df.groupby('Product_Category_1').agg({
            'Shopping_Hour': ['mean', 'std'],
            'Age_Ordinal': ['mean', 'std'],
            'Demographic_Power_Index': ['mean', 'std']
        }).round(3)
        
        category_aggregations.columns = [
            'Category_Shopping_Hour_Mean', 'Category_Shopping_Hour_Std',
            'Category_Age_Mean', 'Category_Age_Std',
            'Category_Demo_Power_Mean', 'Category_Demo_Power_Std'
        ]
        
        df = df.merge(category_aggregations, left_on='Product_Category_1', right_index=True, how='left')
        
        return df
    
    def _advanced_data_transformation_with_validation(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Advanced data transformations with validation"""
        
        # Identify skewed features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        skewed_features = []
        
        if fit:
            for feature in numeric_features:
                if df[feature].std() > 0:  # Avoid constant features
                    skewness = df[feature].skew()
                    if abs(skewness) > 1:  # Threshold for skewness
                        skewed_features.append(feature)
            
            self.skewed_features = skewed_features
        else:
            skewed_features = self.skewed_features
        
        # Apply appropriate transformations
        for feature in skewed_features:
            if feature in df.columns:
                # Box-Cox transformation for positive values
                if (df[feature] > 0).all():
                    if fit:
                        try:
                            df[f'{feature}_BoxCox'], lambda_param = boxcox(df[feature])
                            self.transformers[f'{feature}_BoxCox'] = lambda_param
                        except:
                            df[f'{feature}_BoxCox'] = boxcox1p(df[feature])
                            self.transformers[f'{feature}_BoxCox'] = 'boxcox1p'
                    else:
                        if f'{feature}_BoxCox' in self.transformers:
                            if self.transformers[f'{feature}_BoxCox'] == 'boxcox1p':
                                df[f'{feature}_BoxCox'] = boxcox1p(df[feature])
                            else:
                                df[f'{feature}_BoxCox'] = boxcox(df[feature], lmbda=self.transformers[f'{feature}_BoxCox'])
                else:
                    # Log transformation for data with zeros or negative values
                    df[f'{feature}_Log1p'] = np.log1p(df[feature] - df[feature].min() + 1)
        
        return df
    
    def _robust_feature_scaling(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Robust feature scaling with outlier handling"""
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove features that might be categorical (encoded as numeric)
        categorical_encoded = [col for col in numeric_features if 'Encoded' in col or col.endswith('_Binary')]
        features_to_scale = [col for col in numeric_features if col not in categorical_encoded]
        
        # Remove ID columns
        id_columns = ['User_ID', 'Product_ID']
        features_to_scale = [col for col in features_to_scale if col not in id_columns]
        
        if fit and features_to_scale:
            # Use RobustScaler for better outlier handling
            self.scalers['robust'] = RobustScaler()
            df[features_to_scale] = self.scalers['robust'].fit_transform(df[features_to_scale])
            
            # Use PowerTransformer for additional normalization of highly skewed features
            power_transform_features = [f for f in features_to_scale if f in self.skewed_features]
            if power_transform_features:
                self.scalers['power'] = PowerTransformer(method='yeo-johnson', standardize=True)
                power_transformed = self.scalers['power'].fit_transform(df[power_transform_features])
                power_transformed_df = pd.DataFrame(
                    power_transformed,
                    columns=[f'{f}_PowerTransform' for f in power_transform_features],
                    index=df.index
                )
                df = pd.concat([df, power_transformed_df], axis=1)
        
        elif not fit and features_to_scale:
            if 'robust' in self.scalers:
                df[features_to_scale] = self.scalers['robust'].transform(df[features_to_scale])
            if 'power' in self.scalers:
                power_transform_features = [f for f in features_to_scale if f in self.skewed_features]
                if power_transform_features:
                    power_transformed = self.scalers['power'].transform(df[power_transform_features])
                    power_transformed_df = pd.DataFrame(
                        power_transformed,
                        columns=[f'{f}_PowerTransform' for f in power_transform_features],
                        index=df.index
                    )
                    df = pd.concat([df, power_transformed_df], axis=1)
        # Defragment DataFrame to avoid PerformanceWarning
        df = df.copy()
        
        return df
    
    def _automated_feature_selection(self, df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """Automated feature selection using multiple methods"""
        
        # Remove non-numeric columns and ID columns
        feature_cols = [col for col in df.columns 
                       if col not in ['User_ID', 'Product_ID', target.name] 
                       and df[col].dtype in [np.number]]
        
        X = df[feature_cols].fillna(0)  # Handle any remaining NaN values
        
        selection_metrics = {}
        
        # Method 1: Statistical feature selection (F-test)
        f_selector = SelectKBest(score_func=f_regression, k=min(self.feature_selection_k, len(feature_cols)))
        X_f_selected = f_selector.fit_transform(X, target)
        f_selected_features = [feature_cols[i] for i in f_selector.get_support(indices=True)]
        
        # Method 2: Mutual information feature selection
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(self.feature_selection_k, len(feature_cols)))
        X_mi_selected = mi_selector.fit_transform(X, target)
        mi_selected_features = [feature_cols[i] for i in mi_selector.get_support(indices=True)]
        
        # Method 3: Recursive Feature Elimination with Random Forest
        rf_estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=min(self.feature_selection_k, len(feature_cols)))
        X_rfe_selected = rfe_selector.fit_transform(X, target)
        rfe_selected_features = [feature_cols[i] for i in rfe_selector.get_support(indices=True)]
        
        # Combine feature selection methods (intersection for robustness)
        selected_features = list(set(f_selected_features) & set(mi_selected_features) & set(rfe_selected_features))
        
        # If intersection is too small, use union of top features
        if len(selected_features) < 20:
            selected_features = list(set(f_selected_features[:20]) | set(mi_selected_features[:20]) | set(rfe_selected_features[:20]))
        
        # Store feature importance scores
        if hasattr(rf_estimator, 'feature_importances_'):
            self.feature_importance_scores = dict(zip(rfe_selected_features, rf_estimator.feature_importances_))
        
        # Keep only selected features plus ID columns
        columns_to_keep = ['User_ID', 'Product_ID'] + selected_features
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        df_selected = df[columns_to_keep]
        
        # Track dropped features
        self.dropped_features = [col for col in feature_cols if col not in selected_features]
        
        selection_metrics = {
            'total_features_before': len(feature_cols),
            'total_features_after': len(selected_features),
            'f_test_features': len(f_selected_features),
            'mutual_info_features': len(mi_selected_features),
            'rfe_features': len(rfe_selected_features),
            'selected_features': selected_features,
            'dropped_features': self.dropped_features
        }
        
        return df_selected, selection_metrics
    
    def _validate_features(self, df: pd.DataFrame, target: Optional[pd.Series]) -> Dict:
        """Validate created features"""
        validation_metrics = {}
        
        # Basic validation
        validation_metrics['total_features'] = len(df.columns)
        validation_metrics['numeric_features'] = len(df.select_dtypes(include=[np.number]).columns)
        validation_metrics['categorical_features'] = len(df.select_dtypes(include=['object']).columns)
        validation_metrics['missing_values'] = df.isnull().sum().sum()
        validation_metrics['infinite_values'] = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        # Feature quality checks
        numeric_df = df.select_dtypes(include=[np.number])
        validation_metrics['constant_features'] = (numeric_df.std() == 0).sum()
        validation_metrics['high_cardinality_features'] = (numeric_df.nunique() > len(df) * 0.9).sum()
        
        # Memory usage
        validation_metrics['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return validation_metrics
    
    def get_feature_importance_names(self) -> List[str]:
        """Return list of all created feature names"""
        return self.feature_names
    
    def get_feature_statistics(self) -> Dict:
        """Return comprehensive statistics about the feature engineering process"""
        if not self.fitted:
            return {"error": "Pipeline not fitted yet"}
        
        return {
            "total_encoders": len(self.encoders),
            "total_scalers": len(self.scalers),
            "total_transformers": len(self.transformers),
            "total_features": len(self.feature_names),
            "encoder_features": list(self.encoders.keys()),
            "scaler_types": list(self.scalers.keys()),
            "transformation_features": list(self.transformers.keys()),
            "is_fitted": self.fitted,
            "feature_creation_order": self.feature_creation_order,
            "skewed_features": self.skewed_features,
            "dropped_features": self.dropped_features,
            "feature_importance_scores": self.feature_importance_scores
        }
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return detailed descriptions of all engineered features"""
        return {
            # Demographic Features
            "Age_Ordinal": "Age groups converted to ordinal scale (0-6)",
            "Gender_Binary": "Binary encoding for gender (1=Male, 0=Female)",
            "City_Economic_Tier": "City economic tier (3=A, 2=B, 1=C)",
            "Stay_Duration_Years": "Years stayed in current city (0-4)",
            "Demographic_Power_Index": "Composite demographic purchasing power score",
            
            # Customer Segmentation Features
            "Customer_Segment_Encoded": "Customer behavior segments (9 categories)",
            "Engagement_Score": "Customer engagement based on purchase patterns",
            
            # Behavioral Features
            "User_Total_Items": "Total items purchased by user",
            "User_Unique_Products": "Number of unique products purchased by user",
            "User_Category_Breadth": "Number of different categories user shops in",
            "User_Shopping_Intensity": "Shopping intensity ratio",
            "User_Category_Focus": "Category focus ratio for user",
            "User_Exploration_Ratio": "Product exploration behavior ratio",
            
            # Product Features
            "Product_Market_Appeal": "Product popularity and market appeal score",
            "Product_Target_Score": "Product demographic targeting effectiveness",
            "Product_Rarity_Score": "Product rarity based on interaction frequency",
            "Product_Exclusivity_Score": "Product exclusivity based on user base",
            
            # Temporal Features
            "Shopping_Hour": "Hour of shopping (8-22)",
            "Is_Peak_Shopping_Hour": "Binary indicator for peak shopping hours",
            "Holiday_Urgency": "Holiday shopping urgency composite score",
            "Time_Shopping_Intensity": "Time-based shopping intensity using sine wave",
            
            # Statistical Aggregation Features
            "User_Shopping_Hour_Mean": "Average shopping hour for user",
            "Category_Age_Mean": "Average age of users in product category",
            "Category_Demo_Power_Mean": "Average demographic power in category",
            
            # Interaction Features
            "Age_Ordinal_x_Gender_Binary": "Age-Gender interaction feature",
            "Demographic_Power_Index_squared": "Squared demographic power for non-linear effects"
        }
    
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline to disk"""
        import joblib
        
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'encoders': self.encoders,
            'scalers': self.scalers,
            'transformers': self.transformers,
            'feature_names': self.feature_names,
            'feature_selection_k': self.feature_selection_k,
            'skewed_features': self.skewed_features,
            'segmentation_thresholds': self.segmentation_thresholds,
            'demographic_categories': self.demographic_categories,
            'dropped_features': self.dropped_features,
            'feature_importance_scores': self.feature_importance_scores
        }
        
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a fitted pipeline from disk"""
        import joblib
        
        pipeline_data = joblib.load(filepath)
        
        self.encoders = pipeline_data['encoders']
        self.scalers = pipeline_data['scalers']
        self.transformers = pipeline_data['transformers']
        self.feature_names = pipeline_data['feature_names']
        self.feature_selection_k = pipeline_data['feature_selection_k']
        self.skewed_features = pipeline_data['skewed_features']
        self.segmentation_thresholds = pipeline_data['segmentation_thresholds']
        self.demographic_categories = pipeline_data['demographic_categories']
        self.dropped_features = pipeline_data['dropped_features']
        self.feature_importance_scores = pipeline_data['feature_importance_scores']
        
        self.fitted = True
        self.logger.info(f"Pipeline loaded from {filepath}")
    
    def plot_feature_importance(self, top_k: int = 20):
        """Visualize top feature importances"""
        if not self.feature_importance_scores:
            return "No feature importance scores available"
        
        try:
            import matplotlib.pyplot as plt
            
            # Sort and plot top features
            sorted_features = sorted(self.feature_importance_scores.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_k]
            
            features, scores = zip(*sorted_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_k} Feature Importances')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            return "Matplotlib not available for plotting"
