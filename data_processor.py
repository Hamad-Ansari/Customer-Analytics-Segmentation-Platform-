# backend/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataProcessor:
    """Professional data processing engine for Harminder's Platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.version = "2.1.0"
        
    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate input data structure and quality"""
        validation_report = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Check for required columns
            required_columns = ['Age', 'Annual_Income', 'Spending_Score']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                validation_report['status'] = 'failed'
                validation_report['issues'].append(f"Missing columns: {missing_cols}")
            
            # Check data types
            numeric_columns = ['Age', 'Annual_Income', 'Spending_Score']
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation_report['status'] = 'failed'
                        validation_report['issues'].append(f"Column {col} must be numeric")
            
            # Calculate metrics
            validation_report['metrics'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            self.logger.info(f"Harminder's Data Validation completed: {validation_report['status']}")
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            validation_report['status'] = 'error'
            validation_report['issues'].append(f"Validation error: {str(e)}")
            return validation_report
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for ML models"""
        processed_df = df.copy()
        
        # Handle missing values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
        
        # Remove outliers using IQR method
        for col in numeric_cols:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df = processed_df[(processed_df[col] >= lower_bound) & 
                                       (processed_df[col] <= upper_bound)]
        
        # Feature engineering
        if 'Annual_Income' in processed_df.columns and 'Age' in processed_df.columns:
            processed_df['Income_to_Age_Ratio'] = processed_df['Annual_Income'] / processed_df['Age']
        
        self.logger.info("Harminder's Data Preprocessing completed")
        return processed_df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better segmentation"""
        features_df = df.copy()
        
        # Create customer segments based on spending
        if 'Spending_Score' in features_df.columns:
            conditions = [
                (features_df['Spending_Score'] <= 20),
                (features_df['Spending_Score'] <= 40),
                (features_df['Spending_Score'] <= 60),
                (features_df['Spending_Score'] <= 80),
                (features_df['Spending_Score'] > 80)
            ]
            choices = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            features_df['Spending_Category'] = np.select(conditions, choices, default='Medium')
        
        # Create age groups
        if 'Age' in features_df.columns:
            bins = [0, 25, 35, 50, 65, 100]
            labels = ['18-25', '26-35', '36-50', '51-65', '65+']
            features_df['Age_Group'] = pd.cut(features_df['Age'], bins=bins, labels=labels, right=False)
        
        return features_df