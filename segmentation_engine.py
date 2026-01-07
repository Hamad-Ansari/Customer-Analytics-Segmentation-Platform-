# backend/segmentation_engine.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import logging
import os

class SegmentationEngine:
    """Advanced customer segmentation engine for Harminder's Platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.models = {}
        self.version = "2.1.0"
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering"""
        features = ['Age', 'Annual_Income', 'Spending_Score']
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            raise ValueError(f"Insufficient features for clustering. Need at least 2 from {features}")
        
        X = df[available_features].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, available_features
    
    def perform_kmeans(self, X: np.ndarray, n_clusters: int = 5) -> dict:
        """Perform K-Means clustering"""
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
            
            labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, labels)
            
            results = {
                'model': kmeans,
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'silhouette_score': silhouette_avg,
                'n_clusters': n_clusters,
                'algorithm': 'K-Means'
            }
            
            self.logger.info(f"Harminder's K-Means completed: {n_clusters} clusters, score: {silhouette_avg:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"K-Means error: {str(e)}")
            raise
    
    def perform_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> dict:
        """Perform DBSCAN clustering"""
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Calculate number of clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            results = {
                'model': dbscan,
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_points': sum(labels == -1),
                'algorithm': 'DBSCAN'
            }
            
            self.logger.info(f"Harminder's DBSCAN completed: {n_clusters} clusters")
            return results
            
        except Exception as e:
            self.logger.error(f"DBSCAN error: {str(e)}")
            raise
    
    def analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Analyze cluster characteristics"""
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = df_clustered.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'count'],
            'Annual_Income': ['mean', 'std'],
            'Spending_Score': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.reset_index()
        
        # Add cluster personas
        personas = []
        for idx, row in cluster_stats.iterrows():
            age_mean = row.get('Age_mean', 0)
            income_mean = row.get('Annual_Income_mean', 0)
            spend_mean = row.get('Spending_Score_mean', 0)
            
            if income_mean > 150000 and spend_mean > 70:
                persona = "Premium Spenders"
            elif age_mean < 30 and spend_mean > 60:
                persona = "Young High-Engagement"
            elif income_mean < 50000:
                persona = "Budget Conscious"
            elif age_mean > 50:
                persona = "Established Customers"
            else:
                persona = "Mainstream Segment"
            
            personas.append(persona)
        
        cluster_stats['Persona'] = personas
        
        return cluster_stats
    
    def save_model(self, model, filename: str):
        """Save trained model to file"""
        try:
            os.makedirs('models', exist_ok=True)
            filepath = f'models/{filename}.joblib'
            joblib.dump(model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filename: str):
        """Load trained model from file"""
        try:
            filepath = f'models/{filename}.joblib'
            if os.path.exists(filepath):
                model = joblib.load(filepath)
                self.logger.info(f"Model loaded from {filepath}")
                return model
            else:
                raise FileNotFoundError(f"Model file {filepath} not found")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise