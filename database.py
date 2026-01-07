# backend/database.py
import sqlite3
import pandas as pd
from datetime import datetime
import json
import logging

class DatabaseManager:
    """Professional database management for Harminder's Platform"""
    
    def __init__(self, db_path: str = 'harminder_analytics.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create customers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT UNIQUE,
                    age INTEGER,
                    annual_income REAL,
                    spending_score INTEGER,
                    credit_score INTEGER,
                    segment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create segments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_name TEXT,
                    description TEXT,
                    criteria TEXT,
                    customer_count INTEGER,
                    avg_income REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model_runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT,
                    parameters TEXT,
                    accuracy REAL,
                    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Harminder's Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def save_customers(self, customers_df: pd.DataFrame):
        """Save customer data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Add timestamps
            customers_df['created_at'] = datetime.now()
            customers_df['updated_at'] = datetime.now()
            
            customers_df.to_sql('customers', conn, if_exists='append', index=False)
            conn.close()
            
            self.logger.info(f"Saved {len(customers_df)} customers to database")
            
        except Exception as e:
            self.logger.error(f"Error saving customers: {str(e)}")
            raise
    
    def get_customers(self, segment: str = None) -> pd.DataFrame:
        """Retrieve customers from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if segment:
                query = "SELECT * FROM customers WHERE segment = ?"
                df = pd.read_sql_query(query, conn, params=(segment,))
            else:
                df = pd.read_sql_query("SELECT * FROM customers", conn)
            
            conn.close()
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving customers: {str(e)}")
            raise
    
    def save_segment_analysis(self, segment_name: str, description: str, 
                            criteria: dict, customer_count: int, avg_income: float):
        """Save segment analysis results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO segments (segment_name, description, criteria, customer_count, avg_income)
                VALUES (?, ?, ?, ?, ?)
            ''', (segment_name, description, json.dumps(criteria), customer_count, avg_income))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Segment analysis saved: {segment_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving segment analysis: {str(e)}")
            raise
    
    def log_model_run(self, model_type: str, parameters: dict, accuracy: float, status: str = "completed"):
        """Log model training runs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_runs (model_type, parameters, accuracy, status)
                VALUES (?, ?, ?, ?)
            ''', (model_type, json.dumps(parameters), accuracy, status))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Model run logged: {model_type} - {status}")
            
        except Exception as e:
            self.logger.error(f"Error logging model run: {str(e)}")
            raise