import os
from typing import List
import logging 

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for Cloud Function - Updated for BigQuery storage"""
    
    def __init__(self):
        # MongoDB configuration (still used for wallet data)
        self.mongo_uri = os.getenv('MONGO_URI')
        self.db_name = os.getenv('DB_NAME', 'crypto_tracker')
        self.wallets_collection = os.getenv('WALLETS_COLLECTION', 'smart_wallets')
        
        # BigQuery configuration (for transfer data storage)
        # Can be different from the current project - useful for shared datasets
        self.bigquery_project_id = os.getenv('BIGQUERY_PROJECT_ID', 'crypto-tracker-cloudrun')
        self.bigquery_dataset_id = os.getenv('BIGQUERY_DATASET_ID', 'crypto_analysis')
        self.bigquery_transfers_table = os.getenv('BIGQUERY_TRANSFERS_TABLE', 'transfers')
        self.bigquery_location = os.getenv('BIGQUERY_LOCATION', 'asia-southeast1')
        
        # Current Cloud Function project (for other GCP services)
        self.cloud_function_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        logger.info(f"BigQuery project: {self.bigquery_project_id}")
        logger.info(f"BigQuery location: {self.bigquery_location}")
        logger.info(f"Cloud Function project: {self.cloud_function_project}")
        
        # API configuration
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        
        # General settings
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Function limits
        self.max_wallets = 200
        self.max_days_back = 7.0
        self.timeout_seconds = 300
        
        # Supported options
        self.supported_networks = ['ethereum', 'base']
        self.supported_analysis_types = ['buy', 'sell']
        
        # API configurations
        self.alchemy_endpoints = {
            'ethereum': f'https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}',
            'base': f'https://base-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}'
        }
        
        # Network specific settings
        self.blocks_per_hour = {
            'ethereum': 300,   # ~12 second blocks
            'base': 1800       # ~2 second blocks
        }
        
        # Network wallet limits
        self.network_wallet_limits = {
            'ethereum': 100,
            'base': 100
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # MongoDB validation (for wallet data)
        if not self.mongo_uri:
            errors.append("MONGO_URI not configured")
        
        # Alchemy API validation
        if not self.alchemy_api_key:
            errors.append("ALCHEMY_API_KEY not configured")
        
        # BigQuery validation
        if not self.bigquery_project_id:
            errors.append("BIGQUERY_PROJECT_ID not configured (set BIGQUERY_PROJECT_ID or GOOGLE_CLOUD_PROJECT)")
        
        # Optional BigQuery validations with warnings
        if not self.bigquery_dataset_id:
            errors.append("BIGQUERY_DATASET_ID not configured (defaulting to 'crypto_analysis')")
        
        return errors