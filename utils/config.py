# utils/config.py - Updated to remove MongoDB dependencies

import os
from typing import List
import logging 

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for Cloud Function - BigQuery only, no MongoDB"""
    
    def __init__(self):
        
        # BigQuery configuration (for both wallets and transfers)
        self.bigquery_project_id = os.getenv('BIGQUERY_PROJECT_ID', 'crypto-tracker-cloudrun')
        self.bigquery_dataset_id = os.getenv('BIGQUERY_DATASET_ID', 'crypto_analysis')
        self.bigquery_transfers_table = os.getenv('BIGQUERY_TRANSFERS_TABLE', 'transfers')
        self.bigquery_wallets_table = os.getenv('BIGQUERY_WALLETS_TABLE', 'smart_wallets')
        self.bigquery_location = os.getenv('BIGQUERY_LOCATION', 'asia-southeast1')
        
        # Current Cloud Function project
        self.cloud_function_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        logger.info(f"BigQuery project: {self.bigquery_project_id}")
        logger.info(f"BigQuery dataset: {self.bigquery_dataset_id}")
        logger.info(f"BigQuery location: {self.bigquery_location}")
        logger.info(f"Cloud Function project: {self.cloud_function_project}")
        
        # API configuration
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.etherscan_endpoint = "https://api.etherscan.io/v2/api"
        
        # Chain ID mapping for Etherscan V2
        self.chain_ids = {
            'ethereum': 1,
            'base': 8453
        }
        
        self.etherscan_api_rate_limit = 0.2
        
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
        
        # Alchemy API validation
        if not self.alchemy_api_key:
            errors.append("ALCHEMY_API_KEY not configured")
            
        if not self.etherscan_api_key:
            errors.append("ETHERSCAN_API_KEY not configured (will use heuristics only)")
            
        # BigQuery validation
        if not self.bigquery_project_id:
            errors.append("BIGQUERY_PROJECT_ID not configured")
        
        # Optional BigQuery validations with warnings
        if not self.bigquery_dataset_id:
            errors.append("BIGQUERY_DATASET_ID not configured (defaulting to 'crypto_analysis')")
        
        return errors