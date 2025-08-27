import os
from typing import List

class Config:
    """Configuration management for Cloud Function - Updated for universal wallet storage"""
    
    def __init__(self):
        self.mongo_uri = os.getenv('MONGO_URI')
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        self.db_name = os.getenv('DB_NAME', 'crypto_tracker')
        self.wallets_collection = os.getenv('WALLETS_COLLECTION', 'smart_wallets')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Function limits
        self.max_wallets = 200  # Increased since we're not filtering by network
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
        
        # Since wallets are universal, we need to be more selective about which ones to analyze
        # for each network to avoid timeout issues
        self.network_wallet_limits = {
            'ethereum': 100,  # Analyze top 100 wallets for Ethereum
            'base': 100       # Analyze top 100 wallets for Base
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        if not self.mongo_uri:
            errors.append("MONGO_URI not configured")
        
        if not self.alchemy_api_key:
            errors.append("ALCHEMY_API_KEY not configured")
        
        return errors