import os
from typing import List, Dict, Any
import logging 

logger = logging.getLogger(__name__)

class Config:
    """Unified configuration for all services - eliminates duplicate config loading"""
    
    def __init__(self):
        
        # Core projects
        self.bigquery_project_id = os.getenv('BIGQUERY_PROJECT_ID', 'crypto-tracker-cloudrun')
        self.cloud_function_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # BigQuery configuration
        self.bigquery_dataset_id = os.getenv('BIGQUERY_DATASET_ID', 'crypto_analysis')
        self.bigquery_transfers_table = os.getenv('BIGQUERY_TRANSFERS_TABLE', 'transfers')
        self.bigquery_wallets_table = os.getenv('BIGQUERY_WALLETS_TABLE', 'smart_wallets')
        self.bigquery_location = os.getenv('BIGQUERY_LOCATION', 'asia-southeast1')
        
        # API Keys
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        
        # Telegram
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Environment
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # All network configurations in one place
        self.network_config = {
            'ethereum': {
                'chain_id': 1,
                'alchemy_endpoint': f'https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}',
                'explorer': 'etherscan.io',
                'blocks_per_hour': 300,
                'wallet_limit': 100,
                'uniswap_base': 'https://app.uniswap.org/#/swap?outputCurrency=',
                'dexscreener_base': 'https://dexscreener.com/ethereum/',
                'symbol': 'ETH',
                'emoji': '🔷'
            },
            'base': {
                'chain_id': 8453,
                'alchemy_endpoint': f'https://base-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}',
                'explorer': 'basescan.org',
                'blocks_per_hour': 1800,
                'wallet_limit': 100,
                'uniswap_base': 'https://app.uniswap.org/#/swap?chain=base&outputCurrency=',
                'dexscreener_base': 'https://dexscreener.com/base/',
                'symbol': 'ETH',
                'emoji': '🔵'
            }
        }
        
        # Derived configurations for backwards compatibility
        self.supported_networks = list(self.network_config.keys())
        self.supported_analysis_types = ['buy', 'sell']
        
        self.alchemy_endpoints = {
            network: config['alchemy_endpoint'] 
            for network, config in self.network_config.items()
        }
        
        self.blocks_per_hour = {
            network: config['blocks_per_hour']
            for network, config in self.network_config.items()
        }
        
        self.chain_ids = {
            network: config['chain_id']
            for network, config in self.network_config.items()
        }
        
        # API settings
        self.etherscan_endpoint = "https://api.etherscan.io/v2/api"
        self.etherscan_api_rate_limit = 0.2
        
        # Limits
        self.max_wallets = 200
        self.max_days_back = 7.0
        self.timeout_seconds = 300
        
        logger.info(f"Config loaded: BigQuery={self.bigquery_project_id}, Networks={len(self.supported_networks)}")
    
    def get_network_config(self, network: str) -> Dict[str, Any]:
        """Get complete network configuration"""
        return self.network_config.get(network.lower(), self.network_config['ethereum'])
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        if not self.alchemy_api_key:
            errors.append("ALCHEMY_API_KEY not configured")
            
        if not self.bigquery_project_id:
            errors.append("BIGQUERY_PROJECT_ID not configured")
        
        if not self.telegram_bot_token or not self.telegram_chat_id:
            errors.append("Telegram configuration incomplete")
        
        return errors
    
    def is_telegram_configured(self) -> bool:
        """Check Telegram configuration"""
        return bool(self.telegram_bot_token and self.telegram_chat_id and len(self.telegram_bot_token) > 40)