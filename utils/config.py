import os
from typing import List, Dict
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
        
        # ============================================================================
        # CENTRALIZED PRICE CONVERSION RATES (UPDATE THESE MANUALLY)
        # ============================================================================
        
        # Current ETH price in USD (update manually or from environment)
        self.eth_price_usd = float(os.getenv('ETH_PRICE_USD', '4500.0'))  # Default fallback
        
        # Token to ETH conversion rates
        # Format: token_symbol -> ETH_per_token
        self.token_to_eth_rates = {
            # Major stablecoins (USD value / ETH price)
            'USDT': 1.0 / self.eth_price_usd,     # $1 / $4500 = ~0.0004 ETH
            'USDC': 1.0 / self.eth_price_usd,     # $1 / $4500 = ~0.0004 ETH  
            'USDC.E': 1.0 / self.eth_price_usd,   # $1 / $4500 = ~0.0004 ETH
            'DAI': 1.0 / self.eth_price_usd,      # $1 / $4500 = ~0.0004 ETH
            'FRAX': 1.0 / self.eth_price_usd,     # $1 / $4500 = ~0.0004 ETH
            'LUSD': 1.0 / self.eth_price_usd,     # $1 / $4500 = ~0.0004 ETH
            'BUSD': 1.0 / self.eth_price_usd,     # $1 / $4500 = ~0.0004 ETH
            
            # ETH variants
            'ETH': 1.0,    # 1:1 with ETH
            'WETH': 1.0,   # 1:1 with ETH
            
            # Base ecosystem tokens (update these based on current prices)
            'AERO': 1.2 / self.eth_price_usd,     # ~$1.20 / $4500 = ~0.0005 ETH
            'CBETH': 0.95,                        # Usually slightly less than ETH
            
            # Add more tokens as needed...
        }
        
        # Token to ETH conversion rates for RECEIVING (selling tokens for ETH)
        # Usually the same as above, but separated for flexibility
        self.receiving_token_to_eth_rates = {
            # Major stablecoins 
            'USDT': 1.0 / self.eth_price_usd,
            'USDC': 1.0 / self.eth_price_usd,
            'USDC.E': 1.0 / self.eth_price_usd,
            'DAI': 1.0 / self.eth_price_usd,
            'FRAX': 1.0 / self.eth_price_usd,
            'LUSD': 1.0 / self.eth_price_usd,
            
            # ETH variants
            'ETH': 1.0,
            'WETH': 1.0,
            
            # Base ecosystem
            'AERO': 1.2 / self.eth_price_usd,
            'CBETH': 0.95,
        }
        
        # Log current rates for visibility
        logger.info(f"ETH Price: ${self.eth_price_usd}")
        logger.info(f"USDT->ETH rate: {self.token_to_eth_rates['USDT']:.6f}")
        logger.info(f"AERO->ETH rate: {self.token_to_eth_rates['AERO']:.6f}")
    
    def get_spending_rate(self, token_symbol: str) -> float:
        """Get ETH conversion rate for spending tokens (buying)"""
        return self.token_to_eth_rates.get(token_symbol.upper(), 0.0)
    
    def get_receiving_rate(self, token_symbol: str) -> float:
        """Get ETH conversion rate for receiving tokens (selling)"""
        return self.receiving_token_to_eth_rates.get(token_symbol.upper(), 0.0)
    
    def update_eth_price(self, new_price: float):
        """Update ETH price and recalculate all USD-based rates"""
        old_price = self.eth_price_usd
        self.eth_price_usd = new_price
        
        # Recalculate USD-based rates
        usd_tokens = ['USDT', 'USDC', 'USDC.E', 'DAI', 'FRAX', 'LUSD', 'BUSD']
        for token in usd_tokens:
            if token in self.token_to_eth_rates:
                self.token_to_eth_rates[token] = 1.0 / new_price
                self.receiving_token_to_eth_rates[token] = 1.0 / new_price
        
        # Update AERO (example - adjust multiplier as needed)
        aero_price_usd = 1.2  # Update this manually
        self.token_to_eth_rates['AERO'] = aero_price_usd / new_price
        self.receiving_token_to_eth_rates['AERO'] = aero_price_usd / new_price
        
        logger.info(f"Updated ETH price: ${old_price} -> ${new_price}")
        logger.info(f"New USDT rate: {self.token_to_eth_rates['USDT']:.6f} ETH")
    
    def add_token_rate(self, token_symbol: str, eth_rate: float):
        """Add or update a token conversion rate"""
        token_symbol = token_symbol.upper()
        self.token_to_eth_rates[token_symbol] = eth_rate
        self.receiving_token_to_eth_rates[token_symbol] = eth_rate
        logger.info(f"Added/updated {token_symbol} rate: {eth_rate:.6f} ETH")
    
    def get_all_supported_tokens(self) -> List[str]:
        """Get list of all tokens with conversion rates"""
        return list(self.token_to_eth_rates.keys())
    
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
        
        # Price validation
        if self.eth_price_usd <= 0:
            errors.append("ETH_PRICE_USD must be positive")
        
        # Check for reasonable ETH price range
        if self.eth_price_usd < 1000 or self.eth_price_usd > 10000:
            errors.append(f"ETH_PRICE_USD seems unrealistic: ${self.eth_price_usd}")
        
        # Optional BigQuery validations with warnings
        if not self.bigquery_dataset_id:
            errors.append("BIGQUERY_DATASET_ID not configured (defaulting to 'crypto_analysis')")
        
        return errors