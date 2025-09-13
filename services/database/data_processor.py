from services.database.data_processor_base import DataProcessor
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import geth_poa_middleware
import asyncio
import aiohttp
from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType

logger = logging.getLogger(__name__)

# Standard ERC20 ABI for token metadata
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]

class Web3DataProcessor(DataProcessor):
    """
    Enhanced Data Processor that EXTENDS your existing DataProcessor
    Inherits ALL existing functionality and adds Web3 features
    """
    
    def __init__(self):
        # Initialize parent class - keeps ALL your existing functionality
        super().__init__()
        
        # Add Web3-specific attributes
        self.w3_connections = {}
        self.token_metadata_cache = {}
        self.contract_verification_cache = {}
        self.honeypot_cache = {}
        self.liquidity_cache = {}
        
        # Web3 configuration
        self.web3_enabled = False
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize Web3 connections
        self._initialize_web3()
        
        logger.info("🔷 Web3 Enhanced Data Processor initialized (extends existing DataProcessor)")
    
    def _initialize_web3(self):
        """Initialize Web3 connections for supported networks"""
        import os
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        
        if not alchemy_key:
            logger.warning("No Alchemy API key - Web3 features disabled")
            self.web3_enabled = False
            return
        
        try:
            # Ethereum connection
            self.w3_connections['ethereum'] = Web3(Web3.HTTPProvider(
                f'https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}'
            ))
            
            # Base connection with POA middleware
            base_w3 = Web3(Web3.HTTPProvider(
                f'https://base-mainnet.g.alchemy.com/v2/{alchemy_key}'
            ))
            base_w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            self.w3_connections['base'] = base_w3
            
            self.web3_enabled = True
            logger.info("✅ Web3 connections established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
            self.web3_enabled = False
    
    async def get_token_metadata_web3(self, contract_address: str, network: str) -> Dict:
        """Get comprehensive token metadata using Web3"""
        
        # Check cache first
        cache_key = f"{network}:{contract_address}"
        if cache_key in self.token_metadata_cache:
            cached = self.token_metadata_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                return cached['data']
        
        # Return empty if Web3 not enabled
        if not self.web3_enabled:
            return {}
        
        w3 = self.w3_connections.get(network)
        if not w3 or not w3.is_connected():
            return {}
        
        try:
            checksum_address = Web3.to_checksum_address(contract_address)
            contract = w3.eth.contract(address=checksum_address, abi=ERC20_ABI)
            
            metadata = {}
            
            # Get token info safely
            try:
                metadata['symbol'] = contract.functions.symbol().call()
            except:
                metadata['symbol'] = 'UNKNOWN'
            
            try:
                metadata['name'] = contract.functions.name().call()
            except:
                metadata['name'] = 'Unknown Token'
            
            try:
                metadata['decimals'] = contract.functions.decimals().call()
            except:
                metadata['decimals'] = 18
            
            try:
                total_supply_raw = contract.functions.totalSupply().call()
                metadata['total_supply'] = total_supply_raw / (10 ** metadata['decimals'])
            except:
                metadata['total_supply'] = 0
            
            # Get contract age
            metadata['contract_age'] = await self._get_contract_age(checksum_address, network, w3)
            
            # Check verification
            metadata['is_verified'] = await self._check_contract_verification(checksum_address, w3)
            
            # Cache the result
            self.token_metadata_cache[cache_key] = {
                'data': metadata,
                'timestamp': datetime.now()
            }
            
            logger.info(f"🔷 Retrieved Web3 metadata for {metadata['symbol']}")
            return metadata
            
        except Exception as e:
            logger.debug(f"Error getting token metadata: {e}")
            return {}
    
    async def _get_contract_age(self, contract_address: str, network: str, w3: Web3) -> Dict:
        """Get contract creation time"""
        try:
            # Simple approximation - check recent blocks
            current_block = w3.eth.block_number
            
            # Check if contract exists
            code = w3.eth.get_code(contract_address)
            if not code or code == b'':
                return {}
            
            # Approximate age (simplified)
            blocks_to_check = 100000  # About 2 weeks on Ethereum
            old_block = max(0, current_block - blocks_to_check)
            
            try:
                old_code = w3.eth.get_code(contract_address, old_block)
                if old_code and old_code != b'':
                    # Contract is older than checked range
                    return {'age_hours': blocks_to_check * 12 / 3600}  # Approximate
            except:
                pass
            
            # Contract is relatively new
            return {'age_hours': 24}  # Default to 1 day
            
        except Exception as e:
            logger.debug(f"Error getting contract age: {e}")
            return {}
    
    async def _check_contract_verification(self, contract_address: str, w3: Web3) -> bool:
        """Check if contract appears to be verified"""
        
        # Check cache
        if contract_address in self.contract_verification_cache:
            return self.contract_verification_cache[contract_address]
        
        try:
            code = w3.eth.get_code(contract_address)
            
            if not code or len(code) < 100:
                verified = False
            else:
                # Simple heuristic - verified contracts tend to be larger
                verified = len(code) > 1000
            
            # Cache result
            self.contract_verification_cache[contract_address] = verified
            return verified
            
        except Exception as e:
            logger.debug(f"Error checking verification: {e}")
            return False
    
    async def check_honeypot_web3(self, contract_address: str, network: str) -> Dict:
        """Check for honeypot characteristics"""
        
        # Check cache
        cache_key = f"{network}:{contract_address}"
        if cache_key in self.honeypot_cache:
            return self.honeypot_cache[cache_key]
        
        if not self.web3_enabled:
            return {'risk_score': 0, 'is_honeypot': False}
        
        w3 = self.w3_connections.get(network)
        if not w3:
            return {'risk_score': 0, 'is_honeypot': False}
        
        try:
            code = w3.eth.get_code(Web3.to_checksum_address(contract_address))
            
            if not code:
                return {'risk_score': 0, 'is_honeypot': False}
            
            code_hex = code.hex().lower()
            risk_score = 0
            flags = []
            
            # Check for red flags
            if 'selfdestruct' in code_hex:
                risk_score += 0.3
                flags.append('Self-destruct')
            
            if 'pause' in code_hex:
                risk_score += 0.2
                flags.append('Pausable')
            
            if 'blacklist' in code_hex or 'whitelist' in code_hex:
                risk_score += 0.3
                flags.append('Access list')
            
            if 'onlyowner' in code_hex:
                risk_score += 0.1
                flags.append('Owner controls')
            
            result = {
                'risk_score': min(risk_score, 1.0),
                'is_honeypot': risk_score >= 0.5,
                'flags': flags
            }
            
            # Cache result
            self.honeypot_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.debug(f"Error checking honeypot: {e}")
            return {'risk_score': 0, 'is_honeypot': False}
    
    async def enrich_purchase_with_web3(self, purchase: Purchase, network: str) -> Purchase:
        """Add Web3 data to a purchase"""
        
        if not self.web3_enabled:
            return purchase
        
        contract_address = purchase.web3_analysis.get('contract_address', '') if purchase.web3_analysis else ''
        
        if not contract_address:
            return purchase
        
        try:
            # Get token metadata
            metadata = await self.get_token_metadata_web3(contract_address, network)
            
            # Check honeypot risk
            honeypot_check = await self.check_honeypot_web3(contract_address, network)
            
            # Update web3_analysis
            if not purchase.web3_analysis:
                purchase.web3_analysis = {}
            
            purchase.web3_analysis.update({
                'token_name': metadata.get('name'),
                'token_symbol': metadata.get('symbol'),
                'decimals': metadata.get('decimals'),
                'total_supply': metadata.get('total_supply'),
                'is_verified': metadata.get('is_verified', False),
                'token_age_hours': metadata.get('contract_age', {}).get('age_hours'),
                'honeypot_risk': honeypot_check.get('risk_score', 0),
                'honeypot_flags': honeypot_check.get('flags', []),
                'is_potential_honeypot': honeypot_check.get('is_honeypot', False),
                'web3_enriched': True,
                'enrichment_timestamp': datetime.now().isoformat()
            })
            
            logger.debug(f"🔷 Enriched {metadata.get('symbol', 'UNKNOWN')} with Web3 data")
            
        except Exception as e:
            logger.error(f"Error enriching with Web3: {e}")
        
        return purchase
    
    # OVERRIDE key methods to add Web3 enrichment
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """
        Enhanced version that adds Web3 enrichment to the existing method
        Calls parent method first, then enriches with Web3 data
        """
        
        # Call parent method to get all existing functionality
        purchases = await super().process_transfers_to_purchases(wallets, all_transfers, network, store_data)
        
        # Add Web3 enrichment if enabled
        if self.web3_enabled and purchases:
            logger.info(f"🔷 Adding Web3 enrichment to {len(purchases)} purchases...")
            
            enriched_purchases = []
            enriched_count = 0
            
            for purchase in purchases:
                enriched = await self.enrich_purchase_with_web3(purchase, network)
                enriched_purchases.append(enriched)
                
                if enriched.web3_analysis and enriched.web3_analysis.get('web3_enriched'):
                    enriched_count += 1
            
            logger.info(f"🔷 Web3 enrichment complete: {enriched_count}/{len(purchases)} enriched")
            return enriched_purchases
        
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """
        Enhanced version for sells with Web3 enrichment
        """
        
        # Call parent method
        sells = await super().process_transfers_to_sells(wallets, all_transfers, network, store_data)
        
        # Add Web3 enrichment if enabled
        if self.web3_enabled and sells:
            logger.info(f"🔷 Adding Web3 enrichment to {len(sells)} sells...")
            
            enriched_sells = []
            enriched_count = 0
            
            for sell in sells:
                enriched = await self.enrich_purchase_with_web3(sell, network)
                enriched_sells.append(enriched)
                
                if enriched.web3_analysis and enriched.web3_analysis.get('web3_enriched'):
                    enriched_count += 1
            
            logger.info(f"🔷 Web3 enrichment complete: {enriched_count}/{len(sells)} enriched")
            return enriched_sells
        
        return sells
    
    async def analyze_purchases_enhanced(self, purchases: List, analysis_type: str) -> Dict:
        """
        Enhanced analysis that includes Web3 data in scoring
        Calls parent method and enhances the scores
        """
        
        # Call parent method for base analysis
        result = await super().analyze_purchases_enhanced(purchases, analysis_type)
        
        # Enhance scores with Web3 data if available
        if self.web3_enabled and result.get('scores'):
            logger.info("🔷 Enhancing scores with Web3 data...")
            
            for token, score_data in result['scores'].items():
                # Find purchases for this token to get Web3 data
                token_purchases = [p for p in purchases if p.token_bought == token]
                
                if token_purchases and token_purchases[0].web3_analysis:
                    web3_data = token_purchases[0].web3_analysis
                    
                    # Adjust score based on Web3 factors
                    honeypot_risk = web3_data.get('honeypot_risk', 0)
                    is_verified = web3_data.get('is_verified', False)
                    token_age_hours = web3_data.get('token_age_hours', 999)
                    
                    # Apply Web3 adjustments
                    if honeypot_risk > 0.5:
                        score_data['total_score'] *= (1 - honeypot_risk * 0.5)  # Penalty
                        score_data['honeypot_warning'] = True
                    
                    if is_verified:
                        score_data['total_score'] *= 1.1  # 10% bonus
                        score_data['verified'] = True
                    
                    if token_age_hours and token_age_hours < 24:
                        score_data['new_token'] = True
                        score_data['token_age_hours'] = token_age_hours
                    
                    # Add Web3 metadata to score
                    score_data['web3_enhanced'] = True
                    score_data['honeypot_risk'] = honeypot_risk
                    score_data['token_name'] = web3_data.get('token_name')
                    score_data['token_symbol'] = web3_data.get('token_symbol')
            
            logger.info("🔷 Web3 score enhancement complete")
        
        return result
    
    def get_processing_stats(self) -> Dict:
        """
        Enhanced stats that includes Web3 metrics
        """
        
        # Get parent stats
        stats = super().get_processing_stats()
        
        # Add Web3 stats
        stats.update({
            'web3_enabled': self.web3_enabled,
            'web3_networks': list(self.w3_connections.keys()) if self.web3_enabled else [],
            'cached_tokens': len(self.token_metadata_cache),
            'cached_verifications': len(self.contract_verification_cache),
            'cached_honeypots': len(self.honeypot_cache),
            'web3_enhancement': 'active' if self.web3_enabled else 'disabled'
        })
        
        return stats
    
    def clear_web3_caches(self):
        """Clear all Web3 caches"""
        self.token_metadata_cache.clear()
        self.contract_verification_cache.clear()
        self.honeypot_cache.clear()
        self.liquidity_cache.clear()
        logger.info("🔷 Web3 caches cleared")
    