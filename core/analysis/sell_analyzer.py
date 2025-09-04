import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import logging
import time

# Import your existing services
from services.database.database_client import DatabaseService
from services.database.bigquery_client import BigQueryTransferService
from services.blockchain.alchemy_client import AlchemyService
from api.models.data_models import AnalysisResult, WalletInfo, Purchase
from services.database.data_processor import DataProcessor
from utils.config import Config

logger = logging.getLogger(__name__)

class CloudSellAnalyzer:
    """Cloud-optimized Sell analyzer - Uses your existing BigQuery services"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Use your existing database services
        self.db_service = DatabaseService(self.config)  # Handles smart_wallets table
        self.bigquery_transfer_service = BigQueryTransferService(self.config)  # Handles transfers table
        self.alchemy_service = AlchemyService(self.config)
        
        # Data processor
        self.data_processor = DataProcessor()
        
        self._initialized = False
        
        # Performance tracking
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "transfers_stored": 0,
            "memory_used_mb": 0.0
        }
    
    async def initialize(self):
        """Initialize all services"""
        try:
            logger.info(f"Initializing CloudSellAnalyzer for {self.network}")
            
            # Initialize database service (smart_wallets)
            await self.db_service.initialize()
            logger.info("Database service initialized")
            
            # Initialize BigQuery transfer service
            await self.bigquery_transfer_service.initialize()
            logger.info("BigQuery transfer service initialized")
            
            # Connect data processor to transfer service
            self.data_processor.set_transfer_service(self.bigquery_transfer_service)
            
            self._initialized = True
            logger.info("CloudSellAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"CloudSellAnalyzer initialization failed: {e}")
            self._initialized = False
            raise
    
    async def analyze(self, num_wallets: int, days_back: float) -> AnalysisResult:
        """Main analysis method"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting sell analysis for {self.network}")
            
            # Ensure we're initialized
            if not self._initialized:
                logger.warning("Analyzer not initialized, attempting initialization")
                await self.initialize()
                if not self._initialized:
                    logger.error("Failed to initialize analyzer")
                    return self._empty_result()
            
            # Get wallets from your DatabaseService
            wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
            if not wallets:
                logger.warning(f"No wallets found for {self.network}")
                return self._empty_result()
            
            self.stats["wallets_processed"] = len(wallets)
            logger.info(f"Processing {len(wallets)} wallets for {self.network}")
            
            # Get block range
            start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
            if start_block == 0:
                logger.error("Failed to get block range")
                return self._empty_result()
            
            logger.info(f"Block range: {start_block} to {end_block}")
            
            # Get transfers
            wallet_addresses = [w.address for w in wallets]
            all_transfers = await self.alchemy_service.get_transfers_batch(
                self.network, wallet_addresses, start_block, end_block
            )
            
            # Count total transfers for stats
            total_transfers = 0
            for wallet_transfers in all_transfers.values():
                total_transfers += len(wallet_transfers.get('incoming', []))
                total_transfers += len(wallet_transfers.get('outgoing', []))
            
            self.stats["transfers_processed"] = total_transfers
            logger.info(f"Processing {total_transfers} transfers from {len(wallets)} wallets")
            
            if total_transfers == 0:
                logger.warning("No transfers found")
                return self._empty_result()
            
            # Convert transfers to sells - stores to BigQuery via data_processor
            sells = await self.data_processor.process_transfers_to_sells(
                wallets, all_transfers, self.network
            )
            
            if not sells:
                logger.warning("No sells found after processing transfers")
                return self._empty_result()
            
            # Analyze sells using pandas
            analysis_results = self.data_processor.analyze_purchases(sells, "sell")
            
            # Update stats
            self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
            self.stats["analysis_time"] = time.time() - start_time
            
            logger.info(f"Analysis complete in {self.stats['analysis_time']:.2f}s")
            logger.info(f"Stored {self.stats['transfers_stored']} transfer records to BigQuery")
            
            # Create result
            return self._create_result(analysis_results, sells)
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error(f"Sell analysis failed: {e}")
            return self._empty_result()
    
    def _create_result(self, analysis_results: Dict, sells: List[Purchase]) -> AnalysisResult:
        """Create analysis result for sells"""
        if not analysis_results:
            return self._empty_result()
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        
        # Create ranked tokens
        ranked_tokens = []
        contract_lookup = {s.token_bought: s.web3_analysis.get('contract_address', '') 
                          for s in sells if s.web3_analysis}
        
        if token_stats is not None:
            for token in scores.keys():
                if token in token_stats.index:
                    stats_data = token_stats.loc[token]
                    score_data = scores[token]
                    
                    token_data = {
                        'total_eth_value': float(stats_data['total_value']),
                        'wallet_count': int(stats_data['unique_wallets']),
                        'total_sells': int(stats_data['tx_count']),
                        'avg_wallet_score': float(stats_data['avg_score']),
                        'methods': ['Transfer'],
                        'contract_address': contract_lookup.get(token, ''),
                        'sell_pressure_score': score_data['total_score'],
                        'is_base_native': self.network == 'base'
                    }
                    
                    ranked_tokens.append((token, token_data, score_data['total_score']))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate totals
        total_eth = sum(s.amount_received for s in sells)
        unique_tokens = len(set(s.token_bought for s in sells))
        
        return AnalysisResult(
            network=self.network,
            analysis_type="sell",
            total_transactions=len(sells),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=self.stats,
            web3_enhanced=True
        )
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty result"""
        return AnalysisResult(
            network=self.network,
            analysis_type="sell",
            total_transactions=0,
            unique_tokens=0,
            total_eth_value=0.0,
            ranked_tokens=[],
            performance_metrics=self.stats,
            web3_enhanced=True
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_service:
                await self.db_service.cleanup()
            
            if self.bigquery_transfer_service:
                await self.bigquery_transfer_service.cleanup()
                
            logger.info("CloudSellAnalyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")