# core/analysis/sell_analyzer.py - MINIMAL fix, keep your working buy logic intact

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
    """MINIMAL CHANGES - Keep your working buy logic, just fix sell detection"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Use your existing database services
        self.db_service = DatabaseService(self.config)
        self.bigquery_transfer_service = BigQueryTransferService(self.config)
        self.alchemy_service = AlchemyService(self.config)
        
        # Data processor - USE YOUR EXISTING ONE
        self.data_processor = DataProcessor()
        
        self._initialized = False
        
        # Performance tracking
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "transfers_stored": 0,
            "memory_used_mb": 0.0,
            "zero_eth_count": 0,
            "non_zero_eth_count": 0
        }
        
        logger.info(f"CloudSellAnalyzer created for network: {network}")
    
    async def initialize(self):
        """Initialize all services with detailed logging"""
        try:
            logger.info(f"=== INITIALIZING CloudSellAnalyzer for {self.network} ===")
            
            # Initialize database service (smart_wallets)
            logger.info("Step 1: Initializing DatabaseService...")
            await self.db_service.initialize()
            logger.info("✓ DatabaseService initialized successfully")
            
            # Initialize BigQuery transfer service
            logger.info("Step 2: Initializing BigQueryTransferService...")
            await self.bigquery_transfer_service.initialize()
            logger.info("✓ BigQueryTransferService initialized successfully")
            
            # Connect data processor to transfer service
            logger.info("Step 3: Connecting data processor...")
            self.data_processor.set_transfer_service(self.bigquery_transfer_service)
            logger.info("✓ Data processor connected")
            
            self._initialized = True
            logger.info("=== CloudSellAnalyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== CloudSellAnalyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialized = False
            raise
    
    async def analyze(self, num_wallets: int, days_back: float) -> AnalysisResult:
        """SELL analysis - use your existing transfer processing but focus on sells"""
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"STARTING SELL ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info("=" * 60)
            
            # Step 1: Ensure initialization
            if not self._initialized:
                logger.warning("Analyzer not initialized, attempting initialization...")
                await self.initialize()
                if not self._initialized:
                    logger.error("FATAL: Failed to initialize analyzer")
                    return self._empty_result()
            
            # Step 2: Get wallets - SAME AS YOUR BUY ANALYZER
            logger.info("STEP 1: Fetching wallets from BigQuery...")
            step_start = time.time()
            wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
            wallet_time = time.time() - step_start
            
            if not wallets:
                logger.error(f"FATAL: No wallets found for {self.network}")
                return self._empty_result()
            
            logger.info(f"✓ Retrieved {len(wallets)} wallets in {wallet_time:.2f}s")
            self.stats["wallets_processed"] = len(wallets)
            
            # Step 3: Get block range - SAME AS YOUR BUY ANALYZER
            logger.info("STEP 2: Getting block range from Alchemy...")
            step_start = time.time()
            start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
            block_time = time.time() - step_start
            
            if start_block == 0:
                logger.error("FATAL: Failed to get block range")
                return self._empty_result()
            
            logger.info(f"✓ Block range: {start_block} to {end_block} (took {block_time:.2f}s)")
            
            # Step 4: Get transfers - SAME AS YOUR BUY ANALYZER
            logger.info("STEP 3: Fetching transfers from Alchemy...")
            step_start = time.time()
            wallet_addresses = [w.address for w in wallets]
            
            all_transfers = await self.alchemy_service.get_transfers_batch(
                self.network, wallet_addresses, start_block, end_block
            )
            transfer_time = time.time() - step_start
            
            # Count transfers - SAME AS YOUR BUY ANALYZER
            total_transfers = 0
            incoming_count = 0
            outgoing_count = 0
            
            for wallet_addr, transfers in all_transfers.items():
                incoming_count += len(transfers.get('incoming', []))
                outgoing_count += len(transfers.get('outgoing', []))
                total_transfers += len(transfers.get('incoming', [])) + len(transfers.get('outgoing', []))
            
            logger.info(f"✓ Transfer fetch complete in {transfer_time:.2f}s")
            logger.info(f"Total transfers: {total_transfers} (Incoming: {incoming_count}, Outgoing: {outgoing_count})")
            
            if total_transfers == 0:
                logger.error("FATAL: No transfers found")
                return self._empty_result()
            
            self.stats["transfers_processed"] = total_transfers
            
            # Step 5: Process transfers - USE YOUR EXISTING METHOD BUT RETURN SELLS
            logger.info("STEP 4: Processing transfers to sells...")
            step_start = time.time()
            
            # USE YOUR EXISTING process_transfers_to_sells method
            sells = await self.data_processor.process_transfers_to_sells(
                wallets, all_transfers, self.network
            )
            
            process_time = time.time() - step_start
            logger.info(f"✓ Sell processing complete in {process_time:.2f}s")
            logger.info(f"Found {len(sells) if sells else 0} sell transactions")
            
            if not sells:
                logger.error("FATAL: No sells found after processing transfers")
                return self._empty_result()
            
            # Log sell details
            total_eth_received = sum(s.amount_received for s in sells)
            tokens = set(s.token_bought for s in sells)
            logger.info(f"Sell summary: {total_eth_received:.4f} ETH received, {len(tokens)} unique tokens sold")
            
            # Step 6: Analyze sells - USE YOUR EXISTING ANALYSIS METHOD
            logger.info("STEP 5: Analyzing sells...")
            step_start = time.time()
            
            analysis_results = self.data_processor.analyze_purchases(sells, "sell")
            
            analysis_time = time.time() - step_start
            logger.info(f"✓ Sell analysis complete in {analysis_time:.2f}s")
            
            if not analysis_results:
                logger.error("FATAL: Analysis returned no results")
                return self._empty_result()
            
            # Step 7: Create result - SAME STRUCTURE AS YOUR BUY ANALYZER
            self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
            self.stats["analysis_time"] = time.time() - start_time
            
            result = self._create_result(analysis_results, sells)
            
            logger.info("=" * 60)
            logger.info("SELL ANALYSIS COMPLETE!")
            logger.info(f"Total time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Sell transactions: {result.total_transactions}")
            logger.info(f"Unique tokens: {result.unique_tokens}")
            logger.info(f"Total ETH received: {result.total_eth_value:.4f}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error("SELL ANALYSIS FAILED!")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result()
    
    def _create_result(self, analysis_results: Dict, sells: List[Purchase]) -> AnalysisResult:
        """Create sell analysis result - SAME STRUCTURE AS YOUR BUY ANALYZER"""
        logger.info("Creating final sell analysis result...")
        
        if not analysis_results:
            logger.error("Cannot create result - no analysis results")
            return self._empty_result()
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        
        logger.info(f"Token stats available: {token_stats is not None}")
        logger.info(f"Scores available: {len(scores)} tokens")
        
        # Create ranked tokens - SAME AS YOUR BUY ANALYZER
        ranked_tokens = []
        contract_lookup = {s.token_bought: s.web3_analysis.get('contract_address', '') 
                          for s in sells if s.web3_analysis}
        
        logger.info(f"Contract lookup created for {len(contract_lookup)} tokens")
        
        if token_stats is not None and len(scores) > 0:
            logger.info("Processing token sell pressure rankings...")
            for token in scores.keys():
                if token in token_stats.index:
                    stats_data = token_stats.loc[token]
                    score_data = scores[token]
                    
                    # MINIMAL CHANGES - just rename fields for sell context
                    token_data = {
                        'total_eth_received': float(stats_data['total_value']),  # Changed from eth_spent
                        'wallet_count': int(stats_data['unique_wallets']),
                        'total_sells': int(stats_data['tx_count']),  # Changed from purchases
                        'avg_wallet_score': float(stats_data['avg_score']),
                        'platforms': ['DEX'],
                        'contract_address': contract_lookup.get(token, ''),
                        'sell_pressure_score': score_data['total_score'],  # Renamed for context
                        'is_base_native': self.network == 'base'
                    }
                    
                    ranked_tokens.append((token, token_data, score_data['total_score']))
                    logger.debug(f"Added sell token: {token} (pressure score: {score_data['total_score']:.1f})")
        
        # Sort by score (higher score = more sell pressure)
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Final ranked sell tokens: {len(ranked_tokens)}")
        
        # Calculate totals - FOR SELLS use amount_received as the ETH value
        total_eth = sum(s.amount_received for s in sells)  # This is the key difference
        unique_tokens = len(set(s.token_bought for s in sells))
        
        logger.info(f"Sell result: {len(sells)} transactions, {unique_tokens} tokens, {total_eth:.4f} ETH received")
        
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
        """Return empty sell result"""
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