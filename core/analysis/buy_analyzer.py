# core/analysis/buy_analyzer.py - Enhanced with detailed logging

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

class CloudBuyAnalyzer:
    """Cloud-optimized Buy analyzer with detailed debug logging"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Use your existing database services
        self.db_service = DatabaseService(self.config)
        self.bigquery_transfer_service = BigQueryTransferService(self.config)
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
        
        logger.info(f"CloudBuyAnalyzer created for network: {network}")
    
    async def initialize(self):
        """Initialize all services with detailed logging"""
        try:
            logger.info(f"=== INITIALIZING CloudBuyAnalyzer for {self.network} ===")
            
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
            logger.info("=== CloudBuyAnalyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== CloudBuyAnalyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialized = False
            raise
    
    async def analyze(self, num_wallets: int, days_back: float) -> AnalysisResult:
        """Main analysis method with detailed step logging"""
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"STARTING BUY ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info("=" * 60)
            
            # Step 1: Ensure initialization
            if not self._initialized:
                logger.warning("Analyzer not initialized, attempting initialization...")
                await self.initialize()
                if not self._initialized:
                    logger.error("FATAL: Failed to initialize analyzer")
                    return self._empty_result()
            
            # Step 2: Get wallets
            logger.info("STEP 1: Fetching wallets from BigQuery...")
            step_start = time.time()
            wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
            wallet_time = time.time() - step_start
            
            if not wallets:
                logger.error(f"FATAL: No wallets found for {self.network}")
                logger.error("Check your smart_wallets table has data")
                return self._empty_result()
            
            logger.info(f"✓ Retrieved {len(wallets)} wallets in {wallet_time:.2f}s")
            logger.info(f"Sample wallet: {wallets[0].address} (score: {wallets[0].score})")
            
            self.stats["wallets_processed"] = len(wallets)
            
            # Step 3: Get block range
            logger.info("STEP 2: Getting block range from Alchemy...")
            step_start = time.time()
            start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
            block_time = time.time() - step_start
            
            if start_block == 0:
                logger.error("FATAL: Failed to get block range")
                return self._empty_result()
            
            logger.info(f"✓ Block range: {start_block} to {end_block} (took {block_time:.2f}s)")
            logger.info(f"Block span: {end_block - start_block} blocks")
            
            # Step 4: Get transfers
            logger.info("STEP 3: Fetching transfers from Alchemy...")
            step_start = time.time()
            wallet_addresses = [w.address for w in wallets]
            logger.info(f"Fetching transfers for {len(wallet_addresses)} wallets...")
            
            all_transfers = await self.alchemy_service.get_transfers_batch(
                self.network, wallet_addresses, start_block, end_block
            )
            transfer_time = time.time() - step_start
            
            # Count and log transfer details
            total_transfers = 0
            incoming_count = 0
            outgoing_count = 0
            wallets_with_transfers = 0
            
            for wallet_addr, transfers in all_transfers.items():
                wallet_incoming = len(transfers.get('incoming', []))
                wallet_outgoing = len(transfers.get('outgoing', []))
                
                if wallet_incoming > 0 or wallet_outgoing > 0:
                    wallets_with_transfers += 1
                
                incoming_count += wallet_incoming
                outgoing_count += wallet_outgoing
                total_transfers += wallet_incoming + wallet_outgoing
            
            logger.info(f"✓ Transfer fetch complete in {transfer_time:.2f}s")
            logger.info(f"Total transfers: {total_transfers}")
            logger.info(f"  - Incoming: {incoming_count}")
            logger.info(f"  - Outgoing: {outgoing_count}")
            logger.info(f"Wallets with transfers: {wallets_with_transfers}/{len(wallets)}")
            
            if total_transfers == 0:
                logger.error("FATAL: No transfers found")
                logger.error(f"Check if wallets are active on {self.network} in the last {days_back} days")
                return self._empty_result()
            
            self.stats["transfers_processed"] = total_transfers
            
            # Step 5: Process transfers to purchases
            logger.info("STEP 4: Processing transfers to purchases...")
            step_start = time.time()
            
            purchases = await self.data_processor.process_transfers_to_purchases(
                wallets, all_transfers, self.network
            )
            
            process_time = time.time() - step_start
            logger.info(f"✓ Transfer processing complete in {process_time:.2f}s")
            logger.info(f"Found {len(purchases) if purchases else 0} purchase transactions")
            
            if not purchases:
                logger.error("FATAL: No purchases found after processing transfers")
                logger.error("This could mean:")
                logger.error("  - No incoming ERC20 transfers found")
                logger.error("  - All tokens were excluded (stablecoins, etc.)")
                logger.error("  - ETH amounts too small (< 0.0005)")
                return self._empty_result()
            
            # Log purchase details
            logger.info("Purchase analysis:")
            total_eth = sum(p.eth_spent for p in purchases)
            tokens = set(p.token_bought for p in purchases)
            logger.info(f"  - Total ETH spent: {total_eth:.4f}")
            logger.info(f"  - Unique tokens: {len(tokens)}")
            logger.info(f"  - Top tokens: {list(tokens)[:5]}")
            
            # Step 6: Analyze purchases
            logger.info("STEP 5: Analyzing purchases with pandas...")
            step_start = time.time()
            
            analysis_results = self.data_processor.analyze_purchases(purchases, "buy")
            
            analysis_time = time.time() - step_start
            logger.info(f"✓ Purchase analysis complete in {analysis_time:.2f}s")
            
            if not analysis_results:
                logger.error("FATAL: Analysis returned no results")
                return self._empty_result()
            
            logger.info("Analysis results:")
            if 'token_stats' in analysis_results:
                token_stats = analysis_results['token_stats']
                logger.info(f"  - Analyzed {len(token_stats)} tokens")
            
            if 'scores' in analysis_results:
                scores = analysis_results['scores']
                logger.info(f"  - Generated scores for {len(scores)} tokens")
                if scores:
                    top_token = max(scores.items(), key=lambda x: x[1]['total_score'])
                    logger.info(f"  - Top token: {top_token[0]} (score: {top_token[1]['total_score']:.1f})")
            
            # Step 7: Update stats and create result
            self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
            self.stats["analysis_time"] = time.time() - start_time
            
            logger.info("STEP 6: Creating final result...")
            result = self._create_result(analysis_results, purchases)
            
            # Final summary
            logger.info("=" * 60)
            logger.info("BUY ANALYSIS COMPLETE!")
            logger.info(f"Total time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Transactions: {result.total_transactions}")
            logger.info(f"Unique tokens: {result.unique_tokens}")
            logger.info(f"Total ETH value: {result.total_eth_value:.4f}")
            logger.info(f"Transfers stored to BigQuery: {self.stats['transfers_stored']}")
            logger.info(f"Top tokens: {len(result.ranked_tokens)}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error("=" * 60)
            logger.error("BUY ANALYSIS FAILED!")
            logger.error(f"Error: {e}")
            logger.error(f"Time elapsed: {self.stats['analysis_time']:.2f}s")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            return self._empty_result()
    
    def _create_result(self, analysis_results: Dict, purchases: List[Purchase]) -> AnalysisResult:
        """Create analysis result with logging"""
        logger.info("Creating final analysis result...")
        
        if not analysis_results:
            logger.error("Cannot create result - no analysis results")
            return self._empty_result()
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        
        logger.info(f"Token stats available: {token_stats is not None}")
        logger.info(f"Scores available: {len(scores)} tokens")
        
        # Create ranked tokens
        ranked_tokens = []
        contract_lookup = {p.token_bought: p.web3_analysis.get('contract_address', '') 
                          for p in purchases if p.web3_analysis}
        
        logger.info(f"Contract lookup created for {len(contract_lookup)} tokens")
        
        if token_stats is not None and len(scores) > 0:
            logger.info("Processing token rankings...")
            for token in scores.keys():
                if token in token_stats.index:
                    stats_data = token_stats.loc[token]
                    score_data = scores[token]
                    
                    token_data = {
                        'total_eth_spent': float(stats_data['total_value']),
                        'wallet_count': int(stats_data['unique_wallets']),
                        'total_purchases': int(stats_data['tx_count']),
                        'avg_wallet_score': float(stats_data['avg_score']),
                        'platforms': ['DEX'],
                        'contract_address': contract_lookup.get(token, ''),
                        'alpha_score': score_data['total_score'],
                        'is_base_native': self.network == 'base'
                    }
                    
                    ranked_tokens.append((token, token_data, score_data['total_score']))
                    logger.info(f"Added token: {token} (score: {score_data['total_score']:.1f})")
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Final ranked tokens: {len(ranked_tokens)}")
        
        # Calculate totals
        total_eth = sum(p.eth_spent for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        logger.info(f"Result summary: {len(purchases)} transactions, {unique_tokens} tokens, {total_eth:.4f} ETH")
        
        return AnalysisResult(
            network=self.network,
            analysis_type="buy",
            total_transactions=len(purchases),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=self.stats,
            web3_enhanced=True
        )
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty result"""
        logger.warning("Returning empty result")
        return AnalysisResult(
            network=self.network,
            analysis_type="buy",
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
            logger.info("Starting cleanup...")
            if self.db_service:
                await self.db_service.cleanup()
            
            if self.bigquery_transfer_service:
                await self.bigquery_transfer_service.cleanup()
                
            logger.info("CloudBuyAnalyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")