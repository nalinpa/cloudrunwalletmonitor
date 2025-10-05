import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

from services.database.bigquery_client import BigQueryService
from services.blockchain.alchemy_client import AlchemyService
from api.models.data_models import AnalysisResult, WalletInfo, Purchase
from services.database.data_processor import UnifiedDataProcessor
from utils.config import Config

logger = logging.getLogger(__name__)

class UnifiedAnalyzer:
    """Single analyzer for both buy and sell analysis - ALERTS ONLY"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Single BigQuery service for everything
        self.bigquery_service = BigQueryService(self.config)
        self.alchemy_service = AlchemyService(self.config)
        self.data_processor = UnifiedDataProcessor()
        
        self._initialized = False
        
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "ai_enhanced_tokens": 0,
            "ai_confidence_avg": 0.0
        }
        
        logger.info(f"Unified analyzer created for network: {network} (ALERTS ONLY)")
    
    async def initialize(self):
        """Initialize BigQuery service only"""
        try:
            logger.info(f"=== INITIALIZING Unified Analyzer for {self.network} ===")
            
            # Initialize BigQuery service (handles both wallets and alerts)
            await self.bigquery_service.initialize()
            logger.info("✓ BigQuery service initialized (wallets + alerts)")
            
            logger.info("✓ Data processor ready (analysis mode)")
            
            self._initialized = True
            logger.info("=== Unified Analyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== Unified Analyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            self._initialized = False
            raise
    
    async def analyze(self, analysis_type: str, num_wallets: int, days_back: float, 
                     store_alerts: bool = True) -> AnalysisResult:
        """Unified analysis method for both buy and sell"""
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"STARTING {analysis_type.upper()} ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info(f"📊 Alert Storage: {'ENABLED' if store_alerts else 'DISABLED'}")
            logger.info("=" * 60)
            
            if not self._initialized:
                await self.initialize()
            
            # Get wallets
            wallets = await self._get_wallets(num_wallets)
            if not wallets:
                return self._empty_result(analysis_type)
            
            # Get transfers
            all_transfers = await self._get_transfers(wallets, days_back)
            if not self._validate_transfers(all_transfers):
                return self._empty_result(analysis_type)
            
            # Process transfers
            transactions = await self._process_transfers(wallets, all_transfers, analysis_type)
            
            if not transactions:
                return self._empty_result(analysis_type)
            
            # Run AI analysis
            analysis_results = await self.data_processor.analyze_purchases_enhanced(
                transactions, analysis_type
            )
            
            # Track AI stats
            if analysis_results.get('enhanced'):
                scores = analysis_results.get('scores', {})
                ai_enhanced_count = sum(1 for s in scores.values() if s.get('ai_enhanced'))
                if ai_enhanced_count > 0:
                    confidences = [s.get('confidence', 0) for s in scores.values() if s.get('ai_enhanced')]
                    self.stats["ai_enhanced_tokens"] = ai_enhanced_count
                    self.stats["ai_confidence_avg"] = sum(confidences) / len(confidences)
            
            # Create result
            self.stats["analysis_time"] = time.time() - start_time
            result = self._create_result(analysis_results, transactions, analysis_type)
            
            result.performance_metrics['alert_storage_enabled'] = store_alerts
            
            logger.info("=" * 60)
            logger.info(f"{analysis_type.upper()} ANALYSIS COMPLETE!")
            logger.info(f"Time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Transactions: {result.total_transactions}")
            logger.info(f"Tokens: {result.unique_tokens}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error(f"{analysis_type} analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._empty_result(analysis_type)
    
    async def _get_wallets(self, num_wallets: int) -> List[WalletInfo]:
        """Get wallets from BigQuery"""
        step_start = time.time()
        wallets = await self.bigquery_service.get_top_wallets(self.network, num_wallets)
        
        if wallets:
            logger.info(f"✓ Retrieved {len(wallets)} wallets in {time.time() - step_start:.2f}s")
            self.stats["wallets_processed"] = len(wallets)
        
        return wallets
    
    async def _get_transfers(self, wallets: List[WalletInfo], days_back: float) -> Dict:
        """Get transfers from Alchemy"""
        start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
        if start_block == 0:
            logger.error("Failed to get block range")
            return {}
        
        wallet_addresses = [w.address for w in wallets]
        all_transfers = await self.alchemy_service.get_transfers_batch(
            self.network, wallet_addresses, start_block, end_block
        )
        
        total_transfers = sum(
            len(transfers.get('incoming', [])) + len(transfers.get('outgoing', []))
            for transfers in all_transfers.values()
        )
        
        self.stats["transfers_processed"] = total_transfers
        logger.info(f"✓ Processed {total_transfers} transfers")
        
        return all_transfers
    
    def _validate_transfers(self, all_transfers: Dict) -> bool:
        """Validate transfers exist"""
        return bool(all_transfers and self.stats["transfers_processed"] > 0)
    
    async def _process_transfers(self, wallets: List[WalletInfo], all_transfers: Dict, 
                               analysis_type: str) -> List[Purchase]:
        """Process transfers based on analysis type"""
        if analysis_type == 'buy':
            return await self.data_processor.process_transfers_to_purchases(
                wallets, all_transfers, self.network
            )
        else:
            return await self.data_processor.process_transfers_to_sells(
                wallets, all_transfers, self.network
            )
    
    def _create_result(self, analysis_results: Dict, transactions: List[Purchase], 
                      analysis_type: str) -> AnalysisResult:
        """Create analysis result"""
        scores = analysis_results.get('scores', {})
        
        if not scores and transactions:
            scores = self._create_basic_scores(transactions, analysis_type)
            analysis_results['scores'] = scores
        
        ranked_tokens = self._create_ranked_tokens(scores, transactions, analysis_type)
        
        if analysis_type == 'sell':
            total_eth = sum(t.amount_received for t in transactions)
        else:
            total_eth = sum(t.eth_spent for t in transactions)
        
        unique_tokens = len(set(t.token_bought for t in transactions))
        
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'processing_stats': self.data_processor.get_processing_stats(),
            'analysis_metadata': {
                f'avg_eth_per_{analysis_type}': total_eth / len(transactions) if transactions else 0,
                f'{analysis_type}_signals_detected': len(ranked_tokens),
                f'highest_{analysis_type}_score': ranked_tokens[0][2] if ranked_tokens else 0
            }
        })
        
        return AnalysisResult(
            network=self.network,
            analysis_type=analysis_type,
            total_transactions=len(transactions),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=enhanced_stats,
            web3_enhanced=True
        )
    
    def _create_basic_scores(self, transactions: List[Purchase], analysis_type: str) -> Dict:
        """Create basic scores"""
        scores = {}
        token_groups = {}
        
        for transaction in transactions:
            token = transaction.token_bought
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(transaction)
        
        for token, token_transactions in token_groups.items():
            if analysis_type == 'sell':
                total_eth = sum(t.amount_received for t in token_transactions)
                volume_score = min(total_eth * 100, 60)
                eth_bonus = 20 if total_eth > 1.0 else 10 if total_eth > 0.5 else 0
            else:
                total_eth = sum(t.eth_spent for t in token_transactions)
                volume_score = min(total_eth * 50, 50)
                eth_bonus = 20 if total_eth > 5.0 else 15 if total_eth > 2.0 else 0
            
            unique_wallets = len(set(t.wallet_address for t in token_transactions))
            avg_wallet_score = sum(t.sophistication_score or 0 for t in token_transactions) / len(token_transactions)
            
            wallet_score = min(unique_wallets * 8, 30)
            quality_score = min(avg_wallet_score / 10, 20)
            total_score = volume_score + wallet_score + quality_score + eth_bonus
            
            scores[token] = {
                'total_score': float(total_score),
                'volume_score': float(volume_score),
                'wallet_score': float(wallet_score),
                'quality_score': float(quality_score),
                'ai_enhanced': False,
                'confidence': 0.8
            }
        
        return scores
    
    def _create_ranked_tokens(self, scores: Dict, transactions: List[Purchase], analysis_type: str) -> List:
        """Create ranked tokens - simplified version"""
        ranked_tokens = []
        
        for token, score_data in scores.items():
            token_txs = [t for t in transactions if t.token_bought == token]
            
            if analysis_type == 'sell':
                total_eth = sum(t.amount_received for t in token_txs)
            else:
                total_eth = sum(t.eth_spent for t in token_txs)
            
            token_data = {
                'total_eth': float(total_eth),
                'wallet_count': len(set(t.wallet_address for t in token_txs)),
                'tx_count': len(token_txs),
                'ai_enhanced': score_data.get('ai_enhanced', False)
            }
            
            ranked_tokens.append((token, token_data, score_data['total_score'], score_data))
        
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        return ranked_tokens
    
    def _empty_result(self, analysis_type: str) -> AnalysisResult:
        """Empty result"""
        return AnalysisResult(
            network=self.network,
            analysis_type=analysis_type,
            total_transactions=0,
            unique_tokens=0,
            total_eth_value=0.0,
            ranked_tokens=[],
            performance_metrics=self.stats,
            web3_enhanced=True
        )
    
    async def cleanup(self):
        """Cleanup"""
        try:
            if self.bigquery_service:
                await self.bigquery_service.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Backwards compatibility wrappers
class CloudBuyAnalyzer(UnifiedAnalyzer):
    """Buy analyzer wrapper"""
    
    def __init__(self, network: str):
        super().__init__(network)
    
    async def analyze(self, num_wallets: int, days_back: float, store_alerts: bool = True) -> AnalysisResult:
        return await super().analyze('buy', num_wallets, days_back, store_alerts)

class CloudSellAnalyzer(UnifiedAnalyzer):
    """Sell analyzer wrapper"""
    
    def __init__(self, network: str):
        super().__init__(network)
    
    async def analyze(self, num_wallets: int, days_back: float, store_alerts: bool = True) -> AnalysisResult:
        return await super().analyze('sell', num_wallets, days_back, store_alerts)