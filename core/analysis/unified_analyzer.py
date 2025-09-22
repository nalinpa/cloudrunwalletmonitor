import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

from services.database.database_client import DatabaseService
from services.database.bigquery_client import BigQueryTransferService
from services.blockchain.alchemy_client import AlchemyService
from api.models.data_models import AnalysisResult, WalletInfo, Purchase
from services.database.data_processor import UnifiedDataProcessor
from utils.config import Config

logger = logging.getLogger(__name__)

class UnifiedAnalyzer:
    """Single analyzer for both buy and sell analysis - eliminates 50% duplication"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Shared services - no duplication
        self.db_service = DatabaseService(self.config)
        self.bigquery_transfer_service = BigQueryTransferService(self.config)
        self.alchemy_service = AlchemyService(self.config)
        self.data_processor = UnifiedDataProcessor()
        
        self._initialized = False
        
        # Shared stats tracking
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "transfers_stored": 0,
            "ai_enhanced_tokens": 0,
            "ai_confidence_avg": 0.0
        }
        
        logger.info(f"Unified analyzer created for network: {network}")
    
    async def initialize(self):
        """Single initialization method"""
        try:
            logger.info(f"=== INITIALIZING Unified Analyzer for {self.network} ===")
            
            await self.db_service.initialize()
            await self.bigquery_transfer_service.initialize()
            self.data_processor.set_transfer_service(self.bigquery_transfer_service)
            
            self._initialized = True
            logger.info("=== Unified Analyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== Unified Analyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            self._initialized = False
            raise
    
    async def analyze(self, analysis_type: str, num_wallets: int, days_back: float, store_data: bool = False) -> AnalysisResult:
        """Unified analysis method for both buy and sell"""
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"STARTING UNIFIED {analysis_type.upper()} ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info(f"🗄️ Data Storage: {'ENABLED' if store_data else 'DISABLED'}")
            logger.info("=" * 60)
            
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
            
            # Get wallets (shared step)
            wallets = await self._get_wallets(num_wallets)
            if not wallets:
                return self._empty_result(analysis_type)
            
            # Get transfers (shared step)
            all_transfers = await self._get_transfers(wallets, days_back)
            if not self._validate_transfers(all_transfers):
                return self._empty_result(analysis_type)
            
            # Process transfers based on analysis type
            transactions = await self._process_transfers(
                wallets, all_transfers, analysis_type, store_data
            )
            
            if not transactions:
                return self._empty_result(analysis_type)
            
            # Run AI analysis (shared)
            analysis_results = await self.data_processor.analyze_purchases_enhanced(
                transactions, analysis_type
            )
            
            # Create result
            self.stats["analysis_time"] = time.time() - start_time
            result = self._create_result(analysis_results, transactions, analysis_type)
            
            logger.info("=" * 60)
            logger.info(f"UNIFIED {analysis_type.upper()} ANALYSIS COMPLETE!")
            logger.info(f"Time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Transactions: {result.total_transactions}")
            logger.info(f"Tokens: {result.unique_tokens}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error(f"Unified {analysis_type} analysis failed: {e}")
            return self._empty_result(analysis_type)
    
    async def _get_wallets(self, num_wallets: int) -> List[WalletInfo]:
        """Shared wallet retrieval"""
        step_start = time.time()
        wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
        
        if wallets:
            logger.info(f"✓ Retrieved {len(wallets)} wallets in {time.time() - step_start:.2f}s")
            self.stats["wallets_processed"] = len(wallets)
        
        return wallets
    
    async def _get_transfers(self, wallets: List[WalletInfo], days_back: float) -> Dict:
        """Shared transfer retrieval"""
        # Get block range
        start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
        if start_block == 0:
            logger.error("Failed to get block range")
            return {}
        
        # Get transfers
        wallet_addresses = [w.address for w in wallets]
        all_transfers = await self.alchemy_service.get_transfers_batch(
            self.network, wallet_addresses, start_block, end_block
        )
        
        # Log transfer stats
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
                               analysis_type: str, store_data: bool) -> List[Purchase]:
        """Process transfers based on analysis type"""
        if analysis_type == 'buy':
            return await self.data_processor.process_transfers_to_purchases(
                wallets, all_transfers, self.network, store_data
            )
        else:  # sell
            return await self.data_processor.process_transfers_to_sells(
                wallets, all_transfers, self.network, store_data
            )
    
    def _create_result(self, analysis_results: Dict, transactions: List[Purchase], analysis_type: str) -> AnalysisResult:
        """Shared result creation logic"""
        scores = analysis_results.get('scores', {})
        
        # Ensure basic scores exist
        if not scores and transactions:
            scores = self._create_basic_scores(transactions, analysis_type)
            analysis_results['scores'] = scores
        
        # Create ranked tokens
        ranked_tokens = self._create_ranked_tokens(scores, transactions, analysis_type)
        
        # Calculate totals
        if analysis_type == 'sell':
            total_eth = sum(t.amount_received for t in transactions)
        else:
            total_eth = sum(t.eth_spent for t in transactions)
        
        unique_tokens = len(set(t.token_bought for t in transactions))
        
        # Enhanced stats
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
        """Create basic scores for both buy and sell"""
        scores = {}
        
        # Group by token
        token_groups = {}
        for transaction in transactions:
            token = transaction.token_bought
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(transaction)
        
        for token, token_transactions in token_groups.items():
            if analysis_type == 'sell':
                # Sell scoring
                total_eth = sum(t.amount_received for t in token_transactions)
                volume_score = min(total_eth * 100, 60)
                eth_bonus = 20 if total_eth > 1.0 else 10 if total_eth > 0.5 else 5 if total_eth > 0.1 else 0
            else:
                # Buy scoring  
                total_eth = sum(t.eth_spent for t in token_transactions)
                volume_score = min(total_eth * 50, 50)
                eth_bonus = 20 if total_eth > 5.0 else 15 if total_eth > 2.0 else 10 if total_eth > 1.0 else 5 if total_eth > 0.5 else 0
            
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
                'confidence': 0.8,
                f'{analysis_type}_momentum_detected': True
            }
        
        return scores
    
    def _create_ranked_tokens(self, scores: Dict, transactions: List[Purchase], analysis_type: str) -> List:
        """Create ranked tokens list"""
        ranked_tokens = []
        
        # Build lookups
        contract_lookup = {t.token_bought: t.web3_analysis.get('contract_address', '') 
                          for t in transactions if t.web3_analysis}
        
        # Purchase stats
        purchase_stats = {}
        for transaction in transactions:
            token = transaction.token_bought
            if token not in purchase_stats:
                purchase_stats[token] = {
                    'total_eth': 0, 'count': 0, 'wallets': set(), 'scores': []
                }
            
            if analysis_type == 'sell':
                purchase_stats[token]['total_eth'] += transaction.amount_received
            else:
                purchase_stats[token]['total_eth'] += transaction.eth_spent
                
            purchase_stats[token]['count'] += 1
            purchase_stats[token]['wallets'].add(transaction.wallet_address)
            purchase_stats[token]['scores'].append(transaction.sophistication_score or 0)
        
        # Create ranked results
        for token, score_data in scores.items():
            stats = purchase_stats.get(token, {'total_eth': 0, 'count': 1, 'wallets': set(), 'scores': [0]})
            contract_address = contract_lookup.get(token, '')
            
            # Token data based on analysis type
            if analysis_type == 'sell':
                token_data = {
                    'total_eth_received': float(stats['total_eth']),
                    'total_sells': int(stats['count']),
                    'sell_pressure_score': score_data['total_score'],
                    'analysis_type': 'sell'
                }
            else:
                token_data = {
                    'total_eth_spent': float(stats['total_eth']),
                    'total_purchases': int(stats['count']),
                    'alpha_score': score_data['total_score'],
                    'analysis_type': 'buy'
                }
            
            # Common fields
            token_data.update({
                'wallet_count': len(stats['wallets']),
                'avg_wallet_score': float(sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0),
                'contract_address': contract_address,
                'ca': contract_address,
                'ai_enhanced': score_data.get('ai_enhanced', False),
                'confidence': score_data.get('confidence', 0.75),
                'platforms': ['DEX'],
                'web3_data': {
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'token_symbol': token,
                    'network': self.network
                }
            })
            
            ranked_tokens.append((token, token_data, score_data['total_score'], score_data))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        return ranked_tokens
    
    def _empty_result(self, analysis_type: str) -> AnalysisResult:
        """Shared empty result"""
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
        """Shared cleanup"""
        try:
            if self.db_service:
                await self.db_service.cleanup()
            if self.bigquery_transfer_service:
                await self.bigquery_transfer_service.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_analysis_health(self) -> Dict:
        """Shared health check"""
        return {
            'network': self.network,
            'initialized': self._initialized,
            'ai_enhancement_available': bool(self.data_processor._enhanced_scoring_enabled if self.data_processor else False),
            'last_analysis_stats': self.stats,
            'services_status': {
                'database': bool(self.db_service),
                'bigquery': bool(self.bigquery_transfer_service),
                'alchemy': bool(self.alchemy_service),
                'data_processor': bool(self.data_processor)
            }
        }

# Backwards compatibility - keep same interface
class CloudBuyAnalyzer(UnifiedAnalyzer):
    """Buy-specific wrapper around unified analyzer"""
    
    async def analyze(self, num_wallets: int, days_back: float, store_data: bool = False) -> AnalysisResult:
        return await super().analyze('buy', num_wallets, days_back, store_data)

class CloudSellAnalyzer(UnifiedAnalyzer):
    """Sell-specific wrapper around unified analyzer"""
    
    async def analyze(self, num_wallets: int, days_back: float, store_data: bool = False) -> AnalysisResult:
        return await super().analyze('sell', num_wallets, days_back, store_data)