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
from services.database.data_processor import UnifiedDataProcessor
from utils.config import Config

logger = logging.getLogger(__name__)

class CloudBuyAnalyzer:
    """Enhanced Cloud-optimized Buy analyzer with AI-powered alpha scoring and optional storage"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        
        # Use your existing database services
        self.db_service = DatabaseService(self.config)
        self.bigquery_transfer_service = BigQueryTransferService(self.config)
        self.alchemy_service = AlchemyService(self.config)
        
        # Enhanced data processor with AI capabilities
        self.data_processor = UnifiedDataProcessor()
        
        self._initialized = False
        
        # Performance tracking
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "transfers_stored": 0,
            "memory_used_mb": 0.0,
            "zero_eth_count": 0,
            "non_zero_eth_count": 0,
            "ai_enhanced_tokens": 0,
            "ai_confidence_avg": 0.0
        }
        
        logger.info(f"Enhanced CloudBuyAnalyzer created for network: {network}")
    
    async def initialize(self):
        """Initialize all services including AI enhancement"""
        try:
            logger.info(f"=== INITIALIZING Enhanced CloudBuyAnalyzer for {self.network} ===")
            
            # Step 1: Initialize database service (smart_wallets)
            logger.info("Step 1: Initializing DatabaseService...")
            await self.db_service.initialize()
            logger.info("✓ DatabaseService initialized successfully")
            
            # Step 2: Initialize BigQuery transfer service
            logger.info("Step 2: Initializing BigQueryTransferService...")
            await self.bigquery_transfer_service.initialize()
            logger.info("✓ BigQueryTransferService initialized successfully")
            
            # Step 3: Connect data processor to transfer service
            logger.info("Step 3: Connecting data processor...")
            self.data_processor.set_transfer_service(self.bigquery_transfer_service)
            logger.info("✓ Data processor connected")
            
            # Step 4: AI Enhancement note (no more warnings)
            logger.info("Step 4: AI Enhancement ready (lazy loading)...")
            logger.info("✓ AI will be tested when first analysis runs")
            
            self._initialized = True
            logger.info("=== Enhanced CloudBuyAnalyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== Enhanced CloudBuyAnalyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialized = False
            raise
    
    async def analyze(self, num_wallets: int, days_back: float, store_data: bool = False) -> AnalysisResult:
        """Enhanced buy analysis with AI-powered alpha scoring and optional storage"""
        start_time = time.time()
        
        try:
            storage_status = "ENABLED" if store_data else "DISABLED" 
            logger.info("=" * 60)
            logger.info(f"STARTING ENHANCED AI BUY ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info(f"🗄️ Data Storage: {storage_status}")
            logger.info("🤖 AI-Enhanced Alpha Scoring: ENABLED")
            logger.info("=" * 60)
            
            # Step 1: Ensure initialization
            if not self._initialized:
                logger.warning("Analyzer not initialized, attempting initialization...")
                await self.initialize()
                if not self._initialized:
                    logger.error("FATAL: Failed to initialize analyzer")
                    return self._empty_result()
            
            # Step 2: Get wallets from BigQuery
            logger.info("STEP 1: Fetching wallets from BigQuery...")
            step_start = time.time()
            wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
            wallet_time = time.time() - step_start
            
            if not wallets:
                logger.error(f"FATAL: No wallets found for {self.network}")
                return self._empty_result()
            
            logger.info(f"✓ Retrieved {len(wallets)} wallets in {wallet_time:.2f}s")
            logger.info(f"Sample wallet: {wallets[0].address} (score: {wallets[0].score})")
            
            self.stats["wallets_processed"] = len(wallets)
            
            # Step 3: Get block range from Alchemy
            logger.info("STEP 2: Getting block range from Alchemy...")
            step_start = time.time()
            start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
            block_time = time.time() - step_start
            
            if start_block == 0:
                logger.error("FATAL: Failed to get block range")
                return self._empty_result()
            
            logger.info(f"✓ Block range: {start_block} to {end_block} (took {block_time:.2f}s)")
            logger.info(f"Block span: {end_block - start_block} blocks")
            
            # Step 4: Get transfers from Alchemy
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
                return self._empty_result()
            
            self.stats["transfers_processed"] = total_transfers
            
            # Step 5: Process transfers to purchases WITH OPTIONAL STORAGE
            logger.info("STEP 4: Processing transfers to purchases...")
            logger.info(f"🗄️ Storage mode: {storage_status}")
            step_start = time.time()
            
            # PASS THE STORE_DATA FLAG TO THE PROCESSOR
            purchases = await self.data_processor.process_transfers_to_purchases(
                wallets, all_transfers, self.network, store_data=store_data
            )
            
            process_time = time.time() - step_start
            logger.info(f"✓ Transfer processing complete in {process_time:.2f}s")
            logger.info(f"Found {len(purchases) if purchases else 0} purchase transactions")
            
            if not purchases:
                logger.error("FATAL: No purchases found after processing transfers")
                return self._empty_result()
            
            # Log purchase details with ETH calculation results
            zero_eth = sum(1 for p in purchases if p.eth_spent == 0.0)
            non_zero_eth = len(purchases) - zero_eth
            total_eth = sum(p.eth_spent for p in purchases)
            
            logger.info("=== ETH CALCULATION RESULTS ===")
            logger.info(f"Purchases with 0.0 ETH: {zero_eth}")
            logger.info(f"Purchases with >0.0 ETH: {non_zero_eth}")
            logger.info(f"Total ETH spent: {total_eth:.6f}")
            
            if non_zero_eth > 0:
                avg_eth = total_eth / non_zero_eth
                logger.info(f"Average ETH per non-zero purchase: {avg_eth:.6f}")
                
                # Show top purchases
                non_zero_purchases = [p for p in purchases if p.eth_spent > 0]
                non_zero_purchases.sort(key=lambda x: x.eth_spent, reverse=True)
                logger.info("Top 5 ETH purchases:")
                for i, p in enumerate(non_zero_purchases[:5]):
                    logger.info(f"  {i+1}. {p.token_bought}: {p.eth_spent:.6f} ETH")
            
            self.stats["zero_eth_count"] = zero_eth
            self.stats["non_zero_eth_count"] = non_zero_eth
            
            # Validate data quality
            quality_report = await self.data_processor.validate_data_quality(purchases)
            logger.info(f"Data quality score: {quality_report.get('data_quality_score', 0):.2f}")
            if quality_report.get('warnings'):
                for warning in quality_report['warnings']:
                    logger.warning(f"Data quality: {warning}")
            
            # Step 6: ENHANCED AI ANALYSIS
            logger.info("STEP 5: Running AI-Enhanced Analysis...")
            step_start = time.time()
            
            # Use enhanced analysis with AI scoring
            analysis_results = await self.data_processor.analyze_purchases_enhanced(purchases, "buy")
            
            analysis_time = time.time() - step_start
            logger.info(f"✓ AI-Enhanced analysis complete in {analysis_time:.2f}s")
            
            if not analysis_results:
                logger.error("FATAL: Enhanced analysis returned no results")
                return self._empty_result()
            
            # Log AI enhancement details
            if analysis_results.get('enhanced'):
                scores = analysis_results.get('scores', {})
                ai_enhanced_count = sum(1 for s in scores.values() if s.get('ai_enhanced'))
                
                if ai_enhanced_count > 0:
                    confidences = [s.get('confidence', 0) for s in scores.values() if s.get('ai_enhanced')]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    self.stats["ai_enhanced_tokens"] = ai_enhanced_count
                    self.stats["ai_confidence_avg"] = avg_confidence
                    
                    logger.info(f"🤖 AI Enhancement Results:")
                    logger.info(f"  - Tokens with AI scoring: {ai_enhanced_count}/{len(scores)}")
                    logger.info(f"  - Average confidence: {avg_confidence:.2f}")
                    logger.info(f"  - Web3 data integration: ✓")
                    
                    # Log top AI-enhanced token
                    ai_tokens = [(k, v) for k, v in scores.items() if v.get('ai_enhanced')]
                    if ai_tokens:
                        top_ai_token = max(ai_tokens, key=lambda x: x[1]['total_score'])
                        logger.info(f"  - Top AI token: {top_ai_token[0]} (score: {top_ai_token[1]['total_score']:.1f}, confidence: {top_ai_token[1]['confidence']:.2f})")
                        
                        # Log AI component breakdown for top token
                        ai_scores = top_ai_token[1]
                        logger.info(f"  - AI Components: Volume:{ai_scores.get('volume_score', 0):.1f}, Quality:{ai_scores.get('quality_score', 0):.1f}, Momentum:{ai_scores.get('momentum_score', 0):.1f}")
                else:
                    logger.info("🤖 AI Enhancement: Fallback to basic scoring (no AI data available)")
            else:
                logger.info("🤖 AI Enhancement: Using basic scoring method")
            
            # Log comprehensive analysis summary
            self.data_processor.log_token_analysis_summary(purchases, "buy")
            
            # Step 7: Create enhanced result
            self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
            self.stats["analysis_time"] = time.time() - start_time
            
            logger.info("STEP 6: Creating enhanced result...")
            result = self._create_enhanced_result(analysis_results, purchases)
            
            # Final summary with AI enhancement info and storage status
            storage_msg = f"Transfers stored to BigQuery: {self.stats['transfers_stored']}" if store_data else "No data stored (store_data=False)"
            
            logger.info("=" * 60)
            logger.info("🤖 ENHANCED AI BUY ANALYSIS COMPLETE!")
            logger.info(f"Total time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Transactions: {result.total_transactions}")
            logger.info(f"Unique tokens: {result.unique_tokens}")
            logger.info(f"Total ETH value: {result.total_eth_value:.4f}")
            logger.info(f"🗄️ {storage_msg}")
            logger.info(f"AI-enhanced tokens: {self.stats['ai_enhanced_tokens']}")
            if self.stats['ai_enhanced_tokens'] > 0:
                logger.info(f"Average AI confidence: {self.stats['ai_confidence_avg']:.2f}")
            logger.info(f"Top tokens: {len(result.ranked_tokens)}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error("=" * 60)
            logger.error("🤖 ENHANCED AI BUY ANALYSIS FAILED!")
            logger.error(f"Error: {e}")
            logger.error(f"Time elapsed: {self.stats['analysis_time']:.2f}s")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            return self._empty_result()
    
   
    def _create_enhanced_result(self, analysis_results: Dict, purchases: List[Purchase]) -> AnalysisResult:
        """Create enhanced analysis result with AI scoring details - FIXED to generate notifications"""
        logger.info("Creating enhanced AI analysis result...")
        
        if not analysis_results:
            logger.error("Cannot create result - no analysis results")
            return self._empty_result()
        
        scores = analysis_results.get('scores', {})
        if not scores and purchases:
            logger.info("🔧 No AI scores found, creating basic scores for notifications")
            scores = self._create_basic_scores_for_buys(purchases)
            analysis_results['scores'] = scores
        
        is_enhanced = analysis_results.get('enhanced', False)
        
        logger.info(f"Scores available: {len(scores)} tokens")
        logger.info(f"AI enhanced: {is_enhanced}")
        
        # Create ranked tokens with enhanced data
        ranked_tokens = []
        contract_lookup = {p.token_bought: p.web3_analysis.get('contract_address', '') 
                        for p in purchases if p.web3_analysis}
        
        logger.info(f"Contract lookup created for {len(contract_lookup)} tokens")
        
        if len(scores) > 0:
            logger.info("Processing token rankings with contract addresses...")
            
            # Create purchase stats
            purchase_stats = {}
            for purchase in purchases:
                token = purchase.token_bought
                if token not in purchase_stats:
                    purchase_stats[token] = {
                        'total_eth': 0,
                        'count': 0,
                        'wallets': set(),
                        'scores': []
                    }
                purchase_stats[token]['total_eth'] += purchase.eth_spent
                purchase_stats[token]['count'] += 1
                purchase_stats[token]['wallets'].add(purchase.wallet_address)
                purchase_stats[token]['scores'].append(purchase.sophistication_score or 0)
            
            for token, score_data in scores.items():
                # Get stats from purchases
                pstats = purchase_stats.get(token, {
                    'total_eth': 0, 'count': 1, 'wallets': set(['unknown']), 'scores': [0]
                })
                
                # Get contract address
                contract_address = contract_lookup.get(token, '')
                
                # Enhanced token data with contract address
                token_data = {
                    'total_eth_spent': float(pstats['total_eth']),
                    'wallet_count': len(pstats['wallets']),
                    'total_purchases': int(pstats['count']),
                    'avg_wallet_score': float(sum(pstats['scores']) / len(pstats['scores']) if pstats['scores'] else 0),
                    'platforms': ['DEX'],
                    'contract_address': contract_address,
                    'ca': contract_address,  # Alternative field name for notifications
                    'alpha_score': score_data['total_score'],
                    'analysis_type': 'buy',
                    
                    # AI Enhancement indicators
                    'ai_enhanced': score_data.get('ai_enhanced', False),
                    'confidence': score_data.get('confidence', 0.7),
                    
                    # Web3 data for notifications
                    'web3_data': {
                        'contract_address': contract_address,
                        'ca': contract_address,
                        'token_symbol': token,
                        'network': self.network
                    }
                }
                
                # Include all enhanced data in tuple for notifications
                # Format: (token_name, token_data, score, ai_data)
                ranked_tokens.append((token, token_data, score_data['total_score'], score_data))
                
                ca_display = contract_address[:10] + '...' if len(contract_address) > 10 else 'No CA'
                logger.info(f"✅ Added buy token: {token} (Score: {score_data['total_score']:.1f}, CA: {ca_display})")
        else:
            logger.warning("❌ Still no scores available for buy ranking")
        
        # Sort by enhanced score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"🎯 Final ranked buy tokens: {len(ranked_tokens)}")
        
        # Calculate totals
        total_eth = sum(p.eth_spent for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        logger.info(f"Enhanced result summary: {len(purchases)} transactions, {unique_tokens} tokens, {total_eth:.4f} ETH")
        
        # Enhanced performance metrics
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'ai_enhancement_enabled': is_enhanced,
            'data_quality_score': getattr(self.data_processor, '_last_quality_score', 1.0),
            'processing_stats': self.data_processor.get_processing_stats(),
            'buy_analysis_metadata': {
                'avg_eth_per_buy': total_eth / len(purchases) if purchases else 0,
                'buy_signals_detected': len(ranked_tokens),
                'highest_buy_score': ranked_tokens[0][2] if ranked_tokens else 0
            }
        })
        
        return AnalysisResult(
            network=self.network,
            analysis_type="buy",
            total_transactions=len(purchases),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=enhanced_stats,
            web3_enhanced=True
        )
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty result with enhanced metadata"""
        logger.warning("Returning empty enhanced result")
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
    
    def _create_basic_scores_for_buys(self, purchases: List[Purchase]) -> Dict:
        """Create basic scores when AI scoring fails - ensures notifications are sent"""
        logger.info("🔧 Creating basic buy scores to ensure notifications")
        
        scores = {}
        
        # Group purchases by token
        token_groups = {}
        for purchase in purchases:
            token = purchase.token_bought
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(purchase)
        
        for token, token_purchases in token_groups.items():
            # Calculate basic buy metrics
            total_eth_spent = sum(p.eth_spent for p in token_purchases)
            unique_wallets = len(set(p.wallet_address for p in token_purchases))
            avg_wallet_score = sum(p.sophistication_score or 0 for p in token_purchases) / len(token_purchases)
            
            # Basic buy scoring
            volume_score = min(total_eth_spent * 50, 50)       # Up to 50 points for volume
            wallet_score = min(unique_wallets * 8, 30)         # Up to 30 points for wallet diversity
            quality_score = min(avg_wallet_score / 10, 20)     # Up to 20 points for wallet quality
            
            # ETH bonus for significant buys
            eth_bonus = 0
            if total_eth_spent > 5.0:         # 20 points for >5 ETH
                eth_bonus = 20
            elif total_eth_spent > 2.0:       # 15 points for >2 ETH
                eth_bonus = 15
            elif total_eth_spent > 1.0:       # 10 points for >1 ETH
                eth_bonus = 10
            elif total_eth_spent > 0.5:       # 5 points for >0.5 ETH
                eth_bonus = 5
            
            total_score = volume_score + wallet_score + quality_score + eth_bonus
            
            scores[token] = {
                'total_score': float(total_score),
                'volume_score': float(volume_score),
                'wallet_score': float(wallet_score), 
                'quality_score': float(quality_score),
                'ai_enhanced': False,
                'confidence': 0.8,  # High confidence for basic scoring
                'buy_momentum_detected': True
            }
            
            logger.info(f"✅ Basic buy score for {token}: {total_score:.1f} (ETH: {total_eth_spent:.4f}, Wallets: {unique_wallets}, Bonus: {eth_bonus})")
        
        return scores

    async def cleanup(self):
        """Enhanced cleanup with AI resources"""
        try:
            logger.info("Starting enhanced cleanup...")
            
            # Cleanup existing services
            if self.db_service:
                await self.db_service.cleanup()
            
            if self.bigquery_transfer_service:
                await self.bigquery_transfer_service.cleanup()
                
            logger.info("Enhanced CloudBuyAnalyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {e}")
    
    # Additional utility methods
    
    async def get_analysis_health(self) -> Dict:
        """Get analyzer health and status"""
        return {
            'network': self.network,
            'initialized': self._initialized,
            'ai_enhancement_available': self.data_processor._enhanced_scoring_enabled if self.data_processor else False,
            'last_analysis_stats': self.stats,
            'services_status': {
                'database': bool(self.db_service),
                'bigquery': bool(self.bigquery_transfer_service),
                'alchemy': bool(self.alchemy_service),
                'data_processor': bool(self.data_processor)
            }
        }
    
    def log_performance_summary(self):
        """Log detailed performance summary"""
        logger.info("=== BUY ANALYZER PERFORMANCE SUMMARY ===")
        logger.info(f"Network: {self.network}")
        logger.info(f"Analysis time: {self.stats.get('analysis_time', 0):.2f}s")
        logger.info(f"Wallets processed: {self.stats.get('wallets_processed', 0)}")
        logger.info(f"Transfers processed: {self.stats.get('transfers_processed', 0)}")
        logger.info(f"Transfers stored: {self.stats.get('transfers_stored', 0)}")
        logger.info(f"AI enhanced tokens: {self.stats.get('ai_enhanced_tokens', 0)}")
        logger.info(f"AI confidence average: {self.stats.get('ai_confidence_avg', 0):.2f}")
        logger.info("=" * 45)