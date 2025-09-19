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

class CloudSellAnalyzer:
    """Enhanced Cloud-optimized Sell analyzer with AI-powered alpha scoring and optional storage"""
    
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
        
        logger.info(f"Enhanced CloudSellAnalyzer created for network: {network}")
    
    async def initialize(self):
        """Initialize all services including AI enhancement"""
        try:
            logger.info(f"=== INITIALIZING Enhanced CloudSellAnalyzer for {self.network} ===")
            
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
            
            # Step 4: AI Enhancement note (no more warnings) - UPDATED
            logger.info("Step 4: AI Enhancement ready (lazy loading)...")
            logger.info("✓ AI will be tested when first sell analysis runs")
            
            self._initialized = True
            logger.info("=== Enhanced CloudSellAnalyzer initialization COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== Enhanced CloudSellAnalyzer initialization FAILED ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialized = False
            raise
    
    async def analyze(self, num_wallets: int, days_back: float, store_data: bool = False) -> AnalysisResult:
        """Enhanced sell analysis with AI-powered alpha scoring and optional storage"""
        start_time = time.time()
        
        try:
            storage_status = "ENABLED" if store_data else "DISABLED"
            logger.info("=" * 60)
            logger.info(f"STARTING ENHANCED AI SELL ANALYSIS FOR {self.network.upper()}")
            logger.info(f"Parameters: {num_wallets} wallets, {days_back} days back")
            logger.info(f"🗄️ Data Storage: {storage_status}")
            logger.info("🤖 AI-Enhanced Sell Pressure Analysis: ENABLED")
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
            logger.info(f"  - Outgoing: {outgoing_count} (potential sells)")
            logger.info(f"Wallets with transfers: {wallets_with_transfers}/{len(wallets)}")
            
            if total_transfers == 0:
                logger.error("FATAL: No transfers found")
                return self._empty_result()
            
            self.stats["transfers_processed"] = total_transfers
            
            # Step 5: Process transfers to sells WITH OPTIONAL STORAGE
            logger.info("STEP 4: Processing transfers to identify sells...")
            logger.info(f"🗄️ Storage mode: {storage_status}")
            step_start = time.time()
            
            # PASS THE STORE_DATA FLAG TO THE PROCESSOR
            sells = await self.data_processor.process_transfers_to_sells(
                wallets, all_transfers, self.network, store_data=store_data
            )
            
            process_time = time.time() - step_start
            logger.info(f"✓ Sell processing complete in {process_time:.2f}s")
            logger.info(f"Found {len(sells) if sells else 0} sell transactions")
            
            if not sells:
                logger.error("FATAL: No sells found after processing transfers")
                logger.error("This could indicate:")
                logger.error("  - No outgoing ERC20 transfers detected")
                logger.error("  - ETH received calculation not working")
                logger.error("  - All sells below minimum threshold")
                return self._empty_result()
            
            # Log sell details with ETH calculation results
            zero_eth = sum(1 for s in sells if s.amount_received == 0.0)
            non_zero_eth = len(sells) - zero_eth
            total_eth_received = sum(s.amount_received for s in sells)
            
            logger.info("=== SELL ETH RECEIVED RESULTS ===")
            logger.info(f"Sells with 0.0 ETH received: {zero_eth}")
            logger.info(f"Sells with >0.0 ETH received: {non_zero_eth}")
            logger.info(f"Total ETH received: {total_eth_received:.6f}")
            
            if non_zero_eth > 0:
                avg_eth = total_eth_received / non_zero_eth
                logger.info(f"Average ETH per non-zero sell: {avg_eth:.6f}")
                
                # Show top sells
                non_zero_sells = [s for s in sells if s.amount_received > 0]
                non_zero_sells.sort(key=lambda x: x.amount_received, reverse=True)
                logger.info("Top 5 ETH sells:")
                for i, s in enumerate(non_zero_sells[:5]):
                    amount_sold = s.web3_analysis.get('amount_sold', 0) if s.web3_analysis else 0
                    logger.info(f"  {i+1}. {s.token_bought}: {s.amount_received:.6f} ETH (sold: {amount_sold} tokens)")
            
            self.stats["zero_eth_count"] = zero_eth
            self.stats["non_zero_eth_count"] = non_zero_eth
            
            # Validate data quality
            quality_report = await self.data_processor.validate_data_quality(sells)
            logger.info(f"Sell data quality score: {quality_report.get('data_quality_score', 0):.2f}")
            if quality_report.get('warnings'):
                for warning in quality_report['warnings']:
                    logger.warning(f"Sell data quality: {warning}")
            
            # Step 6: ENHANCED AI ANALYSIS FOR SELL PRESSURE
            logger.info("STEP 5: Running AI-Enhanced Sell Pressure Analysis...")
            step_start = time.time()
            
            # Use enhanced analysis with AI scoring for sell pressure
            analysis_results = await self.data_processor.analyze_purchases_enhanced(sells, "sell")
            
            analysis_time = time.time() - step_start
            logger.info(f"✓ AI-Enhanced sell analysis complete in {analysis_time:.2f}s")
            
            if not analysis_results:
                logger.error("FATAL: Enhanced sell analysis returned no results")
                return self._empty_result()
            
            # Log AI enhancement details for sell pressure
            if analysis_results.get('enhanced'):
                scores = analysis_results.get('scores', {})
                ai_enhanced_count = sum(1 for s in scores.values() if s.get('ai_enhanced'))
                
                if ai_enhanced_count > 0:
                    confidences = [s.get('confidence', 0) for s in scores.values() if s.get('ai_enhanced')]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    self.stats["ai_enhanced_tokens"] = ai_enhanced_count
                    self.stats["ai_confidence_avg"] = avg_confidence
                    
                    logger.info(f"🤖 AI Sell Pressure Enhancement Results:")
                    logger.info(f"  - Tokens with AI sell analysis: {ai_enhanced_count}/{len(scores)}")
                    logger.info(f"  - Average confidence: {avg_confidence:.2f}")
                    logger.info(f"  - Web3 data integration: ✓")
                    
                    # Log top AI-enhanced selling pressure token
                    ai_tokens = [(k, v) for k, v in scores.items() if v.get('ai_enhanced')]
                    if ai_tokens:
                        top_sell_pressure = max(ai_tokens, key=lambda x: x[1]['total_score'])
                        logger.info(f"  - Highest sell pressure: {top_sell_pressure[0]} (score: {top_sell_pressure[1]['total_score']:.1f}, confidence: {top_sell_pressure[1]['confidence']:.2f})")
                        
                        # Log AI component breakdown for top selling pressure
                        ai_scores = top_sell_pressure[1]
                        web3_data = ai_scores.get('web3_data', {}) if 'web3_data' in ai_scores else {}
                        if web3_data.get('smart_money_percentage'):
                            smart_money_pct = web3_data['smart_money_percentage'] * 100
                            logger.info(f"  - Smart money selling: {smart_money_pct:.0f}%")
                        if web3_data.get('token_age_hours'):
                            age = web3_data['token_age_hours']
                            logger.info(f"  - Token age: {age:.1f}h ({age/24:.1f}d)")
                else:
                    logger.info("🤖 AI Sell Enhancement: Fallback to basic scoring (no AI data available)")
            else:
                logger.info("🤖 AI Sell Enhancement: Using basic sell pressure scoring")
            
            # Log comprehensive sell analysis summary
            self.data_processor.log_token_analysis_summary(sells, "sell")
            
            # Step 7: Create enhanced sell result
            self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
            self.stats["analysis_time"] = time.time() - start_time
            
            logger.info("STEP 6: Creating enhanced sell result...")
            result = self._create_enhanced_result(analysis_results, sells)
            
            # Final summary with AI enhancement info and storage status
            storage_msg = f"Transfers stored to BigQuery: {self.stats['transfers_stored']}" if store_data else "No data stored (store_data=False)"
            
            logger.info("=" * 60)
            logger.info("🤖 ENHANCED AI SELL ANALYSIS COMPLETE!")
            logger.info(f"Total time: {self.stats['analysis_time']:.2f}s")
            logger.info(f"Sell transactions: {result.total_transactions}")
            logger.info(f"Unique tokens under pressure: {result.unique_tokens}")
            logger.info(f"Total ETH received from sells: {result.total_eth_value:.4f}")
            logger.info(f"🗄️ {storage_msg}")
            logger.info(f"AI-enhanced tokens: {self.stats['ai_enhanced_tokens']}")
            if self.stats['ai_enhanced_tokens'] > 0:
                logger.info(f"Average AI confidence: {self.stats['ai_confidence_avg']:.2f}")
            logger.info(f"Sell pressure alerts: {len(result.ranked_tokens)}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error("=" * 60)
            logger.error("🤖 ENHANCED AI SELL ANALYSIS FAILED!")
            logger.error(f"Error: {e}")
            logger.error(f"Time elapsed: {self.stats['analysis_time']:.2f}s")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            return self._empty_result()
    
    def _create_enhanced_result(self, analysis_results: Dict, sells: List[Purchase]) -> AnalysisResult:
        """Create enhanced sell analysis result with AI scoring details - NO HOLDER DATA"""
        logger.info("Creating enhanced AI sell analysis result...")
        
        if not analysis_results:
            logger.error("Cannot create result - no analysis results")
            return self._empty_result()
        
        scores = analysis_results.get('scores', {})
        if not scores and sells:
            logger.info("🔧 No AI scores found, creating basic scores for notifications")
            scores = self._create_basic_scores_for_sells(sells)
            analysis_results['scores'] = scores
        
        is_enhanced = analysis_results.get('enhanced', False)
        
        logger.info(f"Scores available: {len(scores)} tokens")
        logger.info(f"AI enhanced: {is_enhanced}")
        
        # Create ranked tokens with enhanced sell pressure data
        ranked_tokens = []
        contract_lookup = {s.token_bought: s.web3_analysis.get('contract_address', '') 
                        for s in sells if s.web3_analysis}
        
        # Create token age lookup
        token_age_lookup = {}
        for sell in sells:
            token = sell.token_bought
            if sell.web3_analysis and sell.web3_analysis.get('token_age_hours'):
                token_age_lookup[token] = sell.web3_analysis['token_age_hours']
        
        logger.info(f"Contract lookup created for {len(contract_lookup)} tokens")
        logger.info(f"Token age data found for {len(token_age_lookup)} tokens")
        
        if len(scores) > 0:
            logger.info("Processing token sell pressure rankings with contract addresses and Web3 data...")
            
            # Create sell stats
            sell_stats = {}
            for sell in sells:
                token = sell.token_bought
                if token not in sell_stats:
                    sell_stats[token] = {
                        'total_eth': 0,
                        'count': 0,
                        'wallets': set(),
                        'scores': [],
                        'tokens_sold': 0
                    }
                sell_stats[token]['total_eth'] += sell.amount_received  # ETH received from sells
                sell_stats[token]['count'] += 1
                sell_stats[token]['wallets'].add(sell.wallet_address)
                sell_stats[token]['scores'].append(sell.sophistication_score or 0)
                if sell.web3_analysis:
                    sell_stats[token]['tokens_sold'] += sell.web3_analysis.get('amount_sold', 0)
            
            for token, score_data in scores.items():
                # Get stats from sells
                sstats = sell_stats.get(token, {
                    'total_eth': 0, 'count': 1, 'wallets': set(['unknown']), 'scores': [0], 'tokens_sold': 0
                })
                
                # Get contract address and age
                contract_address = contract_lookup.get(token, '')
                token_age_hours = token_age_lookup.get(token)
                
                # Enhanced token data with AI sell pressure metrics
                token_data = {
                    'total_eth_received': float(sstats['total_eth']),
                    'wallet_count': len(sstats['wallets']),
                    'total_sells': int(sstats['count']),
                    'avg_wallet_score': float(sum(sstats['scores']) / len(sstats['scores']) if sstats['scores'] else 0),
                    'total_tokens_sold': float(sstats['tokens_sold']),
                    'avg_sell_size': float(sstats['total_eth'] / sstats['count'] if sstats['count'] > 0 else 0),
                    'platforms': ['Transfer'],
                    'contract_address': contract_address,
                    'ca': contract_address,  # Alternative field name for notifications
                    'sell_pressure_score': score_data['total_score'],
                    'analysis_type': 'sell',
                    
                    # AI Enhancement data
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
                
                # Add token age if available
                if token_age_hours is not None:
                    token_data['token_age_hours'] = float(token_age_hours)
                    token_data['token_age_days'] = float(token_age_hours / 24)
                    
                    # Age classification for sell pressure context
                    if token_age_hours < 1:
                        token_data['age_classification'] = 'BRAND_NEW'
                        token_data['sell_urgency'] = 'CRITICAL'  # New tokens dumping is very concerning
                    elif token_age_hours < 24:
                        token_data['age_classification'] = 'FRESH'
                        token_data['sell_urgency'] = 'HIGH'
                    elif token_age_hours < 168:  # 1 week
                        token_data['age_classification'] = 'RECENT'
                        token_data['sell_urgency'] = 'MEDIUM'
                    elif token_age_hours < 720:  # 1 month
                        token_data['age_classification'] = 'ESTABLISHED'
                        token_data['sell_urgency'] = 'NORMAL'
                    else:
                        token_data['age_classification'] = 'MATURE'
                        token_data['sell_urgency'] = 'NORMAL'
                    
                    # Add age data to web3_data
                    token_data['web3_data']['token_age_hours'] = token_age_hours
                    token_data['web3_data']['token_age_days'] = token_age_hours / 24
                    token_data['web3_data']['age_classification'] = token_data['age_classification']
                    token_data['web3_data']['sell_urgency'] = token_data['sell_urgency']
                
                # Calculate sell pressure intensity
                eth_per_sell = sstats['total_eth'] / sstats['count'] if sstats['count'] > 0 else 0
                if eth_per_sell > 5.0:
                    token_data['sell_intensity'] = 'MASSIVE'
                elif eth_per_sell > 2.0:
                    token_data['sell_intensity'] = 'LARGE'
                elif eth_per_sell > 1.0:
                    token_data['sell_intensity'] = 'MODERATE'
                elif eth_per_sell > 0.1:
                    token_data['sell_intensity'] = 'SMALL'
                else:
                    token_data['sell_intensity'] = 'MINIMAL'
                
                # Include all enhanced data in tuple for notifications
                # Format: (token_name, token_data, score, ai_data)
                ai_data_with_web3 = score_data.copy()
                
                # Add Web3 intelligence from any sell of this token
                for sell in sells:
                    if sell.token_bought == token and sell.web3_analysis:
                        ai_data_with_web3.update(sell.web3_analysis)
                        break
                
                ranked_tokens.append((token, token_data, score_data['total_score'], ai_data_with_web3))
                
                # Enhanced logging with age and sell pressure info
                ca_display = contract_address[:10] + '...' if len(contract_address) > 10 else 'No CA'
                age_display = f", {token_age_hours/24:.1f}d old" if token_age_hours else ""
                intensity_display = f", {token_data['sell_intensity']} sells"
                logger.info(f"✅ Added sell pressure token: {token} (Score: {score_data['total_score']:.1f}, CA: {ca_display}{age_display}{intensity_display})")
        else:
            logger.warning("❌ Still no scores available for sell pressure ranking")
        
        # Sort by sell pressure score (higher = more selling pressure)
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"🎯 Final ranked sell pressure tokens: {len(ranked_tokens)}")
        
        # Calculate totals for sells
        total_eth_received = sum(s.amount_received for s in sells)
        unique_tokens = len(set(s.token_bought for s in sells))
        
        logger.info(f"Enhanced sell result: {len(sells)} sells, {unique_tokens} tokens, {total_eth_received:.4f} ETH received")
        
        # Enhanced performance metrics for sells
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'ai_enhancement_enabled': is_enhanced,
            'data_quality_score': getattr(self.data_processor, '_last_quality_score', 1.0),
            'processing_stats': self.data_processor.get_processing_stats() if self.data_processor else {},
            'sell_analysis_metadata': {
                'avg_eth_per_sell': total_eth_received / len(sells) if sells else 0,
                'sell_pressure_detected': len(ranked_tokens),
                'highest_pressure_score': ranked_tokens[0][2] if ranked_tokens else 0,
                'tokens_with_age_data': len(token_age_lookup),
                'tokens_with_contracts': len(contract_lookup),
                'sell_intensity_breakdown': {
                    'massive': len([t for _, t, _, _ in ranked_tokens if t.get('sell_intensity') == 'MASSIVE']),
                    'large': len([t for _, t, _, _ in ranked_tokens if t.get('sell_intensity') == 'LARGE']),
                    'moderate': len([t for _, t, _, _ in ranked_tokens if t.get('sell_intensity') == 'MODERATE']),
                    'small': len([t for _, t, _, _ in ranked_tokens if t.get('sell_intensity') == 'SMALL']),
                    'minimal': len([t for _, t, _, _ in ranked_tokens if t.get('sell_intensity') == 'MINIMAL'])
                }
            }
        })
        
        return AnalysisResult(
            network=self.network,
            analysis_type="sell",
            total_transactions=len(sells),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth_received,
            ranked_tokens=ranked_tokens,
            performance_metrics=enhanced_stats,
            web3_enhanced=True
        )
    
    def _create_basic_scores_for_sells(self, sells: List[Purchase]) -> Dict:
        """Create basic scores when AI scoring fails - ensures notifications are sent"""
        logger.info("🔧 Creating basic sell pressure scores to ensure notifications")
        
        scores = {}
        
        # Group sells by token
        token_groups = {}
        for sell in sells:
            token = sell.token_bought
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(sell)
        
        for token, token_sells in token_groups.items():
            # Calculate basic sell pressure metrics
            total_eth_received = sum(s.amount_received for s in token_sells)
            unique_wallets = len(set(s.wallet_address for s in token_sells))
            avg_wallet_score = sum(s.sophistication_score or 0 for s in token_sells) / len(token_sells)
            
            # Basic sell pressure scoring - Your BERRY would get ~48 points
            volume_score = min(total_eth_received * 15, 40)    # 1.2021 * 15 = 18 points
            wallet_score = min(unique_wallets * 10, 30)        # 1 * 10 = 10 points  
            quality_score = min(avg_wallet_score / 10, 20)     # Wallet quality points
            
            # ETH bonus for significant sells
            eth_bonus = 0
            if total_eth_received > 1.0:      # 20 points for >1 ETH (BERRY qualifies!)
                eth_bonus = 20
            elif total_eth_received > 0.5:    # 10 points for >0.5 ETH
                eth_bonus = 10
            elif total_eth_received > 0.1:    # 5 points for >0.1 ETH
                eth_bonus = 5
            
            total_score = volume_score + wallet_score + quality_score + eth_bonus
            
            scores[token] = {
                'total_score': float(total_score),
                'volume_score': float(volume_score),
                'wallet_score': float(wallet_score), 
                'quality_score': float(quality_score),
                'ai_enhanced': False,
                'confidence': 0.8,  # High confidence for basic scoring
                'sell_pressure_detected': True
            }
            
            logger.info(f"✅ Basic sell score for {token}: {total_score:.1f} (ETH: {total_eth_received:.4f}, Wallets: {unique_wallets}, Bonus: {eth_bonus})")
        
        return scores

    def _calculate_sell_momentum(self, score_data: Dict) -> float:
        """Calculate sell momentum based on AI scores"""
        momentum_score = score_data.get('momentum_score', 0)
        volume_score = score_data.get('volume_score', 0)
        
        # Higher volume + momentum = stronger sell pressure
        sell_momentum = (momentum_score + volume_score) / 200  # Normalize to 0-1
        return min(sell_momentum, 1.0)
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty sell result with enhanced metadata"""
        logger.warning("Returning empty enhanced sell result")
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
        """Enhanced cleanup with AI resources"""
        try:
            logger.info("Starting enhanced sell analyzer cleanup...")
            
            # Cleanup existing services
            if self.db_service:
                await self.db_service.cleanup()
            
            if self.bigquery_transfer_service:
                await self.bigquery_transfer_service.cleanup()
                
            logger.info("Enhanced CloudSellAnalyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during enhanced sell analyzer cleanup: {e}")
    
    # Additional utility methods for sell analysis
    
    async def get_sell_pressure_summary(self, days_back: float = 7) -> Dict:
        """Get sell pressure summary for this network"""
        try:
            if not self.bigquery_transfer_service:
                return {}
            
            # Get transfer stats for sell analysis
            transfer_stats = await self.bigquery_transfer_service.get_transfer_stats(
                network=self.network, days_back=int(days_back)
            )
            
            sell_stats = transfer_stats.get('by_type', {}).get('sell', {})
            
            return {
                'network': self.network,
                'timeframe_days': days_back,
                'total_sell_transactions': sell_stats.get('count', 0),
                'total_eth_received': sell_stats.get('total_eth', 0),
                'unique_selling_wallets': sell_stats.get('networks', {}).get(self.network, {}).get('unique_wallets', 0),
                'avg_eth_per_sell': sell_stats.get('networks', {}).get(self.network, {}).get('avg_eth', 0),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sell pressure summary: {e}")
            return {}
    
    async def get_analysis_health(self) -> Dict:
        """Get sell analyzer health and status"""
        return {
            'network': self.network,
            'analyzer_type': 'sell_pressure',
            'initialized': self._initialized,
            'ai_enhancement_available': self.data_processor._enhanced_scoring_enabled if self.data_processor else False,
            'last_analysis_stats': self.stats,
            'services_status': {
                'database': bool(self.db_service),
                'bigquery': bool(self.bigquery_transfer_service),
                'alchemy': bool(self.alchemy_service),
                'data_processor': bool(self.data_processor)
            },
            'sell_analysis_capabilities': {
                'eth_received_calculation': True,
                'sell_pressure_scoring': True,
                'ai_risk_assessment': self.data_processor._enhanced_scoring_enabled if self.data_processor else False,
                'web3_data_integration': True
            }
        }
    
    def log_performance_summary(self):
        """Log detailed sell analyzer performance summary"""
        logger.info("=== SELL ANALYZER PERFORMANCE SUMMARY ===")
        logger.info(f"Network: {self.network}")
        logger.info(f"Analysis time: {self.stats.get('analysis_time', 0):.2f}s")
        logger.info(f"Wallets processed: {self.stats.get('wallets_processed', 0)}")
        logger.info(f"Transfers processed: {self.stats.get('transfers_processed', 0)}")
        logger.info(f"Transfers stored: {self.stats.get('transfers_stored', 0)}")
        logger.info(f"Sells detected: {self.stats.get('non_zero_eth_count', 0)}")
        logger.info(f"AI enhanced tokens: {self.stats.get('ai_enhanced_tokens', 0)}")
        logger.info(f"AI confidence average: {self.stats.get('ai_confidence_avg', 0):.2f}")
        logger.info("=" * 48)