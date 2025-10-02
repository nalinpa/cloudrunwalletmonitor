import pandas as pd
import np as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp

from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from utils.config import Config

logger = logging.getLogger(__name__)

# LAZY AI LOADING
AI_AVAILABLE = None
AdvancedCryptoAI_CLASS = None

class UnifiedDataProcessor:
    
    def __init__(self):
        # Token exclusion lists
        self.config = Config()
        self.excluded_assets = frozenset({
            'ETH', 'WETH', 'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'LUSD', 'USDC.E'
        })
        
        self.excluded_contracts = frozenset({
            '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
            '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
            '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
            '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH
            '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC on Base
        })
        
        # QUALITY THRESHOLDS
        self.min_eth_value = 0.02  # Minimum 0.02 ETH for verified trades
        
        # Services
        self.bigquery_transfer_service = None
        self._last_stored_count = 0
        self._last_quality_score = 1.0
        
        # AI system (lazy loaded)
        self.ai_engine = None
        self._ai_enabled = None
        
        # Web3 session
        self._session = None
        
        # Performance tracking
        self.stats = {
            'transfers_processed': 0,
            'transfers_stored': 0,
            'ai_enhanced_tokens': 0,
            'web3_enriched_tokens': 0
        }
        
        logger.info("🚀 Data Processor with Quality Filtering (min 0.02 ETH)")
    
    def set_transfer_service(self, transfer_service):
        """Set BigQuery transfer service"""
        self.bigquery_transfer_service = transfer_service
        logger.info("✅ BigQuery transfer service connected")
    
    def _test_ai_availability(self):
        """Test AI availability on first use"""
        global AI_AVAILABLE, AdvancedCryptoAI_CLASS
        
        if AI_AVAILABLE is not None:
            return AI_AVAILABLE
        
        try:
            logger.info("🔍 Testing AI system availability...")
            
            import sklearn
            import numpy as np
            import pandas as pd
            
            from core.analysis.ai_system import AdvancedCryptoAI
            
            test_instance = AdvancedCryptoAI()
            
            AI_AVAILABLE = True
            AdvancedCryptoAI_CLASS = AdvancedCryptoAI
            self._ai_enabled = True
            
            logger.info("🤖 AI system operational")
            return True
            
        except Exception as e:
            logger.warning(f"❌ AI system not available: {e}")
            AI_AVAILABLE = False
            AdvancedCryptoAI_CLASS = None
            self._ai_enabled = False
            return False

    def _get_ai_engine(self):
        """Lazy load AI engine"""
        if self._ai_enabled is None:
            self._test_ai_availability()
        
        if not self._ai_enabled:
            return None
        
        if self.ai_engine is None:
            try:
                self.ai_engine = AdvancedCryptoAI_CLASS()
                logger.info("✅ AI engine created")
            except Exception as e:
                logger.error(f"❌ Failed to create AI engine: {e}")
                self._ai_enabled = False
                self.ai_engine = None
        
        return self.ai_engine
    
    # ============================================================================
    # QUALITY VERIFICATION
    # ============================================================================
    
    def _is_verified_trade(self, eth_value: float, token_symbol: str, 
                          contract_address: str, tx_hash: str) -> tuple:
        """
        Verify if a trade meets quality standards for storage
        Returns: (is_valid, rejection_reason)
        """
        # Check 1: Minimum ETH value
        if eth_value < self.min_eth_value:
            return False, f"below_minimum ({eth_value:.4f} ETH < {self.min_eth_value})"
        
        # Check 2: Not excluded token
        if self.is_excluded_token(token_symbol, contract_address):
            return False, f"excluded_token ({token_symbol})"
        
        # Check 3: Valid transaction hash
        if not tx_hash or len(tx_hash) != 66:
            return False, "invalid_tx_hash"
        
        # Check 4: Valid contract address
        if not contract_address or len(contract_address) != 42:
            return False, "invalid_contract"
        
        # Check 5: Not dust amount
        if eth_value < 0.00001:
            return False, "dust_amount"
        
        # Check 6: Not unreasonably large
        if eth_value > 10000:
            return False, "unrealistic_amount"
        
        return True, "verified"
    
    def _calculate_trade_quality(self, eth_value: float) -> float:
        """Calculate quality score for a trade (0-100)"""
        if eth_value >= 1.0:
            return 100.0
        elif eth_value >= 0.5:
            return 90.0
        elif eth_value >= 0.2:
            return 80.0
        elif eth_value >= 0.1:
            return 70.0
        elif eth_value >= 0.05:
            return 60.0
        else:
            return 50.0
    
    # ============================================================================
    # PROCESSING WITH QUALITY FILTERING
    # ============================================================================
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """Process transfers with QUALITY FILTERING and optional storage"""
        purchases = []
        verified_transfer_records = []
        rejected_count = {"below_minimum": 0, "excluded_token": 0, "other": 0}
        
        wallet_scores = {w.address: w.score for w in wallets}
        
        if store_data:
            logger.info(f"🗄️ STORAGE MODE: Verified trades will be saved (min {self.min_eth_value} ETH)")
        else:
            logger.info(f"📊 ANALYSIS MODE: No trade storage (store_data=False)")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            incoming = transfers.get('incoming', [])
            outgoing = transfers.get('outgoing', [])
            
            for transfer in incoming:
                try:
                    asset = transfer.get("asset")
                    if not asset or asset == "ETH":
                        continue
                    
                    amount = float(transfer.get("value", "0"))
                    if amount <= 0:
                        continue
                    
                    contract_address = self.extract_contract_address(transfer)
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # QUALITY VERIFICATION
                    is_valid, reason = self._is_verified_trade(
                        eth_spent, asset, contract_address, tx_hash
                    )
                    
                    if not is_valid:
                        if "below_minimum" in reason:
                            rejected_count["below_minimum"] += 1
                        elif "excluded" in reason:
                            rejected_count["excluded_token"] += 1
                        else:
                            rejected_count["other"] += 1
                        continue
                    
                    # Create purchase (always created for analysis)
                    purchase = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,
                        amount_received=amount,
                        eth_spent=eth_spent,
                        wallet_address=address,
                        platform="DEX",
                        block_number=block_number,
                        timestamp=self._parse_timestamp(transfer, block_number),
                        sophistication_score=wallet_scores.get(address, 0),
                        web3_analysis={
                            "contract_address": contract_address,
                            "ca": contract_address,
                            "token_symbol": asset,
                            "network": network,
                            "is_verified_trade": True,
                            "quality_score": self._calculate_trade_quality(eth_spent)
                        }
                    )
                    
                    purchases.append(purchase)
                    
                    # CONDITIONAL STORAGE: Only if store_data=True
                    if store_data:
                        transfer_record = Transfer(
                            wallet_address=address,
                            token_address=contract_address,
                            transfer_type=TransferType.BUY,
                            timestamp=self._parse_timestamp(transfer, block_number),
                            cost_in_eth=eth_spent,
                            transaction_hash=tx_hash,
                            block_number=block_number,
                            token_amount=amount,
                            token_symbol=asset,
                            network=network,
                            platform="DEX",
                            wallet_sophistication_score=wallet_scores.get(address, 0)
                        )
                        verified_transfer_records.append(transfer_record)
                
                except Exception as e:
                    logger.debug(f"Error processing transfer: {e}")
                    continue
        
        # Log filtering results
        total_rejected = sum(rejected_count.values())
        logger.info(f"✅ VERIFIED: {len(purchases)} purchases (min {self.min_eth_value} ETH)")
        logger.info(f"❌ REJECTED: {total_rejected} ({rejected_count['below_minimum']} below min, "
                   f"{rejected_count['excluded_token']} excluded, {rejected_count['other']} other)")
        
        # ACTUAL STORAGE
        if store_data and self.bigquery_transfer_service and verified_transfer_records:
            try:
                logger.info(f"💾 Storing {len(verified_transfer_records)} verified trades to BigQuery...")
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(
                    verified_transfer_records
                )
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} verified trades successfully")
            except Exception as e:
                logger.error(f"❌ Failed to store verified trades: {e}")
                self._last_stored_count = 0
        elif store_data:
            logger.warning("⚠️ Storage requested but no verified trades to store")
            self._last_stored_count = 0
        else:
            logger.info(f"📊 Analysis mode: {len(purchases)} purchases processed, none stored")
            self._last_stored_count = 0
        
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """Process sells with QUALITY FILTERING and optional storage"""
        sells = []
        verified_transfer_records = []
        rejected_count = {"below_minimum": 0, "excluded_token": 0, "other": 0}
        
        wallet_scores = {w.address: w.score for w in wallets}
        
        if store_data:
            logger.info(f"🗄️ STORAGE MODE: Verified sell trades will be saved (min {self.min_eth_value} ETH)")
        else:
            logger.info(f"📊 ANALYSIS MODE: No sell trade storage (store_data=False)")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            outgoing = transfers.get('outgoing', [])
            incoming = transfers.get('incoming', [])
            
            for transfer in outgoing:
                try:
                    asset = transfer.get("asset")
                    if not asset or asset == "ETH":
                        continue
                    
                    amount_sold = float(transfer.get("value", "0"))
                    if amount_sold <= 0:
                        continue
                    
                    contract_address = self.extract_contract_address(transfer)
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    # QUALITY VERIFICATION
                    is_valid, reason = self._is_verified_trade(
                        eth_received, asset, contract_address, tx_hash
                    )
                    
                    if not is_valid:
                        if "below_minimum" in reason:
                            rejected_count["below_minimum"] += 1
                        elif "excluded" in reason:
                            rejected_count["excluded_token"] += 1
                        else:
                            rejected_count["other"] += 1
                        continue
                    
                    # Create verified sell
                    sell = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,
                        amount_received=eth_received,
                        eth_spent=0,
                        wallet_address=address,
                        platform="Transfer",
                        block_number=block_number,
                        timestamp=self._parse_timestamp(transfer, block_number),
                        sophistication_score=wallet_scores.get(address, 0),
                        web3_analysis={
                            "contract_address": contract_address,
                            "ca": contract_address,
                            "amount_sold": amount_sold,
                            "is_sell": True,
                            "token_symbol": asset,
                            "network": network,
                            "is_verified_trade": True,
                            "quality_score": self._calculate_trade_quality(eth_received)
                        }
                    )
                    
                    sells.append(sell)
                    
                    # CONDITIONAL STORAGE
                    if store_data:
                        transfer_record = Transfer(
                            wallet_address=address,
                            token_address=contract_address,
                            transfer_type=TransferType.SELL,
                            timestamp=self._parse_timestamp(transfer, block_number),
                            cost_in_eth=eth_received,
                            transaction_hash=tx_hash,
                            block_number=block_number,
                            token_amount=amount_sold,
                            token_symbol=asset,
                            network=network,
                            platform="Transfer",
                            wallet_sophistication_score=wallet_scores.get(address, 0)
                        )
                        verified_transfer_records.append(transfer_record)
                
                except Exception as e:
                    logger.debug(f"Error processing sell: {e}")
                    continue
        
        # Log results
        total_rejected = sum(rejected_count.values())
        logger.info(f"✅ VERIFIED: {len(sells)} sells (min {self.min_eth_value} ETH)")
        logger.info(f"❌ REJECTED: {total_rejected} ({rejected_count['below_minimum']} below min, "
                   f"{rejected_count['excluded_token']} excluded, {rejected_count['other']} other)")
        
        # ACTUAL STORAGE
        if store_data and self.bigquery_transfer_service and verified_transfer_records:
            try:
                logger.info(f"💾 Storing {len(verified_transfer_records)} verified sell records to BigQuery...")
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(
                    verified_transfer_records
                )
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} verified sell records successfully")
            except Exception as e:
                logger.error(f"❌ Failed to store verified sells: {e}")
                self._last_stored_count = 0
        elif store_data:
            logger.warning("⚠️ Storage requested but no verified sell trades to store")
            self._last_stored_count = 0
        else:
            logger.info(f"📊 Analysis mode: {len(sells)} sells processed, none stored")
            self._last_stored_count = 0
        
        return sells
    
    # ============================================================================
    # ENHANCED AI ANALYSIS WITH INTEGRATED WEB3 INTELLIGENCE
    # ============================================================================
    
    async def analyze_purchases_enhanced(self, purchases: List, analysis_type: str) -> Dict:
        """ENHANCED: AI analysis with INTEGRATED Web3 intelligence batch processing"""
        if not purchases:
            return self._create_empty_result(analysis_type)
        
        logger.info(f"🔍 AI Analysis with INTEGRATED Web3 intelligence: {len(purchases)} {analysis_type} transactions")
        
        # Try AI analysis with integrated Web3 intelligence
        ai_engine = self._get_ai_engine()
        if ai_engine:
            try:
                logger.info("🤖 Running AI analysis with INTEGRATED Web3 intelligence...")
                
                result = await ai_engine.complete_ai_analysis_with_web3(purchases, analysis_type)
                
                if result.get('enhanced'):
                    enhanced_result = self._create_result_with_contracts(result, purchases, analysis_type)
                    self.stats['ai_enhanced_tokens'] = len(result.get('scores', {}))
                    self.stats['web3_enriched_tokens'] = result.get('web3_enriched_count', 0)
                    
                    logger.info(f"✅ AI analysis complete: {self.stats['ai_enhanced_tokens']} tokens, {self.stats['web3_enriched_tokens']} Web3 enhanced")
                    return enhanced_result
                else:
                    logger.info("🔄 AI analysis completed, falling back to enhanced basic analysis")
                    
            except Exception as e:
                logger.error(f"AI analysis with Web3 failed: {e}")
        
        # Fallback to enhanced basic analysis
        logger.info("📊 Using enhanced basic analysis with Web3 intelligence...")
        basic_result = await self._analyze_purchases_with_web3_batch(purchases, analysis_type)
        return self._create_result_with_contracts(basic_result, purchases, analysis_type)
    
    async def _analyze_purchases_with_web3_batch(self, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Enhanced basic analysis with BATCH Web3 intelligence processing"""
        try:
            logger.info(f"🔍 Processing {len(purchases)} purchases with batch Web3 intelligence")
            
            # Extract unique tokens
            unique_tokens = {}
            for purchase in purchases:
                token = purchase.token_bought
                if token not in unique_tokens:
                    contract_address = ""
                    if purchase.web3_analysis:
                        contract_address = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                    
                    unique_tokens[token] = {
                        'contract_address': contract_address,
                        'network': purchase.web3_analysis.get('network', 'ethereum') if purchase.web3_analysis else 'ethereum',
                        'purchases': []
                    }
                
                unique_tokens[token]['purchases'].append(purchase)
            
            logger.info(f"📊 Found {len(unique_tokens)} unique tokens for Web3 analysis")
            
            # Batch process Web3 intelligence
            web3_intelligence = await self._batch_process_web3_intelligence(unique_tokens)
            
            # Apply Web3 intelligence to all purchases
            for token, intelligence in web3_intelligence.items():
                for purchase in unique_tokens[token]['purchases']:
                    if purchase.web3_analysis:
                        purchase.web3_analysis.update(intelligence)
                    else:
                        purchase.web3_analysis = intelligence
            
            # Create enhanced DataFrame with Web3 data
            data = []
            for purchase in purchases:
                eth_value = purchase.amount_received if analysis_type == 'sell' else purchase.eth_spent
                web3_data = purchase.web3_analysis or {}
                
                data.append({
                    'token': purchase.token_bought,
                    'eth_value': eth_value,
                    'wallet': purchase.wallet_address,
                    'score': purchase.sophistication_score or 0,
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'liquidity_usd': web3_data.get('liquidity_usd', 0),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0.3),
                    'contract_address': web3_data.get('contract_address', '')
                })
            
            if not data:
                return {'scores': {}, 'analysis_type': analysis_type, 'enhanced': False}
            
            df = pd.DataFrame(data)
            
            # Calculate scores with Web3 intelligence bonuses
            scores = {}
            for token in df['token'].unique():
                token_df = df[df['token'] == token]
                
                # Basic scoring
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['score'].mean()
                
                if analysis_type == 'sell':
                    volume_score = min(total_eth * 100, 60)
                    diversity_score = min(unique_wallets * 10, 25)
                    quality_score = min((avg_score / 100) * 15, 15)
                else:
                    volume_score = min(total_eth * 50, 50)
                    diversity_score = min(unique_wallets * 8, 30)
                    quality_score = min((avg_score / 100) * 20, 20)
                
                # WEB3 INTELLIGENCE BONUSES
                web3_bonus = 0
                is_verified = token_df['is_verified'].any()
                has_liquidity = token_df['has_liquidity'].any()
                max_liquidity = token_df['liquidity_usd'].max()
                avg_risk = token_df['honeypot_risk'].mean()
                
                if is_verified:
                    web3_bonus += 15
                    logger.debug(f"✅ {token}: +15 points for verified contract")
                
                if has_liquidity:
                    web3_bonus += 10
                    logger.debug(f"💧 {token}: +10 points for liquidity")
                
                if max_liquidity > 50000:
                    web3_bonus += 5
                    logger.debug(f"💰 {token}: +5 points for high liquidity (${max_liquidity:,.0f})")
                
                # Risk penalty
                risk_penalty = avg_risk * 15
                if risk_penalty > 2:
                    logger.debug(f"⚠️ {token}: -{risk_penalty:.1f} points for risk")
                
                total_score = volume_score + diversity_score + quality_score + web3_bonus - risk_penalty
                
                scores[token] = {
                    'total_score': float(total_score),
                    'volume_score': float(volume_score),
                    'diversity_score': float(diversity_score),
                    'quality_score': float(quality_score),
                    'web3_bonus': float(web3_bonus),
                    'risk_penalty': float(risk_penalty),
                    'ai_enhanced': False,
                    'confidence': 0.8,
                    'is_verified': bool(is_verified),
                    'has_liquidity': bool(has_liquidity),
                    'liquidity_usd': float(max_liquidity),
                    'honeypot_risk': float(avg_risk)
                }
                
                if web3_bonus > 0:
                    logger.info(f"🚀 {token}: Score={total_score:.1f} (Web3 bonus: +{web3_bonus})")
            
            verified_count = sum(1 for s in scores.values() if s['is_verified'])
            liquid_count = sum(1 for s in scores.values() if s['has_liquidity'])
            logger.info(f"🔍 Web3 Results: {verified_count} verified, {liquid_count} with liquidity")
            
            return {
                'scores': scores,
                'analysis_type': analysis_type,
                'enhanced': True,
                'web3_enhanced': True,
                'web3_enriched_count': verified_count + liquid_count
            }
            
        except Exception as e:
            logger.error(f"Enhanced basic analysis failed: {e}")
            return {'scores': {}, 'analysis_type': analysis_type, 'enhanced': False}
        
# ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def is_excluded_token(self, asset: str, contract_address: str = None) -> bool:
        """Check if token should be excluded"""
        if not asset:
            return True
            
        asset_upper = asset.upper()
        
        if asset_upper in self.excluded_assets:
            return True
        
        if contract_address and contract_address.lower() in self.excluded_contracts:
            return True
        
        if len(asset) <= 6 and any(stable in asset_upper for stable in ['USD', 'DAI']):
            return True
        
        return False
    
    def extract_contract_address(self, transfer: Dict) -> str:
        """Extract contract address from transfer"""
        contract_address = ""
        
        # Method 1: rawContract.address
        raw_contract = transfer.get("rawContract", {})
        if isinstance(raw_contract, dict) and raw_contract.get("address"):
            contract_address = raw_contract["address"]
        
        # Method 2: contractAddress field
        elif transfer.get("contractAddress"):
            contract_address = transfer["contractAddress"]
        
        # Method 3: 'to' address for ERC20
        elif transfer.get("to"):
            to_address = transfer["to"]
            if to_address != "0x0000000000000000000000000000000000000000":
                contract_address = to_address
        
        # Clean and validate
        if contract_address:
            contract_address = contract_address.strip().lower()
            if not contract_address.startswith('0x'):
                contract_address = '0x' + contract_address
            
            if len(contract_address) == 42:
                return contract_address
        
        return ""
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                       target_tx: str, target_block: str) -> float:
        """Calculate ETH spent using centralized conversion rates"""
        if not outgoing_transfers:
            return 0.0
        
        # Get spending rates from config
        spending_rates = {}
        supported_tokens = self.config.get_all_supported_tokens()
        
        for token in supported_tokens:
            rate = self.config.get_spending_rate(token)
            if rate > 0:
                spending_rates[token] = rate
        
        total_eth = 0.0
        
        # Exact transaction match
        for transfer in outgoing_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in spending_rates:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * spending_rates[asset]
                        total_eth += eth_equivalent
                    except (ValueError, TypeError):
                        continue
        
        if total_eth > 0:
            return total_eth
        
        # Block-based matching
        for transfer in outgoing_transfers:
            if transfer.get("blockNum") == target_block:
                asset = transfer.get("asset", "")
                if asset in spending_rates:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * spending_rates[asset]
                        if 0.0001 <= eth_equivalent <= 50.0:
                            total_eth += eth_equivalent
                    except (ValueError, TypeError):
                        continue
        
        return total_eth

    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                            target_tx: str, target_block: str) -> float:
        """Calculate ETH received for sells using centralized conversion rates"""
        if not incoming_transfers:
            return 0.0
        
        # Get receiving rates from config
        receiving_rates = {}
        supported_tokens = self.config.get_all_supported_tokens()
        
        for token in supported_tokens:
            rate = self.config.get_receiving_rate(token)
            if rate > 0:
                receiving_rates[token] = rate
        
        total_eth = 0.0
        
        # Exact transaction match
        for transfer in incoming_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in receiving_rates:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * receiving_rates[asset]
                        total_eth += eth_equivalent
                    except (ValueError, TypeError):
                        continue
        
        if total_eth > 0:
            return total_eth
        
        # Block proximity matching
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            return 0.0
        
        proximity_values = []
        for transfer in incoming_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                if abs(transfer_block_num - target_block_num) <= 10:
                    asset = transfer.get("asset", "")
                    if asset in receiving_rates:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * receiving_rates[asset]
                        if 0.00001 <= eth_equivalent <= 100.0:
                            proximity_values.append(eth_equivalent)
            except (ValueError, TypeError):
                continue
        
        return sum(proximity_values) if proximity_values else 0.0

    def _parse_timestamp(self, transfer: Dict, block_number: int = None) -> datetime:
        """Parse timestamp from transfer"""
        if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata']:
            try:
                return datetime.fromisoformat(transfer['metadata']['blockTimestamp'].replace('Z', '+00:00'))
            except:
                pass
        
        return datetime.utcnow()
    
    def _create_result_with_contracts(self, analysis_results: Dict, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Create result with contract addresses and Web3 intelligence"""
        if not analysis_results or not analysis_results.get('scores'):
            return self._create_empty_result(analysis_type)
        
        scores = analysis_results['scores']
        ranked_tokens = []
        
        # Build lookups
        contract_lookup = {}
        purchase_stats = {}
        web3_intelligence = {}
        
        for purchase in purchases:
            token = purchase.token_bought
            
            if purchase.web3_analysis:
                ca = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                if ca:
                    contract_lookup[token] = ca
                    web3_intelligence[token] = purchase.web3_analysis
            
            if token not in purchase_stats:
                purchase_stats[token] = {'total_eth': 0, 'count': 0, 'wallets': set(), 'scores': []}
            
            if analysis_type == 'sell':
                purchase_stats[token]['total_eth'] += purchase.amount_received
            else:
                purchase_stats[token]['total_eth'] += purchase.eth_spent
            
            purchase_stats[token]['count'] += 1
            purchase_stats[token]['wallets'].add(purchase.wallet_address)
            purchase_stats[token]['scores'].append(purchase.sophistication_score or 0)
        
        # Create ranked results
        for token, score_data in scores.items():
            stats = purchase_stats.get(token, {'total_eth': 0, 'count': 1, 'wallets': set(), 'scores': [0]})
            contract_address = contract_lookup.get(token, '')
            web3_data = web3_intelligence.get(token, {})
            
            # Token data
            if analysis_type == 'sell':
                token_data = {
                    'total_eth_received': float(stats['total_eth']),
                    'total_sells': int(stats['count']),
                    'wallet_count': len(stats['wallets']),
                    'avg_wallet_score': float(np.mean(stats['scores']) if stats['scores'] else 0),
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'sell_pressure_score': score_data['total_score'],
                    'analysis_type': 'sell'
                }
            else:
                token_data = {
                    'total_eth_spent': float(stats['total_eth']),
                    'total_purchases': int(stats['count']),
                    'wallet_count': len(stats['wallets']),
                    'avg_wallet_score': float(np.mean(stats['scores']) if stats['scores'] else 0),
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'alpha_score': score_data['total_score'],
                    'analysis_type': 'buy'
                }
            
            # Add Web3 intelligence
            token_data.update({
                'ai_enhanced': score_data.get('ai_enhanced', False),
                'confidence': score_data.get('confidence', 0.75),
                'platforms': ['DEX'],
                'web3_data': web3_data
            })
            
            # Create AI data
            ai_data_with_web3 = score_data.copy()
            ai_data_with_web3.update(web3_data)
            
            ranked_tokens.append((token, token_data, score_data['total_score'], ai_data_with_web3))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate stats
        if analysis_type == 'sell':
            total_eth = sum(p.amount_received for p in purchases)
        else:
            total_eth = sum(p.eth_spent for p in purchases)
        
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        result = {
            'network': 'unknown',
            'analysis_type': analysis_type,
            'total_transactions': len(purchases),
            'unique_tokens': unique_tokens,
            'total_eth_value': total_eth,
            'ranked_tokens': ranked_tokens,
            'performance_metrics': {
                **self.get_processing_stats(),
            },
            'enhanced': analysis_results.get('enhanced', False),
            'web3_enhanced': True,
            'scores': scores
        }
        
        return result

    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Create empty result"""
        return {
            'network': 'unknown',
            'analysis_type': analysis_type,
            'total_transactions': 0,
            'unique_tokens': 0,
            'total_eth_value': 0.0,
            'ranked_tokens': [],
            'performance_metrics': self.get_processing_stats(),
            'enhanced': False,
            'web3_enhanced': True,
            'scores': {}
        }
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'transfers_processed': self.stats.get('transfers_processed', 0),
            'transfers_stored': self._last_stored_count,
            'last_stored_count': self._last_stored_count,
            'ai_enhanced_tokens': self.stats.get('ai_enhanced_tokens', 0),
            'web3_enriched_tokens': self.stats.get('web3_enriched_tokens', 0),
            'ai_available': self._ai_enabled if self._ai_enabled is not None else False,
            'processing_mode': 'quality_filtered_conditional_storage',
            'min_eth_threshold': self.min_eth_value
        }
    
    async def validate_data_quality(self, purchases: List[Purchase]) -> Dict:
        """Validate data quality"""
        if not purchases:
            return {
                'data_quality_score': 0.0,
                'warnings': ['No purchases to validate'],
                'total_purchases': 0,
                'valid_purchases': 0,
                'invalid_purchases': 0
            }
        
        valid_count = 0
        warnings = []
        
        for purchase in purchases:
            is_valid = True
            
            if not purchase.token_bought:
                warnings.append(f"Purchase missing token_bought")
                is_valid = False
                
            if not purchase.transaction_hash:
                warnings.append(f"Purchase missing transaction_hash")
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        quality_score = valid_count / len(purchases) if purchases else 0
        self._last_quality_score = quality_score
        
        return {
            'data_quality_score': quality_score,
            'warnings': warnings[:10],
            'total_purchases': len(purchases),
            'valid_purchases': valid_count,
            'invalid_purchases': len(purchases) - valid_count,
            'quality_level': 'high' if quality_score >= 0.9 else 'medium' if quality_score >= 0.7 else 'low'
        }
    
    def log_token_analysis_summary(self, purchases: List[Purchase], analysis_type: str):
        """Log token analysis summary"""
        if not purchases:
            logger.info(f"No {analysis_type} data to summarize")
            return
        
        logger.info(f"=== {analysis_type.upper()} ANALYSIS SUMMARY ===")
        logger.info(f"Total transactions: {len(purchases)}")
        logger.info(f"Verified trades (≥{self.min_eth_value} ETH): {len(purchases)}")
        logger.info("=" * (len(f"{analysis_type.upper()} ANALYSIS SUMMARY") + 6))
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self._session:
                await self._session.close()
                self._session = None
            
            logger.info("Data processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Placeholder methods for Web3 intelligence (simplified for now)
    async def _batch_process_web3_intelligence(self, unique_tokens: Dict) -> Dict:
        """Batch process Web3 intelligence - simplified version"""
        web3_results = {}
        for token, token_info in unique_tokens.items():
            web3_results[token] = {
                'contract_address': token_info['contract_address'],
                'ca': token_info['contract_address'],
                'token_symbol': token,
                'network': token_info['network'],
                'is_verified': False,
                'has_liquidity': False,
                'liquidity_usd': 0,
                'honeypot_risk': 0.3
            }
        return web3_results
    
    async def _get_session(self):
        """Get HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session