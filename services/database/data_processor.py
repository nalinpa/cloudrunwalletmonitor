import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp

from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from utils.config import Config

from utils.web3_utils import (
    extract_contract_address,
    is_excluded_token,
    calculate_eth_spent,
    calculate_eth_received,
    parse_timestamp,
    safe_float_conversion,
    safe_int_conversion,
    safe_bool_conversion,
    validate_ethereum_address,
    apply_token_heuristics,
    create_default_web3_intelligence,
    extract_unique_tokens_info,
    chunk_list
)
from utils.constants import (
    EXCLUDED_TOKENS,
    EXCLUDED_CONTRACTS,
    SPENDING_CURRENCIES,
    RECEIVING_CURRENCIES,
    DEFAULTS,
    TIME_WINDOWS,
    API_ENDPOINTS
)

logger = logging.getLogger(__name__)

# LAZY AI LOADING
AI_AVAILABLE = None
AdvancedCryptoAI_CLASS = None

class UnifiedDataProcessor:
    
    def __init__(self):
        # ✅ SIMPLIFIED: Use constants instead of duplicating
        self.config = Config()
        
        # ✅ REMOVED: self.excluded_assets and self.excluded_contracts 
        # Now using EXCLUDED_TOKENS and EXCLUDED_CONTRACTS from constants
        
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
        
        logger.info("🚀 Unified Data Processor with consolidated utilities")
    
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
            
            # Test dependencies
            import sklearn
            import numpy as np
            import pandas as pd
            
            # Import AI class
            from core.analysis.ai_system import AdvancedCryptoAI
            
            # Test instantiation
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
    # SIMPLIFIED PROCESSING - USING CONSOLIDATED UTILITIES
    # ============================================================================
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """✅ UPDATED: Process transfers using consolidated utilities"""
        purchases = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for BUY analysis")
        logger.info(f"Using consolidated Web3 utilities")
        
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
                    
                    amount = safe_float_conversion(transfer.get("value", "0"))
                    if amount <= 0:
                        continue
                    
                    # ✅ USING: Consolidated utility function
                    contract_address = extract_contract_address(transfer)
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = safe_int_conversion(block_num, 16) if block_num != "0x0" else 0
                    
                    # ✅ USING: Consolidated utility function
                    eth_spent = calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # ✅ USING: Consolidated utility function
                    if is_excluded_token(asset, contract_address) or eth_spent < 0.00001:
                        continue
                    
                    # Create purchase with basic contract info
                    purchase = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,
                        amount_received=amount,
                        eth_spent=eth_spent,
                        wallet_address=address,
                        platform="DEX",
                        block_number=block_number,
                        # ✅ USING: Consolidated utility function
                        timestamp=parse_timestamp(transfer, block_number),
                        sophistication_score=wallet_scores.get(address, 0),
                        web3_analysis={
                            "contract_address": contract_address,
                            "ca": contract_address,
                            "token_symbol": asset,
                            "network": network
                        }
                    )
                    
                    purchases.append(purchase)
                    
                    # Storage record
                    if store_data:
                        transfer_record = Transfer(
                            wallet_address=address,
                            token_address=contract_address,
                            transfer_type=TransferType.BUY,
                            timestamp=parse_timestamp(transfer, block_number),
                            cost_in_eth=eth_spent,
                            transaction_hash=tx_hash,
                            block_number=block_number,
                            token_amount=amount,
                            token_symbol=asset,
                            network=network,
                            platform="DEX",
                            wallet_sophistication_score=wallet_scores.get(address, 0)
                        )
                        all_transfer_records.append(transfer_record)
                
                except Exception as e:
                    logger.debug(f"Error processing transfer: {e}")
                    continue
        
        # Store transfers
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} transfer records")
            except Exception as e:
                logger.error(f"❌ Failed to store transfers: {e}")
        
        logger.info(f"📊 Created {len(purchases)} purchases for AI analysis")
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """✅ UPDATED: Process sells using consolidated utilities"""
        sells = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for SELL analysis")
        
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
                    
                    amount_sold = safe_float_conversion(transfer.get("value", "0"))
                    if amount_sold <= 0:
                        continue
                    
                    # ✅ USING: Consolidated utility functions
                    contract_address = extract_contract_address(transfer)
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = safe_int_conversion(block_num, 16) if block_num != "0x0" else 0
                    
                    eth_received = calculate_eth_received(incoming, tx_hash, block_num)
                    
                    if is_excluded_token(asset, contract_address) or eth_received < 0.000001:
                        continue
                    
                    # Create sell record
                    sell = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,
                        amount_received=eth_received,
                        eth_spent=0,
                        wallet_address=address,
                        platform="Transfer",
                        block_number=block_number,
                        timestamp=parse_timestamp(transfer, block_number),
                        sophistication_score=wallet_scores.get(address, 0),
                        web3_analysis={
                            "contract_address": contract_address,
                            "ca": contract_address,
                            "amount_sold": amount_sold,
                            "is_sell": True,
                            "token_symbol": asset,
                            "network": network
                        }
                    )
                    
                    sells.append(sell)
                    
                    if store_data:
                        transfer_record = Transfer(
                            wallet_address=address,
                            token_address=contract_address,
                            transfer_type=TransferType.SELL,
                            timestamp=parse_timestamp(transfer, block_number),
                            cost_in_eth=eth_received,
                            transaction_hash=tx_hash,
                            block_number=block_number,
                            token_amount=amount_sold,
                            token_symbol=asset,
                            network=network,
                            platform="Transfer",
                            wallet_sophistication_score=wallet_scores.get(address, 0)
                        )
                        all_transfer_records.append(transfer_record)
                
                except Exception as e:
                    logger.debug(f"Error processing sell: {e}")
                    continue
        
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} sell records")
            except Exception as e:
                logger.error(f"❌ Failed to store sells: {e}")
        
        logger.info(f"📊 Created {len(sells)} sells for AI analysis")
        return sells
    
    # ============================================================================
    # ENHANCED AI ANALYSIS WITH INTEGRATED WEB3 INTELLIGENCE
    # ============================================================================
    
    async def analyze_purchases_enhanced(self, purchases: List, analysis_type: str) -> Dict:
        """ENHANCED: AI analysis with INTEGRATED Web3 intelligence batch processing"""
        if not purchases:
            return self._create_empty_result(analysis_type)
        
        logger.info(f"🔍 AI Analysis with INTEGRATED Web3 intelligence: {len(purchases)} {analysis_type} transactions")
        
        # Step 1: Try AI analysis with integrated Web3 intelligence
        ai_engine = self._get_ai_engine()
        if ai_engine:
            try:
                logger.info("🤖 Running AI analysis with INTEGRATED Web3 intelligence...")
                
                # ENHANCED: AI will now handle Web3 intelligence internally
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
        
        # Step 2: Fallback to enhanced basic analysis with Web3 intelligence
        logger.info("📊 Using enhanced basic analysis with Web3 intelligence...")
        basic_result = await self._analyze_purchases_with_web3_batch(purchases, analysis_type)
        return self._create_result_with_contracts(basic_result, purchases, analysis_type)
    
    async def _analyze_purchases_with_web3_batch(self, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Enhanced basic analysis with BATCH Web3 intelligence processing"""
        try:
            logger.info(f"🔍 Processing {len(purchases)} purchases with batch Web3 intelligence")
            
            # ✅ USING: Consolidated utility function
            unique_tokens = extract_unique_tokens_info(purchases)
            
            logger.info(f"📊 Found {len(unique_tokens)} unique tokens for Web3 analysis")
            
            # Step 2: Batch process Web3 intelligence for unique tokens
            web3_intelligence = await self._batch_process_web3_intelligence(unique_tokens)
            
            # Step 3: Apply Web3 intelligence to all purchases
            for token, intelligence in web3_intelligence.items():
                for purchase in unique_tokens[token]['purchases']:
                    if purchase.web3_analysis:
                        purchase.web3_analysis.update(intelligence)
                    else:
                        purchase.web3_analysis = intelligence
            
            # Step 4: Create enhanced DataFrame with Web3 data
            data = []
            for purchase in purchases:
                eth_value = purchase.amount_received if analysis_type == 'sell' else purchase.eth_spent
                web3_data = purchase.web3_analysis or {}
                
                data.append({
                    'token': purchase.token_bought,
                    'eth_value': eth_value,
                    'wallet': purchase.wallet_address,
                    'score': purchase.sophistication_score or 0,
                    'is_verified': safe_bool_conversion(web3_data.get('is_verified')),
                    'has_liquidity': safe_bool_conversion(web3_data.get('has_liquidity')),
                    'liquidity_usd': safe_float_conversion(web3_data.get('liquidity_usd')),
                    'honeypot_risk': safe_float_conversion(web3_data.get('honeypot_risk'), DEFAULTS['honeypot_risk']),
                    'contract_address': web3_data.get('contract_address', '')
                })
            
            if not data:
                return {'scores': {}, 'analysis_type': analysis_type, 'enhanced': False}
            
            df = pd.DataFrame(data)
            
            # Step 5: Calculate scores with Web3 intelligence bonuses
            scores = {}
            for token in df['token'].unique():
                token_df = df[df['token'] == token]
                
                # Basic scoring
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['score'].mean()
                
                if analysis_type == 'sell':
                    # Sell scoring
                    volume_score = min(total_eth * 100, 60)
                    diversity_score = min(unique_wallets * 10, 25)
                    quality_score = min((avg_score / 100) * 15, 15)
                else:
                    # Buy scoring  
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
                    web3_bonus += 15  # Big bonus for verified contracts
                    logger.debug(f"✅ {token}: +15 points for verified contract")
                
                if has_liquidity:
                    web3_bonus += 10  # Bonus for liquidity
                    logger.debug(f"💧 {token}: +10 points for liquidity")
                
                if max_liquidity > 50000:
                    web3_bonus += 5   # Extra bonus for high liquidity
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
    
    async def _batch_process_web3_intelligence(self, unique_tokens: Dict) -> Dict:
        """Batch process Web3 intelligence for unique tokens"""
        web3_results = {}
        session = await self._get_session()
        
        logger.info(f"🔍 Batch processing Web3 intelligence for {len(unique_tokens)} tokens")
        
        # ✅ USING: Consolidated utility function
        for batch in chunk_list(list(unique_tokens.items()), 5):
            # Process batch concurrently
            tasks = []
            for token, token_data in batch:
                task = self._get_token_web3_intelligence(
                    session, 
                    token, 
                    token_data['contract_address'], 
                    token_data['network']
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for (token, token_data), result in zip(batch, batch_results):
                if isinstance(result, dict):
                    web3_results[token] = result
                else:
                    logger.debug(f"Web3 intelligence failed for {token}: {result}")
                    # ✅ USING: Consolidated utility function
                    web3_results[token] = create_default_web3_intelligence(token, token_data['network'])
            
            # Rate limiting between batches
            await asyncio.sleep(0.5)
        
        return web3_results
    
    async def _get_token_web3_intelligence(self, session, token_symbol: str, contract_address: str, network: str) -> Dict:
        """Get Web3 intelligence for a single token"""
        try:
            # ✅ USING: Consolidated utility function
            if not validate_ethereum_address(contract_address):
                return apply_token_heuristics(token_symbol, network)
            
            intelligence = {
                'contract_address': contract_address.lower(),
                'ca': contract_address.lower(),
                'token_symbol': token_symbol,
                'network': network,
                'is_verified': False,
                'has_liquidity': False,
                'liquidity_usd': 0,
                'honeypot_risk': DEFAULTS['honeypot_risk'],
                'data_sources': []
            }
            
            # Batch intelligence checks
            tasks = [
                self._check_contract_verification(session, contract_address, network),
                self._check_dexscreener_liquidity(session, contract_address),
                self._check_coingecko_data(session, contract_address, token_symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, dict) and result:
                    intelligence.update(result)
                    if result.get('source'):
                        intelligence['data_sources'].append(result['source'])
            
            # ✅ USING: Apply consolidated heuristics if no API data
            if not intelligence['data_sources']:
                heuristic_data = apply_token_heuristics(token_symbol, network)
                intelligence.update(heuristic_data)
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Token intelligence failed for {token_symbol}: {e}")
            return apply_token_heuristics(token_symbol, network)
    
    async def _get_session(self):
        """Get HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _check_contract_verification(self, session, contract_address: str, network: str) -> Dict:
        """Simple contract verification using Etherscan V2 API"""
        try:
            if not self.config.etherscan_api_key:
                logger.debug("No Etherscan API key - returning unverified")
                return {'is_verified': False, 'source': 'no_api_key'}
            
            # Get chain ID for network
            chain_id = self.config.chain_ids.get(network.lower())
            if not chain_id:
                logger.debug(f"No chain ID configured for {network} - returning unverified")
                return {'is_verified': False, 'source': 'unsupported_network'}
            
            # ✅ USING: API endpoint from constants
            url = f"{API_ENDPOINTS['etherscan_v2']}?chainid={chain_id}&module=contract&action=getsourcecode&address={contract_address}&apikey={self.config.etherscan_api_key}"
            
            # Apply rate limiting
            await asyncio.sleep(self.config.etherscan_api_rate_limit)
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == '1' and data.get('result'):
                        result = data['result'][0] if isinstance(data['result'], list) else data['result']
                        
                        source_code = result.get('SourceCode', '')
                        contract_name = result.get('ContractName', '')
                        
                        # Simple verification check
                        has_source = bool(source_code and source_code.strip() and 
                                        source_code not in ['', '{{}}', 'Contract source code not verified'])
                        
                        is_verified = has_source
                        
                        return {
                            'is_verified': is_verified,
                            'contract_name': contract_name or 'Unknown',
                            'compiler_version': result.get('CompilerVersion', ''),
                            'has_source_code': has_source,
                            'chain_id': chain_id,
                            'source': 'etherscan_v2'
                        }
                    else:
                        return {'is_verified': False, 'source': 'no_result'}
                else:
                    return {'is_verified': False, 'source': f'http_{response.status}'}
        
        except Exception as e:
            logger.debug(f"V2 API exception: {e}")
            return {'is_verified': False, 'source': 'exception', 'error': str(e)}
    
    async def _check_dexscreener_liquidity(self, session, contract_address: str) -> Dict:
        """Enhanced DexScreener liquidity check"""
        try:
            # ✅ USING: API endpoint from constants
            url = f"{API_ENDPOINTS['dexscreener_tokens']}{contract_address}"
            
            await asyncio.sleep(0.1)
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        # Filter out pairs with very low liquidity
                        valid_pairs = [p for p in pairs if safe_float_conversion(p.get('liquidity', {}).get('usd', 0)) > 500]
                        
                        if valid_pairs:
                            # Get the pair with highest liquidity
                            best_pair = max(valid_pairs, key=lambda x: safe_float_conversion(x.get('liquidity', {}).get('usd', 0)))
                            liquidity_usd = safe_float_conversion(best_pair.get('liquidity', {}).get('usd', 0))
                            
                            return {
                                'has_liquidity': liquidity_usd > 1000,
                                'liquidity_usd': liquidity_usd,
                                'volume_24h_usd': safe_float_conversion(best_pair.get('volume', {}).get('h24', 0)),
                                'price_usd': safe_float_conversion(best_pair.get('priceUsd', 0)),
                                'dex_name': best_pair.get('dexId', 'Unknown'),
                                'pair_address': best_pair.get('pairAddress', ''),
                                'source': 'dexscreener'
                            }
                        else:
                            return {
                                'has_liquidity': False,
                                'liquidity_usd': 0,
                                'reason': 'liquidity_too_low',
                                'source': 'dexscreener'
                            }
                    else:
                        return {
                            'has_liquidity': False,
                            'liquidity_usd': 0,
                            'reason': 'no_pairs_found',
                            'source': 'dexscreener'
                        }
                
                elif response.status == 429:
                    logger.warning("⚠️ DexScreener rate limit")
                    return {'rate_limited': True, 'source': 'dexscreener'}
                
                else:
                    logger.debug(f"DexScreener API failed: HTTP {response.status}")
                    return {}
        
        except asyncio.TimeoutError:
            logger.debug(f"DexScreener timeout for {contract_address}")
            return {}
        except Exception as e:
            logger.debug(f"DexScreener check failed: {e}")
        
        return {}
    
    async def _check_coingecko_data(self, session, contract_address: str, token_symbol: str) -> Dict:
        """Check CoinGecko data"""
        try:
            # ✅ USING: API endpoint from constants
            url = f"{API_ENDPOINTS['coingecko_contract']}{contract_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('market_data', {})
                    
                    if market_data:
                        return {
                            'price_usd': safe_float_conversion(market_data.get('current_price', {}).get('usd', 0)),
                            'volume_24h': safe_float_conversion(market_data.get('total_volume', {}).get('usd', 0)),
                            'has_coingecko_listing': True,
                            'source': 'coingecko'
                        }
        
        except Exception as e:
            logger.debug(f"CoinGecko check failed: {e}")
        
        return {}
    
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
            
            # Contract lookup
            if purchase.web3_analysis:
                ca = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                if ca:
                    contract_lookup[token] = ca
                    web3_intelligence[token] = purchase.web3_analysis
            
            # Purchase stats
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
                'web3_data': {
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'token_symbol': token,
                    'network': web3_data.get('network', 'unknown'),
                    **web3_data
                }
            })
            
            # Create AI data with Web3 intelligence
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
        verified_tokens = sum(1 for _, _, _, ai_data in ranked_tokens if ai_data.get('is_verified', False))
        liquid_tokens = sum(1 for _, _, _, ai_data in ranked_tokens if ai_data.get('has_liquidity', False))
        
        result = {
            'network': 'unknown',
            'analysis_type': analysis_type,
            'total_transactions': len(purchases),
            'unique_tokens': unique_tokens,
            'total_eth_value': total_eth,
            'ranked_tokens': ranked_tokens,
            'performance_metrics': {
                **self.get_processing_stats(),
                'web3_intelligence_stats': {
                    'verified_tokens': verified_tokens,
                    'liquid_tokens': liquid_tokens,
                    'total_analyzed': len(ranked_tokens)
                }
            },
            'enhanced': analysis_results.get('enhanced', False),
            'web3_enhanced': True,
            'scores': scores
        }
        
        logger.info(f"✅ {analysis_type.upper()}: {verified_tokens} verified, {liquid_tokens} with liquidity")
        
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
            'transfers_stored': self.stats.get('transfers_stored', 0),
            'last_stored_count': self._last_stored_count,
            'ai_enhanced_tokens': self.stats.get('ai_enhanced_tokens', 0),
            'web3_enriched_tokens': self.stats.get('web3_enriched_tokens', 0),
            'ai_available': self._ai_enabled if self._ai_enabled is not None else False,
            'web3_intelligence_mode': 'batch_integrated',
            'processing_mode': 'unified_batch_web3'
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
        logger.info(f"Web3 intelligence: BATCH processed during AI analysis")
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