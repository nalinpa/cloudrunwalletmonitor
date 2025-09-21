import pandas as pd
import numpy as np
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
        
        # Services
        self.bigquery_transfer_service = None
        self._last_stored_count = 0
        self._last_quality_score = 1.0
        
        # AI system (lazy loaded)
        self.ai_engine = None
        self._ai_enabled = None
        
        # SIMPLIFIED: No individual Web3 processing
        self._session = None
        
        # Performance tracking
        self.stats = {
            'transfers_processed': 0,
            'transfers_stored': 0,
            'ai_enhanced_tokens': 0,
            'web3_enriched_tokens': 0
        }
        
        logger.info("🚀 Simplified Data Processor - Web3 intelligence integrated into AI analysis")
    
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
    # SIMPLIFIED PROCESSING - NO INDIVIDUAL WEB3 CALLS
    # ============================================================================
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """SIMPLIFIED: Process transfers without individual Web3 calls"""
        purchases = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for BUY analysis")
        logger.info(f"Web3 intelligence will be processed in batch during AI analysis")
        
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
                    
                    if self.is_excluded_token(asset, contract_address) or eth_spent < 0.00001:
                        continue
                    
                    # SIMPLIFIED: Create purchase with basic contract info only
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
                            "network": network
                            # NO Web3 intelligence here - will be added during AI analysis
                        }
                    )
                    
                    purchases.append(purchase)
                    
                    # Storage record
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
        """SIMPLIFIED: Process sells without individual Web3 calls"""
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
                    
                    amount_sold = float(transfer.get("value", "0"))
                    if amount_sold <= 0:
                        continue
                    
                    contract_address = self.extract_contract_address(transfer)
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    if self.is_excluded_token(asset, contract_address) or eth_received < 0.000001:
                        continue
                    
                    # SIMPLIFIED: Create sell with basic info
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
                            "network": network
                        }
                    )
                    
                    sells.append(sell)
                    
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
            
            # Step 1: Extract unique tokens for batch Web3 processing
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
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'liquidity_usd': web3_data.get('liquidity_usd', 0),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0.3),
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
        
        # Process tokens in batches to avoid overwhelming APIs
        batch_size = 5
        token_items = list(unique_tokens.items())
        
        for i in range(0, len(token_items), batch_size):
            batch = token_items[i:i + batch_size]
            
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
                    web3_results[token] = self._default_web3_intelligence(token, token_data['network'])
            
            # Rate limiting between batches
            await asyncio.sleep(0.5)
        
        return web3_results
    
    async def _get_token_web3_intelligence(self, session, token_symbol: str, contract_address: str, network: str) -> Dict:
        """Get Web3 intelligence for a single token"""
        try:
            if not contract_address or len(contract_address) != 42:
                return self._apply_heuristic_intelligence(token_symbol, network, token_info)
            
            intelligence = {
                'contract_address': contract_address.lower(),
                'ca': contract_address.lower(),
                'token_symbol': token_symbol,
                'network': network,
                'is_verified': False,
                'has_liquidity': False,
                'liquidity_usd': 0,
                'honeypot_risk': 0.3,
                'data_sources': [],
                'smart_money_buying': False,
                'whale_accumulation': False,
                'token_age_hours': None
            }
            
            tasks = [
                self._check_contract_verification_ai(session, contract_address, network),
                self._check_dexscreener_liquidity_ai(session, contract_address),
                self._check_coingecko_data_ai(session, contract_address, token_symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    if i == 3:  # Holder data is the 4th task
                        # CRITICAL: Log holder data processing
                        logger.info(f"📊 Processing holder data for {token_symbol}: {result}")
                        if 'holder_count' in result:
                            intelligence['holder_count'] = result['holder_count']
                            intelligence['data_sources'].append('etherscan_v2_holders')
                            logger.info(f"✅ {token_symbol} holder count integrated: {result['holder_count']}")
                        else:
                            logger.warning(f"❌ {token_symbol} holder data empty: {result}")
                    else:
                        intelligence.update(result)
                        if result.get('source'):
                            intelligence['data_sources'].append(result['source'])
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Token intelligence failed for {token_symbol}: {e}")
            return self._apply_heuristic_intelligence(token_symbol, network, token_info)


    async def _get_session(self):
        """Get HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _check_contract_verification(self, session, contract_address: str, network: str) -> Dict:
        """Contract verification using your V2 API config"""
        try:
            if not contract_address or len(contract_address) != 42:
                return {'is_verified': False, 'source': 'invalid_address'}
            
            # Use your config
            if not self.config.etherscan_api_key:
                logger.debug("No Etherscan API key - returning unverified")
                return {'is_verified': False, 'source': 'no_api_key'}
            
            # Get chain ID for network from your config
            chain_id = self.config.chain_ids.get(network.lower())
            if not chain_id:
                logger.debug(f"No chain ID configured for {network} - returning unverified")
                return {'is_verified': False, 'source': 'unsupported_network'}
            
            # Build API URL using your config
            url = f"{self.config.etherscan_endpoint}?chainid={chain_id}&module=contract&action=getsourcecode&address={contract_address}&apikey={self.config.etherscan_api_key}"
            
            # Apply rate limiting from your config
            await asyncio.sleep(self.config.etherscan_api_rate_limit)
            
            logger.debug(f"🔍 Contract verification: {contract_address[:10]}... on {network} (chain {chain_id})")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle API errors
                    if data.get('status') == '0':
                        error_msg = data.get('message', 'Unknown error')
                        logger.debug(f"API error for {network}: {error_msg}")
                        
                        if "rate limit" in error_msg.lower():
                            return {'is_verified': False, 'source': 'rate_limited'}
                        elif "invalid" in error_msg.lower():
                            return {'is_verified': False, 'source': 'invalid_api_key'}
                        else:
                            return {'is_verified': False, 'source': 'api_error', 'error': error_msg}
                    
                    # Parse successful response
                    if data.get('status') == '1' and data.get('result'):
                        result = data['result'][0] if isinstance(data['result'], list) else data['result']
                        
                        source_code = result.get('SourceCode', '')
                        contract_name = result.get('ContractName', '')
                        
                        # Verification check
                        has_source = bool(source_code and source_code.strip() and 
                                        source_code not in ['', '{{}}', 'Contract source code not verified'])
                        
                        is_verified = has_source
                        
                        # Get token age
                        token_age_hours = await self._get_token_age_from_api(session, contract_address, chain_id)
                        
                        if is_verified:
                            logger.info(f"✅ {network.upper()} verified: {contract_name} ({contract_address[:10]}...)")
                        else:
                            logger.info(f"❌ {network.upper()} unverified: ({contract_address[:10]}...)")
                        
                        return {
                            'is_verified': is_verified,
                            'contract_name': contract_name or 'Unknown',
                            'compiler_version': result.get('CompilerVersion', ''),
                            'optimization_used': result.get('OptimizationUsed') == '1',
                            'has_source_code': has_source,
                            'chain_id': chain_id,
                            'token_age_hours': token_age_hours,
                            'source': 'etherscan'
                        }
                    else:
                        logger.debug(f"API no result for {contract_address}")
                        return {'is_verified': False, 'source': 'no_result'}
                
                elif response.status == 403:
                    return {'is_verified': False, 'source': 'forbidden'}
                elif response.status == 429:
                    return {'is_verified': False, 'source': 'rate_limited'}
                else:
                    logger.debug(f"API HTTP {response.status} for {network}")
                    return {'is_verified': False, 'source': f'http_{response.status}'}
        
        except Exception as e:
            logger.debug(f"Contract verification exception: {e}")
            return {'is_verified': False, 'source': 'exception', 'error': str(e)}

    async def _get_token_age_from_api(self, session, contract_address: str, chain_id: int) -> float:
        """Get token age using your API config"""
        try:
            # Get first transaction (contract creation) using your config
            creation_url = f"{self.config.etherscan_endpoint}?chainid={chain_id}&module=account&action=txlist&address={contract_address}&startblock=0&endblock=99999999&page=1&offset=1&sort=asc&apikey={self.config.etherscan_api_key}"
            
            # Apply rate limiting
            await asyncio.sleep(self.config.etherscan_api_rate_limit)
            
            async with session.get(creation_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == '1' and data.get('result'):
                        transactions = data.get('result', [])
                        if transactions:
                            first_tx = transactions[0]
                            creation_timestamp = int(first_tx.get('timeStamp', 0))
                            
                            if creation_timestamp > 0:
                                age_seconds = datetime.now().timestamp() - creation_timestamp
                                token_age_hours = age_seconds / 3600
                                
                                logger.debug(f"🕐 Token age: {token_age_hours:.1f} hours ({token_age_hours/24:.1f} days)")
                                return token_age_hours
            
            return None
            
        except Exception as e:
            logger.debug(f"Token age calculation failed: {e}")
            return None

    async def _get_token_web3_intelligence(self, session, token_symbol: str, contract_address: str, network: str) -> Dict:
        """FIXED: Get Web3 intelligence using your API config with proper holder count flow"""
        try:
            if not contract_address or len(contract_address) != 42:
                return self._default_web3_intelligence(token_symbol, network)
            
            intelligence = {
                'contract_address': contract_address.lower(),
                'ca': contract_address.lower(),
                'token_symbol': token_symbol,
                'network': network,
                'is_verified': False,
                'has_liquidity': False,
                'liquidity_usd': 0,
                'honeypot_risk': 0.3,
                'data_sources': [],
                'token_age_hours': None
            }
            
            tasks = [
                self._check_contract_verification(session, contract_address, network),
                self._check_dexscreener_liquidity(session, contract_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # FIXED: Process results and ensure holder_count is preserved
            for result in results:
                if isinstance(result, dict) and result:
                    # Debug log to see what data we're getting
                    if 'holder_count' in result and result['holder_count'] is not None:
                        logger.info(f"🔍 Found holder data for {token_symbol}: {result['holder_count']} holders")
                    
                    intelligence.update(result)
                    if result.get('source'):
                        intelligence['data_sources'].append(result['source'])
            
            # Apply heuristics if no API data
            if not intelligence['data_sources']:
                intelligence = self._apply_heuristic_analysis(intelligence, token_symbol)
            
            # CRITICAL: Log final holder count for debugging
            final_holder_count = intelligence.get('holder_count')
            if final_holder_count is not None:
                logger.info(f"✅ {token_symbol}: Final holder count = {final_holder_count}")
            else:
                logger.warning(f"❌ {token_symbol}: No holder count data available")
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Token Web3 intelligence failed for {token_symbol}: {e}")
            return self._default_web3_intelligence(token_symbol, network)

    def _apply_heuristic_analysis(self, intelligence: Dict, token_symbol: str) -> Dict:
        """Apply heuristic analysis with holder count estimates"""
        
        # If we already have API data, don't override it
        if intelligence.get('data_sources'):
            return intelligence
        
        symbol_upper = token_symbol.upper()
        
        # Major tokens with known characteristics
        if symbol_upper in ['WETH', 'USDC', 'USDT', 'DAI', 'ETH']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.0,
                'token_age_hours': 8760 * 3,  # 3+ years
                'heuristic_classification': 'major_token'
            })
        
        # DeFi tokens
        elif symbol_upper in ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'SUSHI', 'CRV']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.1,
                'token_age_hours': 8760 * 2,  # 2+ years
                'heuristic_classification': 'defi_token'
            })
        
        # Popular meme tokens
        elif symbol_upper in ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'WIF', 'BONK']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.2,
                'token_age_hours': 8760,  # 1+ year
                'heuristic_classification': 'meme_token'
            })
        
        # Base ecosystem tokens
        elif symbol_upper in ['AERO', 'ZORA'] and intelligence.get('network') == 'base':
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.1,
                'token_age_hours': 4380,  # 6 months
                'heuristic_classification': 'l2_token'
            })
        
        # Unknown tokens - conservative estimate
        else:
            intelligence.update({
                'is_verified': False,
                'has_liquidity': False,
                'honeypot_risk': 0.5,
                'heuristic_classification': 'unknown'
            })
        
        return intelligence
    
    async def _check_dexscreener_liquidity(self, session, contract_address: str) -> Dict:
        """Enhanced DexScreener liquidity check with better error handling"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            # Add small delay to avoid overwhelming DexScreener
            await asyncio.sleep(0.1)
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        # Filter out pairs with very low liquidity
                        valid_pairs = [p for p in pairs if p.get('liquidity', {}).get('usd', 0) > 500]
                        
                        if valid_pairs:
                            # Get the pair with highest liquidity
                            best_pair = max(valid_pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                            liquidity_usd = float(best_pair.get('liquidity', {}).get('usd', 0))
                            
                            # Enhanced liquidity analysis
                            volume_24h = float(best_pair.get('volume', {}).get('h24', 0))
                            price_usd = best_pair.get('priceUsd')
                            dex_name = best_pair.get('dexId', 'Unknown')
                            
                            logger.debug(f"💧 {contract_address[:10]}... liquidity: ${liquidity_usd:,.0f} on {dex_name}")
                            
                            return {
                                'has_liquidity': liquidity_usd > 1000,  # At least $1k liquidity
                                'liquidity_usd': liquidity_usd,
                                'volume_24h_usd': volume_24h,
                                'price_usd': float(price_usd) if price_usd else 0,
                                'dex_name': dex_name,
                                'pair_address': best_pair.get('pairAddress', ''),
                                'liquidity_pools': [dex_name],  # For backwards compatibility
                                'source': 'dexscreener'
                            }
                        else:
                            # Pairs exist but liquidity too low
                            logger.debug(f"💧 {contract_address[:10]}... has pairs but low liquidity")
                            return {
                                'has_liquidity': False,
                                'liquidity_usd': 0,
                                'reason': 'liquidity_too_low',
                                'source': 'dexscreener'
                            }
                    else:
                        # No pairs found
                        logger.debug(f"💧 {contract_address[:10]}... no trading pairs found")
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
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{contract_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('market_data', {})
                    
                    if market_data:
                        return {
                            'price_usd': market_data.get('current_price', {}).get('usd', 0),
                            'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                            'has_coingecko_listing': True,
                            'source': 'coingecko'
                        }
        
        except Exception as e:
            logger.debug(f"CoinGecko check failed: {e}")
        
        return {}
    
    async def debug_web3_config(self):
        """Debug your Web3 configuration"""
        logger.info("=== WEB3 CONFIG DEBUG ===")
        logger.info(f"Etherscan API key: {self.config.etherscan_api_key[:10]}...{self.config.etherscan_api_key[-5:] if self.config.etherscan_api_key else 'None'}")
        logger.info(f"Etherscan endpoint: {self.config.etherscan_endpoint}")
        logger.info(f"Chain IDs: {self.config.chain_ids}")
        logger.info(f"Rate limit: {self.config.etherscan_api_rate_limit}")
        logger.info("========================")
    
    def _apply_heuristic_intelligence(self, intelligence: Dict, token_symbol: str) -> Dict:
        """Apply heuristic intelligence with estimated Gini for well-known tokens"""
        
        # If we have API data, use it
        if intelligence.get('data_sources'):
            return intelligence
        
        symbol_upper = token_symbol.upper()
        
        # Major tokens - assume good distribution
        if symbol_upper in ['WETH', 'USDC', 'USDT', 'DAI', 'ETH']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.0,
                'gini_coefficient': 0.4,  # Good distribution estimate
                'distribution_quality': 'Good (Well Distributed)',
                'concentration_risk': 'low',
                'heuristic_classification': 'major_token'
            })
        
        # DeFi tokens - usually well distributed
        elif symbol_upper in ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'SUSHI', 'CRV']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'gini_coefficient': 0.5,  # Fair distribution
                'distribution_quality': 'Fair (Some Concentration)',
                'concentration_risk': 'medium',
                'heuristic_classification': 'defi_token'
            })
        
        # Meme tokens - often concentrated
        elif symbol_upper in ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'WIF', 'BONK']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.2,
                'gini_coefficient': 0.7,  # Often concentrated
                'distribution_quality': 'Poor (Highly Concentrated)',
                'concentration_risk': 'high',
                'heuristic_classification': 'meme_token'
            })
        
        # Unknown tokens - assume poor distribution until proven otherwise
        else:
            intelligence.update({
                'is_verified': False,
                'has_liquidity': False,
                'honeypot_risk': 0.5,
                'gini_coefficient': 0.8,  # Assume concentrated
                'distribution_quality': 'Very Poor (Whale Dominated)',
                'concentration_risk': 'very_high',
                'heuristic_classification': 'unknown'
            })
        
        return intelligence

    def _default_web3_intelligence(self, token_symbol: str, network: str) -> Dict:
        """Default Web3 intelligence when APIs fail"""
        return {
            'contract_address': '',
            'ca': '',
            'token_symbol': token_symbol,
            'network': network,
            'is_verified': False,
            'has_liquidity': False,
            'liquidity_usd': 0,
            'honeypot_risk': 0.4,
            'data_sources': [], 
            'token_age_hours': None,
            'error': 'api_unavailable'
        }
    
    # ============================================================================
    # UTILITY METHODS (same as before)
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
            
            # Validate Ethereum address format
            if len(contract_address) == 42:
                return contract_address
        
        return ""
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                       target_tx: str, target_block: str) -> float:
        """Calculate ETH spent using centralized conversion rates from config"""
        if not outgoing_transfers:
            return 0.0
        
        # Get spending rates from config
        spending_rates = {}
        supported_tokens = self.config.get_all_supported_tokens()
        
        for token in supported_tokens:
            rate = self.config.get_spending_rate(token)
            if rate > 0:
                spending_rates[token] = rate
        
        # Log current rates for debugging
        logger.debug(f"Using spending rates from config: {len(spending_rates)} tokens")
        logger.debug(f"ETH price: ${self.config.eth_price_usd}")
        
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
                        logger.debug(f"Exact match: {amount} {asset} = {eth_equivalent:.6f} ETH (rate: {spending_rates[asset]:.6f})")
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
                        if 0.0001 <= eth_equivalent <= 50.0:  # Sanity check
                            total_eth += eth_equivalent
                            logger.debug(f"Block match: {amount} {asset} = {eth_equivalent:.6f} ETH")
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
        
        logger.debug(f"Using receiving rates from config: {len(receiving_rates)} tokens")
        
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
                        logger.debug(f"Sell exact: received {amount} {asset} = {eth_equivalent:.6f} ETH")
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
                        if 0.00001 <= eth_equivalent <= 100.0:  # Sanity check for sells
                            proximity_values.append(eth_equivalent)
                            logger.debug(f"Sell proximity: {amount} {asset} = {eth_equivalent:.6f} ETH")
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
        """Create result with contract addresses and Web3 intelligence INCLUDING token age"""
        if not analysis_results or not analysis_results.get('scores'):
            return self._create_empty_result(analysis_type)
        
        scores = analysis_results['scores']
        ranked_tokens = []
        
        # Build lookups including Web3 intelligence
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
                    
                    # IMPORTANT: Store the complete Web3 intelligence data
                    web3_intelligence[token] = purchase.web3_analysis
                    
                    # Log token age if available
                    age = purchase.web3_analysis.get('token_age_hours')
                    if age is not None:
                        logger.info(f"Token age data available for {token}: {age:.1f} hours")
            
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
        
        # Create ranked results with complete Web3 data
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
            
            # Add Web3 intelligence - ENSURE ALL DATA IS PRESERVED
            token_data.update({
                'ai_enhanced': score_data.get('ai_enhanced', False),
                'confidence': score_data.get('confidence', 0.75),
                'platforms': ['DEX'],
                'web3_data': {
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'token_symbol': token,
                    'network': web3_data.get('network', 'unknown'),
                    'token_age_hours': web3_data.get('token_age_hours'),
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'liquidity_usd': web3_data.get('liquidity_usd', 0),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0),
                    'age_source': web3_data.get('age_source'),
                    'data_sources': web3_data.get('data_sources', [])
                }
            })
            
            # Create AI data with Web3 intelligence - PRESERVE ALL FIELDS
            ai_data_with_web3 = score_data.copy()
            ai_data_with_web3.update({
                # Add Web3 fields directly to ai_data for easier access
                'token_age_hours': web3_data.get('token_age_hours'),
                'is_verified': web3_data.get('is_verified', False),
                'has_liquidity': web3_data.get('has_liquidity', False),
                'liquidity_usd': web3_data.get('liquidity_usd', 0),
                'honeypot_risk': web3_data.get('honeypot_risk', 0),
                'has_coingecko_listing': web3_data.get('has_coingecko_listing', False),
                'age_source': web3_data.get('age_source'),
                'data_sources': web3_data.get('data_sources', [])
            })
            
            # Log what we're including
            age = web3_data.get('token_age_hours')
            if age is not None:
                logger.info(f"Including token age in alert data for {token}: {age:.1f} hours from {web3_data.get('age_source', 'unknown')}")
            else:
                logger.info(f"No token age data available for {token}")
            
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
        tokens_with_age = sum(1 for _, _, _, ai_data in ranked_tokens if ai_data.get('token_age_hours') is not None)
        
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
                    'tokens_with_age_data': tokens_with_age,
                    'total_analyzed': len(ranked_tokens)
                }
            },
            'enhanced': analysis_results.get('enhanced', False),
            'web3_enhanced': True,
            'scores': scores
        }
        
        logger.info(f"{analysis_type.upper()}: {verified_tokens} verified, {liquid_tokens} with liquidity, {tokens_with_age} with age data")
        
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
        """Get processing statistics including token age collection"""
        return {
            'transfers_processed': self.stats.get('transfers_processed', 0),
            'transfers_stored': self.stats.get('transfers_stored', 0),
            'last_stored_count': self._last_stored_count,
            'ai_enhanced_tokens': self.stats.get('ai_enhanced_tokens', 0),
            'web3_enriched_tokens': self.stats.get('web3_enriched_tokens', 0),
            'ai_available': self._ai_enabled if self._ai_enabled is not None else False,
            'web3_intelligence_mode': 'batch_integrated_with_age',
            'processing_mode': 'unified_batch_web3_age_collection',
            'token_age_collection': 'api_sources_only'
        }
        
    def _apply_web3_intelligence_to_purchases(self, purchases: List, web3_intelligence: Dict):
        """Apply Web3 intelligence to all purchases - PRESERVE ALL FIELDS"""
        for purchase in purchases:
            token = purchase.token_bought
            if token in web3_intelligence:
                # Update existing web3_analysis with intelligence
                if purchase.web3_analysis:
                    purchase.web3_analysis.update(web3_intelligence[token])
                else:
                    purchase.web3_analysis = web3_intelligence[token].copy()
                
                # Log what we applied
                age = purchase.web3_analysis.get('token_age_hours')
                if age is not None:
                    logger.debug(f"Applied token age to {token}: {age:.1f} hours")
                
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