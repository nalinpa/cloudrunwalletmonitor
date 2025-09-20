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
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                   all_transfers: Dict, network: str,
                                   store_data: bool = False) -> List[Purchase]:
        """Enhanced sell processing with contract address debugging"""
        sells = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for SELL analysis")
        logger.info(f"🔍 Contract address extraction debugging enabled")
        
        total_outgoing_transfers = 0
        contracts_found = 0
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            outgoing = transfers.get('outgoing', [])
            incoming = transfers.get('incoming', [])
            
            total_outgoing_transfers += len(outgoing)
            
            # DEBUG: Show structure of first few outgoing transfers
            if outgoing and len(sells) < 2:  # Only debug first few
                logger.info(f"🔍 Debugging outgoing transfers for wallet {address[:10]}...")
                self.debug_transfer_structure(outgoing, "sell")
            
            for transfer in outgoing:
                try:
                    asset = transfer.get("asset")
                    if not asset or asset == "ETH":
                        continue
                    
                    amount_sold = float(transfer.get("value", "0"))
                    if amount_sold <= 0:
                        continue
                    
                    # ENHANCED: Contract address extraction with logging
                    contract_address = self.extract_contract_address(transfer)
                    if contract_address:
                        contracts_found += 1
                        logger.info(f"✅ {asset}: Found contract {contract_address[:10]}...")
                    else:
                        logger.warning(f"❌ {asset}: No contract address found")
                    
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    if self.is_excluded_token(asset, contract_address) or eth_received < 0.000001:
                        continue
                    
                    # Create sell with contract address
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
                    
                    # Storage record
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
        
        # Summary logging
        logger.info(f"📊 SELL PROCESSING SUMMARY:")
        logger.info(f"  Total outgoing transfers: {total_outgoing_transfers}")
        logger.info(f"  Contract addresses found: {contracts_found}")
        logger.info(f"  Valid sells created: {len(sells)}")
        logger.info(f"  Contract extraction rate: {contracts_found/total_outgoing_transfers*100 if total_outgoing_transfers > 0 else 0:.1f}%")
        
        # Store transfers
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} sell records")
            except Exception as e:
                logger.error(f"❌ Failed to store sells: {e}")
        
        return sells

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
        """Get Web3 intelligence for a single token - ADD HOLDER DATA LOGGING"""
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
                self._check_contract_verification(session, contract_address, network),
                self._check_dexscreener_liquidity_ai(session, contract_address),
                self._check_coingecko_data_ai(session, contract_address, token_symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    intelligence.update(result)
                    if result.get('source'):
                        intelligence['data_sources'].append(result['source'])
            
            logger.info(f"🎯 Final intelligence for {token_symbol}: data_sources={intelligence.get('data_sources')}")
            return intelligence
            
        except Exception as e:
            logger.debug(f"Token intelligence failed for {token_symbol}: {e}")
            return {}


    async def _get_session(self):
        """Get HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _check_contract_verification(self, session, contract_address: str, network: str) -> Dict:
        """Fixed Etherscan V2 API contract verification"""
        try:
            if not self.config.etherscan_api_key:
                logger.debug("No Etherscan API key - skipping verification")
                return {'verification_skipped': 'no_api_key'}
            
            # Get chain ID for Etherscan V2
            chain_id = self.config.chain_ids.get(network.lower())
            if not chain_id:
                logger.debug(f"No chain ID for {network}")
                return {'verification_skipped': 'unsupported_network'}
            
            url = f"{self.config.etherscan_endpoint}?chainid={chain_id}&module=contract&action=getsourcecode&address={contract_address}&apikey={self.config.etherscan_api_key}"
            
            # Rate limiting
            await asyncio.sleep(self.config.etherscan_api_rate_limit)
            
            logger.info(f"🔍 V2 API check: {contract_address[:10]}... on {network} (chain {chain_id})")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle V2 API errors
                    if data.get('status') == '0':
                        error_msg = data.get('message', 'Unknown error')
                        logger.warning(f"V2 API error for {network}: {error_msg}")
                        return {'verification_failed': 'api_error', 'source': 'etherscan_v2_error', 'error': error_msg}
                    
                    # Parse V2 API successful response
                    if data.get('status') == '1' and data.get('result'):
                        result = data['result'][0] if isinstance(data['result'], list) else data['result']
                        
                        source_code = result.get('SourceCode', '')
                        contract_name = result.get('ContractName', '')
                        
                        # Enhanced verification check
                        has_source = bool(source_code and source_code.strip() and 
                                        source_code not in ['', '{{}}', 'Contract source code not verified'])
                        
                        is_verified = has_source
                        
                        if is_verified:
                            logger.info(f"✅ V2 {network.upper()} verified: {contract_name} ({contract_address[:10]}...)")
                        else:
                            logger.info(f"❌ V2 {network.upper()} unverified: ({contract_address[:10]}...)")
                        
                        return {
                            'is_verified': is_verified,
                            'contract_name': contract_name or 'Unknown',
                            'compiler_version': result.get('CompilerVersion', ''),
                            'optimization_used': result.get('OptimizationUsed') == '1',
                            'has_source_code': has_source,
                            'source_code_length': len(source_code) if source_code else 0,
                            'chain_id': chain_id,
                            'source': 'etherscan_v2',
                            'verification_source': 'etherscan_v2'
                        }
                    else:
                        logger.warning(f"V2 API no result for {contract_address}")
                        return {'verification_failed': 'no_result', 'source': 'etherscan_v2_no_result'}
                
                else:
                    logger.warning(f"V2 API HTTP {response.status} for {network}")
                    return {'verification_failed': f'http_{response.status}', 'source': f'etherscan_v2_http_{response.status}'}
            
        except Exception as e:
            logger.warning(f"V2 API exception: {e}")
            return {'verification_failed': 'exception', 'source': 'etherscan_v2_exception', 'error': str(e)}
 
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
        """Get Web3 intelligence for a single token - NO HOLDER COUNT"""
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
                'data_sources': []
            }
            
            # Batch intelligence checks - REMOVED HOLDER COUNT CHECK
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
            
            # Apply heuristics
            intelligence = self._apply_heuristic_analysis(intelligence, token_symbol)
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Token intelligence failed for {token_symbol}: {e}")
            return self._default_web3_intelligence(token_symbol, network)
    
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
        """ENHANCED contract address extraction from Alchemy transfers"""
        contract_address = ""
        
        # Method 1: rawContract.address (primary for Alchemy)
        raw_contract = transfer.get("rawContract", {})
        if isinstance(raw_contract, dict) and raw_contract.get("address"):
            contract_address = raw_contract["address"]
            logger.debug(f"📋 Found CA via rawContract: {contract_address[:10]}...")
        
        # Method 2: contractAddress field (alternative)
        elif transfer.get("contractAddress"):
            contract_address = transfer["contractAddress"]
            logger.debug(f"📋 Found CA via contractAddress: {contract_address[:10]}...")
        
        # Method 3: 'to' address for ERC20 (fallback)
        elif transfer.get("to"):
            to_address = transfer["to"]
            if to_address != "0x0000000000000000000000000000000000000000":
                contract_address = to_address
                logger.debug(f"📋 Found CA via 'to' address: {contract_address[:10]}...")
        
        # Method 4: Check if this is an ERC20 transfer with contract info
        elif transfer.get("category") == "erc20" and transfer.get("rawContract"):
            # Sometimes rawContract is a different structure
            raw = transfer.get("rawContract")
            if isinstance(raw, dict):
                contract_address = raw.get("address", "")
            elif isinstance(raw, str):
                contract_address = raw
        
        # Clean and validate the address
        if contract_address:
            contract_address = contract_address.strip().lower()
            if not contract_address.startswith('0x'):
                contract_address = '0x' + contract_address
            
            # Validate Ethereum address format (42 characters)
            if len(contract_address) == 42:
                logger.info(f"✅ Extracted contract address: {contract_address[:10]}...{contract_address[-4:]}")
                return contract_address
            else:
                logger.warning(f"❌ Invalid contract address length: {len(contract_address)} chars")
        
        # If no contract address found, log the transfer structure for debugging
        logger.warning("❌ No contract address found in transfer")
        logger.debug(f"Transfer keys: {list(transfer.keys())}")
        if "rawContract" in transfer:
            logger.debug(f"rawContract: {transfer['rawContract']}")
        
        return ""

    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH spent"""
        if not outgoing_transfers:
            return 0.0
        
        spending_currencies = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,
            'USDC': 1/2400,
            'AERO': 1/4800,
        }
        
        total_eth = 0.0
        
        # Exact transaction match
        for transfer in outgoing_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in spending_currencies:
                    try:
                        amount = float(transfer.get("value", "0"))
                        total_eth += amount * spending_currencies[asset]
                    except (ValueError, TypeError):
                        continue
        
        if total_eth > 0:
            return total_eth
        
        # Block-based matching
        for transfer in outgoing_transfers:
            if transfer.get("blockNum") == target_block:
                asset = transfer.get("asset", "")
                if asset in spending_currencies:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * spending_currencies[asset]
                        if 0.0001 <= eth_equivalent <= 50.0:
                            total_eth += eth_equivalent
                    except (ValueError, TypeError):
                        continue
        
        return total_eth
    
    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """Calculate ETH received for sells"""
        if not incoming_transfers:
            return 0.0
        
        receiving_currencies = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,
            'USDC': 1/2400,
            'USDC.E': 1/2400,
            'DAI': 1/2400,
            'AERO': 1/4800,
            'FRAX': 1/2400,
            'LUSD': 1/2400,
        }
        
        total_eth = 0.0
        
        # Exact transaction match
        for transfer in incoming_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in receiving_currencies:
                    try:
                        amount = float(transfer.get("value", "0"))
                        total_eth += amount * receiving_currencies[asset]
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
                    if asset in receiving_currencies:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * receiving_currencies[asset]
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
        """Create result with contract addresses and Web3 intelligence - FIXED verification data"""
        if not analysis_results or not analysis_results.get('scores'):
            return self._create_empty_result(analysis_type)
        
        scores = analysis_results['scores']
        ranked_tokens = []
        
        # Build lookups with enhanced Web3 intelligence
        contract_lookup = {}
        purchase_stats = {}
        web3_intelligence = {}
        
        for purchase in purchases:
            token = purchase.token_bought
            
            # ENHANCED: Extract Web3 intelligence more thoroughly
            if purchase.web3_analysis:
                ca = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                if ca:
                    contract_lookup[token] = ca
                    
                    # CRITICAL FIX: Ensure verification status is preserved
                    web3_data = purchase.web3_analysis.copy()
                    
                    # Ensure verification status is properly set
                    if 'is_verified' not in web3_data and 'contract_verified' in web3_data:
                        web3_data['is_verified'] = web3_data['contract_verified']
                    
                    # Ensure all Web3 intelligence fields are included
                    web3_intelligence[token] = {
                        'contract_address': ca,
                        'ca': ca,
                        'token_symbol': token,
                        'network': web3_data.get('network', 'unknown'),
                        'is_verified': bool(web3_data.get('is_verified', False)),
                        'has_liquidity': bool(web3_data.get('has_liquidity', False)),
                        'liquidity_usd': float(web3_data.get('liquidity_usd', 0)),
                        'honeypot_risk': float(web3_data.get('honeypot_risk', 0)),
                        'data_sources': web3_data.get('data_sources', []),
                        'contract_name': web3_data.get('contract_name', ''),
                        'compiler_version': web3_data.get('compiler_version', ''),
                        'source': web3_data.get('source', 'unknown')
                    }
                    
                    logger.info(f"🔍 {token}: Verification={web3_intelligence[token]['is_verified']}, Source={web3_intelligence[token]['source']}")
            
            # Purchase stats (same as before)
            if token not in purchase_stats:
                purchase_stats[token] = {'total_eth': 0, 'count': 0, 'wallets': set(), 'scores': []}
            
            if analysis_type == 'sell':
                purchase_stats[token]['total_eth'] += purchase.amount_received
            else:
                purchase_stats[token]['total_eth'] += purchase.eth_spent
            
            purchase_stats[token]['count'] += 1
            purchase_stats[token]['wallets'].add(purchase.wallet_address)
            purchase_stats[token]['scores'].append(purchase.sophistication_score or 0)
        
        # Create ranked results with ENHANCED Web3 data
        for token, score_data in scores.items():
            stats = purchase_stats.get(token, {'total_eth': 0, 'count': 1, 'wallets': set(), 'scores': [0]})
            contract_address = contract_lookup.get(token, '')
            web3_data = web3_intelligence.get(token, {})
            
            # Enhanced token data with verification status
            if analysis_type == 'sell':
                token_data = {
                    'total_eth_received': float(stats['total_eth']),
                    'total_sells': int(stats['count']),
                    'wallet_count': len(stats['wallets']),
                    'avg_wallet_score': float(np.mean(stats['scores']) if stats['scores'] else 0),
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'sell_pressure_score': score_data['total_score'],
                    'analysis_type': 'sell',
                    
                    # CRITICAL: Include verification status in main data
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'verification_source': web3_data.get('source', 'unknown')
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
                    'analysis_type': 'buy',
                    
                    # CRITICAL: Include verification status in main data
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'verification_source': web3_data.get('source', 'unknown')
                }
            
            # Add comprehensive Web3 intelligence
            token_data.update({
                'ai_enhanced': score_data.get('ai_enhanced', False),
                'confidence': score_data.get('confidence', 0.75),
                'platforms': ['DEX'],
                
                # ENHANCED: Complete Web3 data for notifications
                'web3_data': {
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'token_symbol': token,
                    'network': web3_data.get('network', 'unknown'),
                    'is_verified': web3_data.get('is_verified', False),
                    'has_liquidity': web3_data.get('has_liquidity', False),
                    'liquidity_usd': web3_data.get('liquidity_usd', 0),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0),
                    'contract_name': web3_data.get('contract_name', ''),
                    'verification_source': web3_data.get('source', 'unknown'),
                    'data_sources': web3_data.get('data_sources', [])
                }
            })
            
            # Create comprehensive AI data that includes all Web3 intelligence
            ai_data_with_web3 = score_data.copy()
            ai_data_with_web3.update(web3_data)
            
            ranked_tokens.append((token, token_data, score_data['total_score'], ai_data_with_web3))
            
            # Log verification status for debugging
            verification_status = "VERIFIED" if web3_data.get('is_verified') else "UNVERIFIED"
            logger.info(f"🪙 {token}: {verification_status} (source: {web3_data.get('source', 'unknown')})")
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'network': 'unknown',
            'analysis_type': analysis_type,
            'total_transactions': len(purchases),
            'unique_tokens': len(set(p.token_bought for p in purchases)),
            'total_eth_value': sum(p.amount_received if analysis_type == 'sell' else p.eth_spent for p in purchases),
            'ranked_tokens': ranked_tokens,
            'performance_metrics': {
                **self.get_processing_stats(),
                'web3_intelligence_stats': {
                    'verified_tokens': sum(1 for _, _, _, ai_data in ranked_tokens if ai_data.get('is_verified', False)),
                    'liquid_tokens': sum(1 for _, _, _, ai_data in ranked_tokens if ai_data.get('has_liquidity', False)),
                    'total_analyzed': len(ranked_tokens)
                }
            },
            'enhanced': analysis_results.get('enhanced', False),
            'web3_enhanced': True,
            'scores': scores
        }
        
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