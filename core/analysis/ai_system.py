import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.config import Config
from core.data.models import AnalysisResult, Purchase

logger = logging.getLogger(__name__)

class AdvancedCryptoAI:
    """Enhanced AI system with integrated batch Web3 intelligence processing"""
    
    def __init__(self):
        self.config = Config()
        try:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
            self.scaler = StandardScaler()
            logger.info("🤖 AI models initialized successfully")
        except Exception as e:
            logger.error(f"❌ AI model initialization failed: {e}")
            raise
        
        # Simplified thresholds - easier to tune
        self.thresholds = {
            'whale_threshold': 0.7,
            'pump_threshold': 0.65,
            'anomaly_threshold': 0.3,
            'smart_money_threshold': 200,
            'min_holder_count': 50,
            'min_liquidity_eth': 5.0
        }
        
        # Scoring weights - easily adjustable
        self.weights = {
            'volume_score': 0.25,
            'wallet_quality': 0.20,
            'momentum': 0.15,
            'liquidity': 0.15,
            'holder_distribution': 0.10,
            'age_factor': 0.08,
            'risk_factor': 0.07
        }
        
        # Web3 intelligence session
        self._web3_session = None
        
        logger.info("🚀 Enhanced AI initialized with batch Web3 intelligence")
    
    # ============================================================================
    # NEW METHOD: AI Analysis with Integrated Web3 Intelligence
    # ============================================================================
    
    async def complete_ai_analysis_with_web3(self, purchases: List, analysis_type: str) -> Dict:
        """
        ENHANCED: AI analysis with integrated batch Web3 intelligence processing
        """
        try:
            logger.info(f"🤖 AI Analysis with INTEGRATED Web3 intelligence: {len(purchases)} {analysis_type}")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Extract unique tokens for batch Web3 processing
            unique_tokens = self._extract_unique_tokens_info(purchases)
            logger.info(f"🔍 Found {len(unique_tokens)} unique tokens for batch Web3 analysis")
            
            # Step 2: Batch process Web3 intelligence with error handling
            try:
                web3_intelligence = await self._batch_process_web3_intelligence(unique_tokens)
            except Exception as e:
                logger.warning(f"Web3 intelligence processing failed: {e}, using defaults")
                web3_intelligence = {token: self._default_web3_intelligence(token, 'ethereum') 
                                for token in unique_tokens.keys()}
            
            # Step 3: Apply Web3 intelligence to all purchases
            self._apply_web3_intelligence_to_purchases(purchases, web3_intelligence)
            
            # Step 4: Run enhanced AI analysis with Web3 data
            try:
                result = await self._run_enhanced_ai_analysis(purchases, analysis_type)
            except KeyError as e:
                logger.error(f"KeyError in AI analysis: {e}")
                logger.info("Falling back to basic analysis")
                result = await self._run_basic_ai_analysis(purchases, analysis_type)
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                result = self._create_empty_result(analysis_type)
            
            # Step 5: Add Web3 statistics
            result['web3_enriched_count'] = len(web3_intelligence)
            result['web3_enhanced'] = True
            
            logger.info(f"✅ AI with Web3 complete: {len(result.get('scores', {}))} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI analysis with Web3 failed: {e}")
            logger.error(f"Full error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_result(analysis_type)
    
    def _extract_unique_tokens_info(self, purchases: List) -> Dict:
        """Extract unique token information for batch processing"""
        unique_tokens = {}
        
        for purchase in purchases:
            token = purchase.token_bought
            if token not in unique_tokens:
                # Extract basic info for Web3 analysis
                contract_address = ""
                network = "ethereum"  # Default
                
                if purchase.web3_analysis:
                    contract_address = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                    network = purchase.web3_analysis.get('network', 'ethereum')
                
                unique_tokens[token] = {
                    'contract_address': contract_address,
                    'network': network,
                    'token_symbol': token,
                    'purchase_count': 0,
                    'total_eth_value': 0
                }
            
            # Aggregate stats
            unique_tokens[token]['purchase_count'] += 1
            if hasattr(purchase, 'eth_spent'):
                unique_tokens[token]['total_eth_value'] += purchase.eth_spent
            elif hasattr(purchase, 'amount_received'):
                unique_tokens[token]['total_eth_value'] += purchase.amount_received
        
        return unique_tokens
    
    async def _batch_process_web3_intelligence(self, unique_tokens: Dict) -> Dict:
        """Batch process Web3 intelligence for unique tokens"""
        web3_results = {}
        session = await self._get_web3_session()
        
        logger.info(f"🔍 Batch processing Web3 intelligence for {len(unique_tokens)} tokens")
        
        # Process in small batches to avoid overwhelming APIs
        batch_size = 3  # Conservative batch size
        token_items = list(unique_tokens.items())
        
        for i in range(0, len(token_items), batch_size):
            batch = token_items[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for token, token_info in batch:
                task = self._analyze_single_token_web3(session, token, token_info)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for (token, token_info), result in zip(batch, batch_results):
                if isinstance(result, dict) and result:
                    web3_results[token] = result
                    logger.info(f"✅ {token}: Verified={result.get('is_verified')}, Liquidity=${result.get('liquidity_usd', 0):,.0f}")
                else:
                    logger.debug(f"Web3 intelligence failed for {token}: {result}")
                    web3_results[token] = self._default_web3_intelligence(token, token_info['network'])
            
            # Rate limiting between batches
            if i + batch_size < len(token_items):
                await asyncio.sleep(0.8)  # 800ms between batches
        
        successful_count = sum(1 for result in web3_results.values() if result.get('data_sources'))
        logger.info(f"🎯 Web3 intelligence: {successful_count}/{len(unique_tokens)} tokens enriched with API data")
        
        return web3_results
    

    async def _analyze_single_token_web3(self, session, token_symbol: str, token_info: Dict) -> Dict:
        """Check if holder analysis is being called"""
        try:
            contract_address = token_info['contract_address']
            network = token_info['network']
            
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
                'whale_accumulation': False
            }
            
            # CHECK: Is holder analysis being called?
            # This is likely missing in your current code
            tasks = [
                self._check_contract_verification_ai(session, contract_address, network),
                self._check_dexscreener_liquidity_ai(session, contract_address),
                self._check_coingecko_data_ai(session, contract_address, token_symbol),
                # ADD THIS LINE if it's missing:
                self._get_holder_data(contract_address, network)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results with debugging
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    logger.info(f"DEBUG: Task {i} returned: {list(result.keys())}")
                    intelligence.update(result)
                    source = result.get('source') or result.get('data_source')
                    if source:
                        intelligence['data_sources'].append(source)
                else:
                    logger.warning(f"DEBUG: Task {i} failed: {result}")
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Single token Web3 analysis failed for {token_symbol}: {e}")
            return self._apply_heuristic_intelligence(token_symbol, network, token_info)

    def debug_analyzer_result_creation(self, analysis_results: Dict, purchases: List[Purchase]) -> AnalysisResult:
        """Debug version to check data extraction"""
        
        for purchase in purchases:
            token = purchase.token_bought
            if purchase.web3_analysis:
                logger.info(f"DEBUG: {token} web3_analysis keys: {list(purchase.web3_analysis.keys())}")
                
                # Check specifically for holder data
                holder_count = purchase.web3_analysis.get('holder_count')
                gini_coefficient = purchase.web3_analysis.get('gini_coefficient')
                
                logger.info(f"DEBUG: {token} - holder_count: {holder_count}, gini_coefficient: {gini_coefficient}")
            else:
                logger.warning(f"DEBUG: {token} has no web3_analysis")
        
            
    async def _get_token_age_from_dexscreener(self, session, contract_address: str) -> Dict:
        """Get token age from DexScreener pair creation data"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        oldest_timestamp = None
                        
                        for pair in pairs:
                            created_at = pair.get('pairCreatedAt')
                            if created_at:
                                try:
                                    # Handle millisecond timestamps
                                    if isinstance(created_at, (int, float)):
                                        if created_at > 1e12:  # Milliseconds
                                            timestamp = created_at / 1000
                                        else:  # Seconds
                                            timestamp = created_at
                                        
                                        if oldest_timestamp is None or timestamp < oldest_timestamp:
                                            oldest_timestamp = timestamp
                                            
                                except Exception as e:
                                    logger.debug(f"Error parsing DexScreener timestamp: {e}")
                                    continue
                        
                        if oldest_timestamp:
                            import time
                            age_hours = (time.time() - oldest_timestamp) / 3600
                            
                            return {
                                'token_age_hours': age_hours,
                                'age_source': 'dexscreener',
                                'source': 'dexscreener_age'
                            }
            
            return {}
            
        except Exception as e:
            logger.debug(f"DexScreener age check failed: {e}")
            return {}

    async def _get_etherscan_token_age(self, session, contract_address: str, network: str) -> Dict:
        """Get token age from Etherscan contract creation"""
        try:
            if not hasattr(self.config, 'etherscan_api_key') or not self.config.etherscan_api_key:
                return {}
            
            # Only support Ethereum and Base for now
            if network.lower() not in ['ethereum', 'base']:
                return {}
            
            # Get chain ID
            chain_id = getattr(self.config, 'chain_ids', {}).get(network.lower())
            if not chain_id:
                chain_id = 1 if network.lower() == 'ethereum' else 8453
            
            # Use Etherscan V2 API for contract creation
            endpoint = getattr(self.config, 'etherscan_endpoint', 'https://api.etherscan.io/v2/api')
            url = f"{endpoint}?chainid={chain_id}&module=contract&action=getcontractcreation&contractaddresses={contract_address}&apikey={self.config.etherscan_api_key}"
            
            # Rate limiting
            await asyncio.sleep(0.2)
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == '1' and data.get('result'):
                        result = data['result'][0] if isinstance(data['result'], list) else data['result']
                        tx_hash = result.get('txHash')
                        
                        if tx_hash:
                            # Get transaction details for block timestamp
                            tx_url = f"{endpoint}?chainid={chain_id}&module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey={self.config.etherscan_api_key}"
                            
                            await asyncio.sleep(0.2)
                            
                            async with session.get(tx_url, timeout=aiohttp.ClientTimeout(total=10)) as tx_response:
                                if tx_response.status == 200:
                                    tx_data = await tx_response.json()
                                    
                                    if tx_data.get('result'):
                                        block_number = tx_data['result'].get('blockNumber')
                                        
                                        if block_number:
                                            # Get block timestamp
                                            block_url = f"{endpoint}?chainid={chain_id}&module=proxy&action=eth_getBlockByNumber&tag={block_number}&boolean=false&apikey={self.config.etherscan_api_key}"
                                            
                                            await asyncio.sleep(0.2)
                                            
                                            async with session.get(block_url, timeout=aiohttp.ClientTimeout(total=10)) as block_response:
                                                if block_response.status == 200:
                                                    block_data = await block_response.json()
                                                    
                                                    if block_data.get('result'):
                                                        timestamp_hex = block_data['result'].get('timestamp')
                                                        
                                                        if timestamp_hex:
                                                            import time
                                                            timestamp = int(timestamp_hex, 16)
                                                            age_hours = (time.time() - timestamp) / 3600
                                                            
                                                            return {
                                                                'token_age_hours': age_hours,
                                                                'age_source': 'etherscan',
                                                                'creation_tx': tx_hash,
                                                                'source': 'etherscan_age'
                                                            }
            
            return {}
            
        except Exception as e:
            logger.debug(f"Etherscan age check failed: {e}")
            return {}

    async def _check_contract_verification_ai(self, session, contract_address: str, network: str) -> Dict:
        """Enhanced contract verification with token age calculation"""
        try:
            from utils.config import Config
            config = Config()
            
            if not config.etherscan_api_key:
                logger.debug("No Etherscan API key - returning unverified")
                return {'is_verified': False, 'source': 'no_api_key'}
            
            # Get chain ID for network from your config
            chain_id = config.chain_ids.get(network.lower())
            if not chain_id:
                logger.debug(f"No chain ID configured for {network} - returning unverified")
                return {'is_verified': False, 'source': 'unsupported_network'}
            
            # Build API URL using your config
            url = f"{config.etherscan_endpoint}?chainid={chain_id}&module=contract&action=getsourcecode&address={contract_address}&apikey={config.etherscan_api_key}"
            
            # Apply rate limiting from your config
            await asyncio.sleep(config.etherscan_api_rate_limit)
            
            logger.debug(f"🔍 API check: {contract_address[:10]}... on {network} (chain {chain_id})")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle API errors
                    if data.get('status') == '0':
                        error_msg = data.get('message', 'Unknown error')
                        logger.debug(f"API error for {network}: {error_msg}")
                        return {'is_verified': False, 'source': 'api_error', 'error': error_msg}
                    
                    # Parse successful response
                    if data.get('status') == '1' and data.get('result'):
                        result = data['result'][0] if isinstance(data['result'], list) else data['result']
                        
                        source_code = result.get('SourceCode', '')
                        contract_name = result.get('ContractName', '')
                        
                        # Simple verification check
                        has_source = bool(source_code and source_code.strip() and 
                                        source_code not in ['', '{{}}', 'Contract source code not verified'])
                        
                        is_verified = has_source
                        
                        # ENHANCED: Get token age
                        token_age_hours = await self._get_token_age(session, contract_address, chain_id, config)
                        
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
                            'token_age_hours': token_age_hours,  # Added token age
                            'source': 'etherscan'
                        }
                    else:
                        logger.debug(f"API no result for {contract_address}")
                        return {'is_verified': False, 'source': 'no_result'}
                
                else:
                    logger.debug(f"API HTTP {response.status} for {network}")
                    return {'is_verified': False, 'source': f'http_{response.status}'}
        
        except Exception as e:
            logger.debug(f"API exception: {e}")
            return {'is_verified': False, 'source': 'exception', 'error': str(e)}

    async def _get_token_age(self, session, contract_address: str, chain_id: int, config) -> float:
        """Get token age using API for contract creation"""
        try:
            # Use V2 API to get first transaction (contract creation)
            creation_url = f"{config.etherscan_endpoint}?chainid={chain_id}&module=account&action=txlist&address={contract_address}&startblock=0&endblock=99999999&page=1&offset=1&sort=asc&apikey={config.etherscan_api_key}"
            
            # Apply rate limiting
            await asyncio.sleep(config.etherscan_api_rate_limit)
            
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

    async def _get_holder_data(self, contract_address: str, network: str) -> Dict:
        """Get holder count using V2 Etherscan API configuration WITH enhanced debugging"""
        try:
            logger.info(f"🔍 DEBUGGING: Starting holder data fetch for {contract_address[:10]}... on {network}")
            
            if not self._web3_session:
                self._web3_session = await self._get_web3_session()
            
            # Use V2 API configuration from config
            if not self.config.etherscan_api_key:
                logger.warning("❌ DEBUGGING: No Etherscan API key - returning empty holder data")
                return {}
            
            # Get chain ID for V2 API
            chain_id = self.config.chain_ids.get(network.lower())
            if not chain_id:
                logger.warning(f"❌ DEBUGGING: No chain ID configured for {network} - returning empty holder data")
                return {}
            
            logger.info(f"🔍 DEBUGGING: Using chain ID {chain_id} for {network}")
            
            # ENHANCED: Use V2 endpoint and configuration with more holders for better Gini calculation
            url = f"{self.config.etherscan_endpoint}?chainid={chain_id}&module=token&action=tokenholderlist&contractaddress={contract_address}&page=1&offset=1000&apikey={self.config.etherscan_api_key}"
            
            logger.info(f"🔍 DEBUGGING: API URL: {url[:100]}...") # Don't log full URL with API key
            
            # Apply V2 rate limiting
            await asyncio.sleep(self.config.etherscan_api_rate_limit)
            
            logger.debug(f"🔍 V2 API holder check: {contract_address[:10]}... on {network} (chain {chain_id})")
            
            async with self._web3_session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                logger.info(f"🔍 DEBUGGING: API response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"🔍 DEBUGGING: API response keys: {list(data.keys())}")
                    logger.info(f"🔍 DEBUGGING: API status: {data.get('status')}")
                    logger.info(f"🔍 DEBUGGING: API message: {data.get('message', 'No message')}")
                    
                    # Handle V2 API errors
                    if data.get('status') == '0':
                        error_msg = data.get('message', 'Unknown error')
                        logger.warning(f"❌ DEBUGGING: V2 API error for {network}: {error_msg}")
                        
                        if 'rate limit' in error_msg.lower():
                            logger.warning("⚠️ V2 API rate limit hit for holder data")
                        elif 'invalid' in error_msg.lower():
                            logger.warning("⚠️ V2 API invalid request for holder data")
                        
                        return {}
                    
                    # Parse successful V2 response
                    if data.get('status') == '1' and data.get('result'):
                        holders = data.get('result', [])
                        holder_count = len(holders)
                        
                        logger.info(f"🔍 DEBUGGING: Raw holders count: {holder_count}")
                        logger.info(f"🔍 DEBUGGING: First holder keys: {list(holders[0].keys()) if holders else 'No holders'}")
                        
                        if holder_count > 0:
                            logger.info(f"✅ V2 {network.upper()} holders: {holder_count:,} ({contract_address[:10]}...)")
                            
                            # ENHANCED: Calculate comprehensive holder metrics with debugging
                            logger.info(f"🔍 DEBUGGING: Calculating basic concentration...")
                            basic_concentration = self._calculate_concentration(holders)
                            logger.info(f"🔍 DEBUGGING: Basic concentration: {basic_concentration}")
                            
                            logger.info(f"🔍 DEBUGGING: Calculating Gini metrics...")
                            gini_metrics = self._calculate_comprehensive_gini_metrics(holders)
                            logger.info(f"🔍 DEBUGGING: Gini metrics keys: {list(gini_metrics.keys())}")
                            logger.info(f"🔍 DEBUGGING: Gini coefficient: {gini_metrics.get('gini_coefficient')}")
                            
                            # Combine all metrics
                            result = {
                                'holder_count': holder_count,
                                'top_holder_percentage': basic_concentration,
                                'data_source': 'etherscan_v2',
                                'chain_id': chain_id,
                                'holders_list': holders,  # Keep for debugging
                                **gini_metrics  # Add Gini and distribution metrics
                            }
                            
                            logger.info(f"🔍 DEBUGGING: Final result keys: {list(result.keys())}")
                            
                            # Enhanced logging
                            gini = gini_metrics.get('gini_coefficient')
                            risk = gini_metrics.get('holder_concentration_risk', 'UNKNOWN')
                            if gini is not None:
                                logger.info(f"📊 V2 {network.upper()} distribution: {holder_count:,} holders, "
                                        f"Gini: {gini:.3f}, Risk: {risk}")
                            
                            return result
                        else:
                            logger.warning(f"❌ DEBUGGING: V2 API returned empty holder list for {contract_address[:10]}...")
                            return {}
                    else:
                        logger.warning(f"❌ DEBUGGING: V2 API no result for holder data: {contract_address[:10]}...")
                        logger.info(f"🔍 DEBUGGING: Full response: {data}")
                        return {}
                
                else:
                    logger.warning(f"❌ DEBUGGING: V2 API HTTP {response.status} for holder data on {network}")
                    response_text = await response.text()
                    logger.info(f"🔍 DEBUGGING: Response text: {response_text[:200]}...")
                    return {}
            
        except Exception as e:
            logger.error(f"❌ DEBUGGING: V2 holder data exception: {e}")
            import traceback
            logger.error(f"🔍 DEBUGGING: Traceback: {traceback.format_exc()}")
            return {}
    
    def _calculate_comprehensive_gini_metrics(self, holders: List[Dict]) -> Dict:
        """Calculate comprehensive distribution metrics including Gini coefficient"""
        try:
            if not holders or len(holders) < 5:
                # Use estimation for small datasets
                return self._estimate_distribution_from_count(len(holders))
            
            # Extract balances from V2 API format
            balances = []
            for holder in holders:
                quantity_str = holder.get('TokenHolderQuantity', '0')
                
                try:
                    # Handle hex or decimal values
                    if quantity_str.startswith('0x'):
                        quantity = int(quantity_str, 16)
                    else:
                        quantity = float(quantity_str)
                    
                    if quantity > 0:
                        balances.append(quantity)
                        
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse holder quantity: {quantity_str}")
                    continue
            
            if len(balances) < 5:
                return self._estimate_distribution_from_count(len(balances))
            
            # Sort balances in descending order for analysis
            balances.sort(reverse=True)
            total_supply = sum(balances)
            
            # Calculate Gini coefficient using your existing method (fixed)
            gini = self._calculate_gini_coefficient(balances)
            
            # Calculate concentration metrics
            top_10_count = min(10, len(balances))
            top_10_percentage = sum(balances[:top_10_count]) / total_supply * 100 if total_supply > 0 else 0
            
            # Calculate top holder percentage
            top_holder_percentage = balances[0] / total_supply * 100 if total_supply > 0 else 0
            
            # Assess concentration risk
            concentration_risk = self._assess_concentration_risk(gini, top_10_percentage, top_holder_percentage, len(balances))
            
            return {
                'gini_coefficient': round(gini, 4),
                'top_10_holders_percentage': round(top_10_percentage, 2),
                'top_holder_percentage': round(top_holder_percentage, 2),
                'holder_concentration_risk': concentration_risk,
                'distribution_analysis': {
                    'total_sample_holders': len(balances),
                    'whale_holders': len([b for b in balances if b / total_supply > 0.05]),  # >5% holders
                    'retail_holders': len([b for b in balances if b / total_supply < 0.01]),  # <1% holders
                    'median_holding_percentage': (sorted([b/total_supply for b in balances])[len(balances)//2] * 100) if balances else 0
                }
            }
            
        except Exception as e:
            logger.debug(f"Comprehensive metrics calculation failed: {e}")
            return self._estimate_distribution_from_count(len(holders) if holders else 0)

    def _calculate_gini_coefficient(self, balances: List[float]) -> float:
        """Fixed Gini coefficient calculation (your method had a static reference issue)"""
        try:
            if not balances or len(balances) < 2:
                return 0.5  # Default moderate inequality
            
            # Sort balances in ascending order for Gini calculation
            sorted_balances = sorted(balances)
            n = len(sorted_balances)
            total_sum = sum(sorted_balances)
            
            if total_sum == 0:
                return 0.5
            
            # Calculate Gini coefficient using the standard formula
            cumulative_sum = 0
            for i, balance in enumerate(sorted_balances):
                cumulative_sum += balance * (2 * (i + 1) - n - 1)
            
            gini = cumulative_sum / (n * total_sum)
            
            # Ensure Gini is between 0 and 1
            return max(0.0, min(1.0, abs(gini)))
            
        except Exception as e:
            logger.debug(f"Gini calculation failed: {e}")
            return 0.5

    def _estimate_distribution_from_count(self, holder_count: int) -> Dict:
        """Estimate distribution metrics when we only have holder count"""
        if holder_count < 50:
            estimated_gini = 0.8  # High concentration
            concentration_risk = 'VERY_HIGH'
            top_10_percentage = 85.0
        elif holder_count < 200:
            estimated_gini = 0.7  # Moderate-high concentration
            concentration_risk = 'HIGH'
            top_10_percentage = 75.0
        elif holder_count < 1000:
            estimated_gini = 0.6  # Moderate concentration
            concentration_risk = 'MEDIUM'
            top_10_percentage = 60.0
        elif holder_count < 5000:
            estimated_gini = 0.5  # Lower concentration
            concentration_risk = 'LOW'
            top_10_percentage = 45.0
        else:
            estimated_gini = 0.4  # Good distribution
            concentration_risk = 'VERY_LOW'
            top_10_percentage = 35.0
        
        return {
            'gini_coefficient': estimated_gini,
            'top_10_holders_percentage': top_10_percentage,
            'top_holder_percentage': top_10_percentage / 5,  # Estimate top holder
            'holder_concentration_risk': concentration_risk,
            'estimated_data': True,
            'distribution_analysis': {
                'estimation_based': True,
                'confidence': 'low' if holder_count < 100 else 'medium'
            }
        }

    def _assess_concentration_risk(self, gini: float, top_10_pct: float, top_holder_pct: float, holder_count: int) -> str:
        """Assess overall concentration risk based on multiple factors"""
        risk_factors = 0
        
        # Gini coefficient risk
        if gini > 0.8:
            risk_factors += 3
        elif gini > 0.6:
            risk_factors += 2
        elif gini > 0.4:
            risk_factors += 1
        
        # Top 10 holders concentration risk
        if top_10_pct > 80:
            risk_factors += 3
        elif top_10_pct > 60:
            risk_factors += 2
        elif top_10_pct > 40:
            risk_factors += 1
        
        # Top holder concentration risk
        if top_holder_pct > 20:
            risk_factors += 2
        elif top_holder_pct > 10:
            risk_factors += 1
        
        # Holder count risk
        if holder_count < 50:
            risk_factors += 3
        elif holder_count < 200:
            risk_factors += 2
        elif holder_count < 1000:
            risk_factors += 1
        
        # Map risk factors to categories
        if risk_factors >= 8:
            return 'VERY_HIGH'
        elif risk_factors >= 6:
            return 'HIGH'
        elif risk_factors >= 4:
            return 'MEDIUM'
        elif risk_factors >= 2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    async def _check_dexscreener_liquidity_ai(self, session, contract_address: str) -> Dict:
        """Check DexScreener liquidity - optimized for AI analysis"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        # Find best liquidity pair
                        valid_pairs = [p for p in pairs if p.get('liquidity', {}).get('usd', 0) > 500]
                        
                        if valid_pairs:
                            best_pair = max(valid_pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                            liquidity_usd = float(best_pair.get('liquidity', {}).get('usd', 0))
                            
                            return {
                                'has_liquidity': liquidity_usd > 1000,
                                'liquidity_usd': liquidity_usd,
                                'price_usd': best_pair.get('priceUsd'),
                                'volume_24h': float(best_pair.get('volume', {}).get('h24', 0)),
                                'dex_name': best_pair.get('dexId', 'Unknown'),
                                'source': 'dexscreener'
                            }
        
        except Exception as e:
            logger.debug(f"DexScreener check failed: {e}")
        
        return {}
    
    async def _check_coingecko_data_ai(self, session, contract_address: str, token_symbol: str) -> Dict:
        """Check CoinGecko data - optimized for AI analysis"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{contract_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('market_data', {})
                    
                    if market_data:
                        return {
                            'price_usd': market_data.get('current_price', {}).get('usd', 0),
                            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                            'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                            'has_coingecko_listing': True,
                            'source': 'coingecko'
                        }
        
        except Exception as e:
            logger.debug(f"CoinGecko check failed: {e}")
        
        return {}
    
    def _apply_heuristic_intelligence(self, token_symbol: str, network: str, token_info: Dict) -> Dict:
        """Apply heuristic intelligence when APIs fail"""
        symbol_upper = token_symbol.upper()
        
        # Base intelligence structure
        intelligence = {
            'contract_address': token_info.get('contract_address', ''),
            'ca': token_info.get('contract_address', ''),
            'token_symbol': token_symbol,
            'network': network,
            'is_verified': False,
            'has_liquidity': False,
            'liquidity_usd': 0,
            'honeypot_risk': 0.4,
            'data_sources': ['heuristic'],
            'smart_money_buying': False,
            'whale_accumulation': False
        }
        
        # Apply heuristics based on token characteristics
        
        # Major tokens
        if symbol_upper in ['WETH', 'USDC', 'USDT', 'DAI', 'ETH']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.0,
                'heuristic_classification': 'major_token'
            })
        
        # DeFi tokens  
        elif symbol_upper in ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'SUSHI', 'CRV']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.1,
                'heuristic_classification': 'defi_token'
            })
        
        # Popular meme tokens
        elif symbol_upper in ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'WIF', 'BONK']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.2,
                'heuristic_classification': 'meme_token'
            })
        
        # Base/L2 ecosystem tokens
        elif symbol_upper in ['AERO', 'ZORA'] and network == 'base':
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.1,
                'heuristic_classification': 'l2_token'
            })
        
        # High-volume tokens (likely legitimate)
        elif token_info.get('total_eth_value', 0) > 5.0:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.2,
                'heuristic_classification': 'high_volume'
            })
        
        # Default for unknown tokens
        else:
            intelligence.update({
                'is_verified': False,
                'has_liquidity': False,
                'honeypot_risk': 0.5,
                'heuristic_classification': 'unknown'
            })
        
        logger.debug(f"🎯 Heuristic: {token_symbol} → Verified={intelligence['is_verified']}, Liquidity={intelligence['has_liquidity']}")
        
        return intelligence
    
    def _apply_web3_intelligence_to_purchases(self, purchases: List, web3_intelligence: Dict):
        """Apply Web3 intelligence to all purchases with enhanced debugging"""
        for purchase in purchases:
            token = purchase.token_bought
            if token in web3_intelligence:
                intelligence = web3_intelligence[token]
                
                logger.info(f"🔍 DEBUGGING: Applying intelligence to {token}")
                logger.info(f"🔍 DEBUGGING: Intelligence keys: {list(intelligence.keys())}")
                
                # Log comprehensive data when found with debugging
                token_age = intelligence.get('token_age_hours')
                holder_count = intelligence.get('holder_count')
                gini = intelligence.get('gini_coefficient')
                concentration_risk = intelligence.get('holder_concentration_risk')
                
                logger.info(f"🔍 DEBUGGING: Token {token} - Age: {token_age}, Holders: {holder_count}, Gini: {gini}, Risk: {concentration_risk}")
                
                # Enhanced logging
                if token_age is not None:
                    age_days = token_age / 24
                    age_category = ("BRAND NEW" if age_days < 1 else "FRESH" if age_days < 7 else 
                                "RECENT" if age_days < 30 else "ESTABLISHED" if age_days < 365 else "MATURE")
                    logger.info(f"Token age data available for {token}: {token_age:.1f} hours ({age_days:.1f}d) - {age_category}")
                
                if holder_count is not None:
                    logger.info(f"Holder count data available for {token}: {holder_count:,} holders")
                
                if gini is not None:
                    gini_category = ("VERY CONCENTRATED" if gini > 0.8 else "CONCENTRATED" if gini > 0.6 else
                                "MODERATE" if gini > 0.4 else "DISTRIBUTED" if gini > 0.2 else "WELL DISTRIBUTED")
                    logger.info(f"Distribution analysis for {token}: Gini {gini:.3f} ({gini_category})")
                
                if concentration_risk and concentration_risk != 'UNKNOWN':
                    logger.info(f"Concentration risk for {token}: {concentration_risk}")
                
                # Update purchase web3_analysis with comprehensive intelligence
                if purchase.web3_analysis:
                    existing_data = purchase.web3_analysis.copy()
                    logger.info(f"🔍 DEBUGGING: Existing web3_analysis keys: {list(existing_data.keys())}")
                    existing_data.update(intelligence)
                    purchase.web3_analysis = existing_data
                else:
                    purchase.web3_analysis = intelligence.copy()
                    
                logger.info(f"🔍 DEBUGGING: Final web3_analysis keys: {list(purchase.web3_analysis.keys())}")
                    
                # Ensure critical fields are properly stored with correct types
                for field, field_type in [
                    ('token_age_hours', float),
                    ('holder_count', int),
                    ('gini_coefficient', float),
                    ('top_10_holders_percentage', float),
                    ('top_holder_percentage', float),
                    ('is_verified', bool),
                    ('has_liquidity', bool),
                    ('liquidity_usd', float),
                    ('honeypot_risk', float)
                ]:
                    if field in intelligence and intelligence[field] is not None:
                        try:
                            purchase.web3_analysis[field] = field_type(intelligence[field])
                            if field in ['token_age_hours', 'holder_count', 'gini_coefficient']:
                                logger.info(f"✅ DEBUGGING: {field} stored for {token}: {intelligence[field]}")
                        except (ValueError, TypeError) as e:
                            logger.error(f"❌ DEBUGGING: Error converting {field} for {token}: {e}")
                
                # Store string fields safely
                if 'holder_concentration_risk' in intelligence:
                    purchase.web3_analysis['holder_concentration_risk'] = str(intelligence['holder_concentration_risk'])
                    logger.info(f"✅ DEBUGGING: holder_concentration_risk stored for {token}: {intelligence['holder_concentration_risk']}")
                
                logger.info(f"🔍 Comprehensive Web3 intelligence applied to {token}: {len(purchase.web3_analysis)} fields")
            else:
                logger.warning(f"❌ DEBUGGING: No Web3 intelligence found for token {token}")

           
    async def _run_enhanced_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Run the enhanced AI analysis with Web3-enriched data"""
        try:
            # Step 1: Create enhanced DataFrame with Web3 data
            df = self._create_enhanced_dataframe_with_web3(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Step 2: Run core analyses
            analyses = await self._run_core_analyses(df)
            
            # Step 3: Create enhanced scores with Web3 bonuses
            enhanced_scores = self._create_enhanced_scores_with_web3(analyses, df)
            
            # Step 4: Build result
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe_with_web3(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame with Web3 intelligence data - FIXED KeyError issues"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                # Extract Web3 intelligence with safe defaults
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # SAFE extraction of Web3 fields with defaults
                is_verified = web3_data.get('is_verified', False)
                has_liquidity = web3_data.get('has_liquidity', False)
                liquidity_usd = web3_data.get('liquidity_usd', 0)
                honeypot_risk = web3_data.get('honeypot_risk', 0.3)
                
                # FIXED: Safe token age extraction
                token_age_hours = web3_data.get('token_age_hours', 999999)  # Default to old token
                if token_age_hours is None:
                    token_age_hours = 999999
                
                # FIXED: Safe holder count extraction
                holder_count = web3_data.get('holder_count', 0)
                if holder_count is None:
                    holder_count = 0
                
                # FIXED: Safe smart money extraction with proper defaults
                smart_money_buying = web3_data.get('smart_money_buying', False)
                whale_accumulation = web3_data.get('whale_accumulation', False)
                
                # Core data with Web3 enhancements
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 intelligence with SAFE defaults
                    'is_verified': bool(is_verified),
                    'has_liquidity': bool(has_liquidity),
                    'liquidity_usd': float(liquidity_usd),
                    'honeypot_risk': float(honeypot_risk),
                    'smart_money_buying': bool(smart_money_buying),
                    'whale_accumulation': bool(whale_accumulation),
                    'has_coingecko_listing': bool(web3_data.get('has_coingecko_listing', False)),
                    'data_sources_count': len(web3_data.get('data_sources', [])),                    
                    'token_age_hours': float(token_age_hours),
                    'holder_count': int(holder_count)
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase for AI: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features with SAFE operations
        df['log_eth'] = np.log1p(df['eth_value'])
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > 200  # Fixed threshold
        df['has_api_data'] = df['data_sources_count'] > 0
        df['is_high_liquidity'] = df['liquidity_usd'] > 10000
        
        # FIXED: Safe new token detection
        df['is_new_token'] = df['token_age_hours'] < 168  # Less than 1 week
        
        # FIXED: Create has_smart_money column safely - THIS WAS THE ISSUE
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        # Log Web3 intelligence statistics
        verified_count = df['is_verified'].sum()
        liquidity_count = df['has_liquidity'].sum()
        api_data_count = df['has_api_data'].sum()
        new_token_count = df['is_new_token'].sum()
        smart_money_count = df['has_smart_money'].sum()  # Now safe to use
        
        logger.info(f"🤖 AI DataFrame with Web3: {len(df)} purchases")
        logger.info(f"✅ Verified: {verified_count}/{len(df)} ({verified_count/len(df)*100:.1f}%)")
        logger.info(f"💧 With liquidity: {liquidity_count}/{len(df)} ({liquidity_count/len(df)*100:.1f}%)")
        logger.info(f"🔍 API data: {api_data_count}/{len(df)} ({api_data_count/len(df)*100:.1f}%)")
        logger.info(f"🆕 New tokens: {new_token_count}/{len(df)} ({new_token_count/len(df)*100:.1f}%)")
        logger.info(f"🧠 Smart money: {smart_money_count}/{len(df)} ({smart_money_count/len(df)*100:.1f}%)")
        
        return df

    def _create_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame - FIXED has_smart_money error"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                # Extract Web3 data efficiently with SAFE defaults
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # Core data with Web3 enhancements
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 data with SAFE defaults
                    'token_age_hours': float(web3_data.get('token_age_hours', 999999)),
                    'holder_count': int(web3_data.get('holder_count', 0)),
                    'is_verified': bool(web3_data.get('is_verified', False)),
                    'honeypot_risk': float(web3_data.get('honeypot_risk', 0.3)),
                    'smart_money_buying': bool(web3_data.get('smart_money_buying', False)),
                    'whale_accumulation': bool(web3_data.get('whale_accumulation', False)),
                    'has_liquidity': bool(web3_data.get('has_liquidity', False))
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features efficiently with SAFE operations
        df['log_eth'] = np.log1p(df['eth_value'])
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > 200
        df['is_new_token'] = df['token_age_hours'] < 168
        
        # FIXED: Create has_smart_money column safely
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        return df


    def _create_enhanced_scores_with_web3(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores with Web3 intelligence bonuses"""
        enhanced_scores = {}
        
        # Group by token for scoring
        for token, token_df in df.groupby('token'):
            try:
                # Basic metrics
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['wallet_score'].mean()
                
                # Component scores
                volume_score = min(total_eth * 40, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 200) * 20, 20)
                
                # WEB3 INTELLIGENCE BONUSES
                web3_bonus = 0
                
                # Verification bonus (major impact)
                if token_df['is_verified'].any():
                    web3_bonus += 15
                    logger.debug(f"✅ {token}: +15 points for verified contract")
                
                # Liquidity bonus
                if token_df['has_liquidity'].any():
                    web3_bonus += 10
                    logger.debug(f"💧 {token}: +10 points for liquidity")
                
                # High liquidity bonus
                if token_df['is_high_liquidity'].any():
                    web3_bonus += 5
                    logger.debug(f"💰 {token}: +5 points for high liquidity")
                
                # API data bonus
                if token_df['has_api_data'].any():
                    web3_bonus += 3
                    logger.debug(f"🔍 {token}: +3 points for API data")
                
                # CoinGecko listing bonus
                if token_df['has_coingecko_listing'].any():
                    web3_bonus += 7
                    logger.debug(f"🦎 {token}: +7 points for CoinGecko listing")
                
                # Risk penalty
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
                risk_penalty = avg_honeypot_risk * 20
                
                if risk_penalty > 3:
                    logger.debug(f"⚠️ {token}: -{risk_penalty:.1f} points for risk")
                
                # AI multiplier from analyses
                ai_multiplier = 1.0
                if analyses.get('whale_coordination', {}).get('detected'):
                    ai_multiplier += analyses['whale_coordination']['score'] * 0.3
                if analyses.get('pump_signals', {}).get('detected'):
                    ai_multiplier += analyses['pump_signals']['score'] * 0.4
                
                # Final score calculation
                base_score = volume_score + diversity_score + quality_score + web3_bonus - risk_penalty
                final_score = base_score * ai_multiplier
                
                # Store enhanced score data
                enhanced_scores[token] = {
                    'total_score': final_score,
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'web3_bonus': web3_bonus,
                    'risk_penalty': risk_penalty,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'confidence': min(0.95, 0.6 + (ai_multiplier - 1) * 0.2),
                    
                    # Web3 intelligence metadata
                    'is_verified': token_df['is_verified'].any(),
                    'has_liquidity': token_df['has_liquidity'].any(),
                    'liquidity_usd': token_df['liquidity_usd'].max(),
                    'honeypot_risk': avg_honeypot_risk,
                    'has_api_data': token_df['has_api_data'].any(),
                    'has_coingecko_listing': token_df['has_coingecko_listing'].any()
                }
                
                if web3_bonus > 0:
                    logger.info(f"🚀 {token}: Total={final_score:.1f} (Web3 bonus: +{web3_bonus})")
                
            except Exception as e:
                logger.debug(f"Error scoring token {token}: {e}")
                continue
        
        return enhanced_scores
    
    async def _get_web3_session(self):
        """Get HTTP session for Web3 API calls"""
        if not self._web3_session:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            headers = {
                'User-Agent': 'CryptoAnalysis-AI/1.0',
                'Accept': 'application/json'
            }
            self._web3_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=8)
            )
        return self._web3_session
    
    def _default_web3_intelligence(self, token_symbol: str, network: str) -> Dict:
        """Default Web3 intelligence when everything fails"""
        return {
            'contract_address': '',
            'ca': '',
            'token_symbol': token_symbol,
            'network': network,
            'is_verified': False,
            'has_liquidity': False,
            'liquidity_usd': 0,
            'honeypot_risk': 0.4,
            'smart_money_buying': False,
            'whale_accumulation': False,
            'data_sources': [],
            'error': 'no_data_available'
        }
    
    # ============================================================================
    # YOUR EXISTING AI METHODS (keep these as they are)
    # ============================================================================
    
    async def complete_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Your existing AI analysis method - keep this unchanged"""
        # This is your original method - keep it as is
        # It will be called by complete_ai_analysis_with_web3 after Web3 processing
        try:
            logger.info(f"🤖 AI ANALYSIS: {len(purchases)} {analysis_type} transactions")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Create enhanced DataFrame (your existing method)
            df = self._create_enhanced_dataframe(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Step 2: Run core AI analyses (your existing method)
            analyses = await self._run_core_analyses(df)
            
            # Step 3: Create enhanced scores (your existing method) 
            enhanced_scores = self._create_enhanced_scores(analyses, df)
            
            # Step 4: Build comprehensive result (your existing method)
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            logger.info(f"✅ AI SUCCESS: {len(enhanced_scores)} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame - more efficient approach"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                # Extract Web3 data efficiently
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # Core data with Web3 enhancements
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 data (with defaults)
                    'token_age_hours': web3_data.get('token_age_hours', 999999),
                    'holder_count': web3_data.get('holder_count', 0),
                    'is_verified': web3_data.get('is_verified', False),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0),
                    'smart_money_buying': web3_data.get('smart_money_buying', False),
                    'whale_accumulation': web3_data.get('whale_accumulation', False),
                    'has_liquidity': len(web3_data.get('liquidity_pools', [])) > 0
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features efficiently
        df['log_eth'] = np.log1p(df['eth_value'])
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > self.thresholds['smart_money_threshold']
        df['is_new_token'] = df['token_age_hours'] < 168  # 1 week
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        return df
    
    async def _run_core_analyses(self, df: pd.DataFrame) -> Dict:
        """Run core AI analyses - streamlined but comprehensive"""
        analyses = {}
        
        # Whale coordination detection
        analyses['whale_coordination'] = self._detect_whale_coordination(df)
        
        # Pump signal detection
        analyses['pump_signals'] = self._detect_pump_signals(df)
        
        # Smart money flow analysis
        analyses['smart_money_flow'] = self._analyze_smart_money_flow(df)
        
        # Momentum analysis
        analyses['momentum_analysis'] = self._analyze_momentum(df)
        
        # Risk assessment
        analyses['risk_assessment'] = self._assess_risks(df)
        
        # Anomaly detection (if enough data)
        if len(df) > 10:
            analyses['anomaly_detection'] = self._detect_anomalies(df)
        else:
            analyses['anomaly_detection'] = {'detected': False, 'score': 0}
        
        return analyses
    
    def _detect_whale_coordination(self, df: pd.DataFrame) -> Dict:
        """Detect whale coordination - simplified but effective"""
        if df.empty or len(df) < 3:
            return {'detected': False, 'score': 0}
        
        # Group by token for analysis
        coordination_scores = []
        
        for token, token_df in df.groupby('token'):
            if len(token_df) < 2:
                continue
            
            # Whale metrics
            whale_transactions = token_df[token_df['is_whale']]
            whale_ratio = len(whale_transactions) / len(token_df)
            
            # Timing coordination (transactions close in time)
            if len(token_df) > 1:
                time_diffs = token_df['unix_time'].diff().dropna()
                avg_time_diff = time_diffs.mean() / 3600  # Convert to hours
                timing_score = 1.0 / (1.0 + avg_time_diff) if avg_time_diff > 0 else 0
            else:
                timing_score = 0
            
            # Wallet quality correlation
            high_score_ratio = len(token_df[token_df['is_smart_wallet']]) / len(token_df)
            
            # Combined coordination score
            coord_score = (whale_ratio * 0.4 + timing_score * 0.3 + high_score_ratio * 0.3)
            coordination_scores.append(coord_score)
        
        avg_coordination = np.mean(coordination_scores) if coordination_scores else 0
        detected = avg_coordination >= self.thresholds['whale_threshold']
        
        return {
            'detected': detected,
            'score': avg_coordination,
            'evidence_strength': 'STRONG' if avg_coordination > 0.8 else 'MODERATE' if avg_coordination > 0.5 else 'WEAK'
        }
    
    def _detect_pump_signals(self, df: pd.DataFrame) -> Dict:
        """Detect pump signals - FIXED to handle missing columns"""
        if df.empty or len(df) < 5:
            return {'detected': False, 'score': 0}
        
        df_sorted = df.sort_values('unix_time')
        
        # Volume acceleration
        mid_point = len(df_sorted) // 2
        early_volume = df_sorted.iloc[:mid_point]['eth_value'].sum()
        recent_volume = df_sorted.iloc[mid_point:]['eth_value'].sum()
        volume_acceleration = recent_volume / early_volume if early_volume > 0 else 1
        
        # Wallet growth
        early_wallets = df_sorted.iloc[:mid_point]['wallet'].nunique()
        recent_wallets = df_sorted.iloc[mid_point:]['wallet'].nunique()
        wallet_growth = recent_wallets / early_wallets if early_wallets > 0 else 1
        
        # Smart money involvement
        smart_ratio = len(df[df['is_smart_wallet']]) / len(df)
        
        # FIXED: Safe check for is_new_token column
        if 'is_new_token' in df.columns:
            new_token_ratio = len(df[df['is_new_token']]) / len(df)
        else:
            # Fallback: calculate based on token_age_hours
            new_token_ratio = len(df[df['token_age_hours'] < 168]) / len(df) if 'token_age_hours' in df.columns else 0
        
        # FIXED: Safe check for has_smart_money column (this was causing the error)
        smart_money_active = 0
        if 'has_smart_money' in df.columns:
            smart_money_active = 0.15 if df['has_smart_money'].any() else 0
        else:
            # Fallback: check individual components
            if 'smart_money_buying' in df.columns and 'whale_accumulation' in df.columns:
                smart_money_active = 0.15 if (df['smart_money_buying'].any() or df['whale_accumulation'].any()) else 0
        
        # Calculate pump score
        pump_score = (
            min((volume_acceleration - 1) * 0.3, 0.3) +
            min((wallet_growth - 1) * 0.2, 0.2) +
            min(smart_ratio * 0.2, 0.2) +
            (0.15 if new_token_ratio > 0.5 else 0) +
            smart_money_active  # Now safely calculated
        )
        
        detected = pump_score >= 0.65  # threshold
        
        return {
            'detected': detected,
            'score': pump_score,
            'volume_acceleration': volume_acceleration,
            'wallet_growth': wallet_growth,
            'phase': self._get_pump_phase(pump_score),
            'confidence': min(pump_score * 1.2, 1.0)
        }
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze smart money flow - FIXED column access"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_trades = df[df['is_smart_wallet']]
        smart_volume = smart_trades['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        # FIXED: Safe smart money signals check
        smart_money_buying = 0
        whale_accumulation = 0
        
        if 'smart_money_buying' in df.columns:
            smart_money_buying = df['smart_money_buying'].sum()
        if 'whale_accumulation' in df.columns:
            whale_accumulation = df['whale_accumulation'].sum()
        
        # Determine flow direction
        if smart_ratio > 0.4 or smart_money_buying > 0:
            direction = 'STRONG_BULLISH' if smart_ratio > 0.6 else 'BULLISH'
        elif smart_ratio < 0.05:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'flow_direction': direction,
            'confidence': min(smart_ratio * 2, 1.0),
            'smart_money_ratio': smart_ratio,
            'smart_wallet_count': len(smart_trades),
            'whale_accumulation_detected': whale_accumulation > 0
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum - simplified but effective"""
        if df.empty or len(df) < 3:
            return {'momentum_detected': False, 'strength': 0}
        
        momentum_signals = {}
        
        for token, token_df in df.groupby('token'):
            if len(token_df) < 2:
                continue
            
            token_df = token_df.sort_values('unix_time').reset_index(drop=True)
            
            # Volume momentum
            if len(token_df) >= 4:
                mid = len(token_df) // 2
                early_vol = token_df.iloc[:mid]['eth_value'].mean()
                recent_vol = token_df.iloc[mid:]['eth_value'].mean()
                vol_momentum = recent_vol / early_vol if early_vol > 0 else 1
            else:
                vol_momentum = 1
            
            # Wallet momentum
            wallet_momentum = token_df['wallet'].nunique() / len(token_df)
            
            # Time momentum (frequency)
            time_span = (token_df['unix_time'].max() - token_df['unix_time'].min()) / 3600  # hours
            frequency = len(token_df) / max(time_span, 1)
            
            # Combined momentum
            momentum_score = (
                min((vol_momentum - 1) * 0.4, 0.4) +
                min(wallet_momentum * 0.3, 0.3) +
                min(frequency * 0.1, 0.3)
            )
            
            momentum_signals[token] = {
                'overall_score': momentum_score,
                'volume_momentum': vol_momentum,
                'wallet_momentum': wallet_momentum,
                'frequency': frequency
            }
        
        if momentum_signals:
            avg_momentum = np.mean([s['overall_score'] for s in momentum_signals.values()])
            momentum_detected = avg_momentum > 0.3
        else:
            avg_momentum = 0
            momentum_detected = False
        
        return {
            'momentum_detected': momentum_detected,
            'strength': avg_momentum,
            'token_momentum': momentum_signals
        }
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assess risks - comprehensive but efficient"""
        if df.empty:
            return {'overall_risk': 0.5, 'risk_level': 'MEDIUM'}
        
        risk_factors = {}
        
        # Concentration risk
        wallet_volumes = df.groupby('wallet')['eth_value'].sum()
        risk_factors['concentration'] = wallet_volumes.max() / wallet_volumes.sum() if wallet_volumes.sum() > 0 else 0
        
        # Timing risk (unusual hours)
        unusual_hours = [0, 1, 2, 3, 4, 5]
        risk_factors['timing'] = len(df[df['hour'].isin(unusual_hours)]) / len(df)
        
        # Honeypot risk
        risk_factors['honeypot'] = df['honeypot_risk'].mean()
        
        # New token risk
        risk_factors['new_token'] = len(df[df['is_new_token']]) / len(df)
        
        # Liquidity risk
        risk_factors['no_liquidity'] = len(df[~df['has_liquidity']]) / len(df)
        
        # Calculate overall risk
        overall_risk = (
            risk_factors['concentration'] * 0.25 +
            risk_factors['timing'] * 0.15 +
            risk_factors['honeypot'] * 0.25 +
            risk_factors['new_token'] * 0.15 +
            risk_factors['no_liquidity'] * 0.20
        )
        
        risk_level = 'HIGH' if overall_risk >= 0.7 else 'MEDIUM' if overall_risk >= 0.4 else 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies using ML - simplified approach"""
        try:
            # Prepare features for anomaly detection
            features = df[['eth_value', 'wallet_score', 'hour']].copy()
            features['log_eth'] = np.log1p(features['eth_value'])
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features[['log_eth', 'wallet_score', 'hour']])
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
            anomaly_ratio = (anomaly_scores == -1).sum() / len(anomaly_scores)
            
            detected = anomaly_ratio > self.thresholds['anomaly_threshold']
            
            return {
                'detected': detected,
                'anomaly_ratio': anomaly_ratio,
                'anomaly_count': (anomaly_scores == -1).sum()
            }
            
        except Exception as e:
            logger.debug(f"Anomaly detection failed: {e}")
            return {'detected': False, 'anomaly_ratio': 0}
    
    def _create_enhanced_scores(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores - streamlined approach"""
        enhanced_scores = {}
        
        # Group by token for scoring
        for token, token_df in df.groupby('token'):
            try:
                # Basic metrics
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['wallet_score'].mean()
                
                # Component scores
                volume_score = min(total_eth * 40, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 200) * 20, 20)
                
                # Web3 enhancements
                web3_bonus = 0
                if token_df['is_verified'].any():
                    web3_bonus += 5
                if token_df['has_liquidity'].any():
                    web3_bonus += 5
                if token_df['has_smart_money'].any():
                    web3_bonus += 10
                
                # Risk penalty
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
                risk_penalty = avg_honeypot_risk * 15
                
                # AI multiplier
                ai_multiplier = 1.0
                if analyses['whale_coordination']['detected']:
                    ai_multiplier += analyses['whale_coordination']['score'] * 0.3
                if analyses['pump_signals']['detected']:
                    ai_multiplier += analyses['pump_signals']['score'] * 0.4
                
                # Final score
                base_score = volume_score + diversity_score + quality_score + web3_bonus - risk_penalty
                final_score = base_score * ai_multiplier
                
                enhanced_scores[token] = {
                    'total_score': final_score,
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'web3_bonus': web3_bonus,
                    'risk_penalty': risk_penalty,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'confidence': min(0.9, 0.6 + (ai_multiplier - 1) * 0.2),
                    'honeypot_risk': avg_honeypot_risk,
                    'is_verified': token_df['is_verified'].any(),
                    'has_liquidity': token_df['has_liquidity'].any(),
                    'has_smart_money': token_df['has_smart_money'].any()
                }
                
            except Exception as e:
                logger.debug(f"Error scoring token {token}: {e}")
                continue
        
        return enhanced_scores
    
    def _build_enhanced_result(self, analyses: Dict, enhanced_scores: Dict, analysis_type: str) -> Dict:
        """Build enhanced result - comprehensive"""
        return {
            'scores': enhanced_scores,
            'analysis_type': analysis_type,
            'enhanced': True,
            'web3_enhanced': True,
            'ai_analyses': analyses,
            'analysis_summary': {
                'total_tokens': len(enhanced_scores),
                'ai_patterns_detected': sum(1 for analysis in analyses.values() 
                                          if isinstance(analysis, dict) and analysis.get('detected', False)),
                'avg_confidence': np.mean([s['confidence'] for s in enhanced_scores.values()]) if enhanced_scores else 0,
                'high_confidence_tokens': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.85),
                'verified_tokens': sum(1 for s in enhanced_scores.values() if s.get('is_verified')),
                'tokens_with_liquidity': sum(1 for s in enhanced_scores.values() if s.get('has_liquidity')),
                'honeypot_warnings': sum(1 for s in enhanced_scores.values() if s.get('honeypot_risk', 0) > 0.5),
                'smart_money_tokens': sum(1 for s in enhanced_scores.values() if s.get('has_smart_money'))
            }
        }
    
    def _get_pump_phase(self, score: float) -> str:
        """Determine pump phase"""
        if score >= 0.8:
            return "PUMP_IMMINENT"
        elif score >= 0.6:
            return "LATE_ACCUMULATION" 
        elif score >= 0.4:
            return "EARLY_ACCUMULATION"
        else:
            return "NORMAL"
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Create empty result"""
        return {
            'scores': {},
            'analysis_type': analysis_type,
            'enhanced': False,
            'web3_enhanced': False,
            'error': 'No data to analyze'
        }
    
    async def cleanup(self):
        """Cleanup Web3 session"""
        if self._web3_session:
            await self._web3_session.close()
            self._web3_session = None