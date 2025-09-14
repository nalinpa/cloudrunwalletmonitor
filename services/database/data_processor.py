import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio

from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType

logger = logging.getLogger(__name__)

# LAZY AI LOADING - No direct import to avoid circular dependencies
AI_AVAILABLE = None  # Will be determined on first use
AdvancedCryptoAI_CLASS = None

# Optional Web3 integration
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

class UnifiedDataProcessor:
    """
    Unified Data Processor - combines data_processor.py + data_processor_base.py
    Handles all data processing, Web3 integration, and AI analysis in one place
    """
    
    def __init__(self):
        # Token exclusion lists (simplified)
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
        self._ai_enabled = None  # Will be determined on first use
        
        # Web3 system (lazy loaded)
        self.w3_connections = {}
        self.web3_enabled = WEB3_AVAILABLE
        self._web3_cache = {}
        
        # Performance tracking
        self.stats = {
            'transfers_processed': 0,
            'transfers_stored': 0,
            'ai_enhanced_tokens': 0,
            'web3_enriched_tokens': 0
        }
        
        logger.info("🚀 Unified Data Processor initialized")
        logger.info(f"  AI: ⏳ Will test on first use (lazy loading)")
        logger.info(f"  Web3: {'✓ Available' if WEB3_AVAILABLE else '✗ Not available'}")
    
    def set_transfer_service(self, transfer_service):
        """Set BigQuery transfer service"""
        self.bigquery_transfer_service = transfer_service
        logger.info("✅ BigQuery transfer service connected")
    
    def _test_ai_availability(self):
        """Test AI availability on first use to avoid circular imports"""
        global AI_AVAILABLE, AdvancedCryptoAI_CLASS
        
        if AI_AVAILABLE is not None:
            return AI_AVAILABLE
        
        try:
            logger.info("🔍 Testing AI system availability (lazy load)...")
            
            # Test core dependencies first
            import sklearn
            import numpy as np
            import pandas as pd
            logger.info("✅ AI dependencies available")
            
            # Now try to import the AI class
            from core.analysis.ai_system import AdvancedCryptoAI
            logger.info("✅ AI class imported successfully")
            
            # Test if it can be instantiated
            test_instance = AdvancedCryptoAI()
            logger.info("✅ AI instance created successfully")
            
            AI_AVAILABLE = True
            AdvancedCryptoAI_CLASS = AdvancedCryptoAI
            self._ai_enabled = True
            
            logger.info("🤖 AI system fully operational (lazy loaded)")
            return True
            
        except ImportError as e:
            logger.info(f"📊 AI dependencies missing: {e}")
            AI_AVAILABLE = False
            AdvancedCryptoAI_CLASS = None
            self._ai_enabled = False
            return False
            
        except Exception as e:
            logger.warning(f"❌ AI system failed: {e}")
            AI_AVAILABLE = False
            AdvancedCryptoAI_CLASS = None
            self._ai_enabled = False
            return False

    def _get_ai_engine(self):
        """Lazy load AI engine to avoid circular imports"""
        # Test availability on first use
        if self._ai_enabled is None:
            self._test_ai_availability()
        
        # If AI is not available, return None
        if not self._ai_enabled:
            return None
        
        # Create engine if needed
        if self.ai_engine is None:
            try:
                logger.info("🤖 Creating AI engine instance...")
                self.ai_engine = AdvancedCryptoAI_CLASS()
                logger.info("✅ AI engine created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create AI engine: {e}")
                self._ai_enabled = False
                self.ai_engine = None
        
        return self.ai_engine
    
    # ============================================================================
    # TOKEN FILTERING & VALIDATION
    # ============================================================================
    
    def is_excluded_token(self, asset: str, contract_address: str = None) -> bool:
        """Check if token should be excluded - simplified logic"""
        if not asset:
            return True
            
        asset_upper = asset.upper()
        
        # Check excluded assets
        if asset_upper in self.excluded_assets:
            return True
        
        # Check excluded contracts
        if contract_address and contract_address.lower() in self.excluded_contracts:
            return True
        
        # Check stablecoin patterns
        if len(asset) <= 6 and any(stable in asset_upper for stable in ['USD', 'DAI']):
            return True
        
        return False
    
    def extract_contract_address(self, transfer: Dict) -> str:
        """Extract contract address - unified approach"""
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
    
    # ============================================================================
    # ETH CALCULATION - UNIFIED APPROACH
    # ============================================================================
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH spent - unified efficient approach"""
        if not outgoing_transfers:
            return 0.0
        
        # Currency conversion rates
        spending_currencies = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,  # ~$2400/ETH
            'USDC': 1/2400,
            'AERO': 1/4800,  # Base network token
        }
        
        total_eth = 0.0
        
        # Step 1: Exact transaction match
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
        
        # Step 2: Block-based matching
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
        
        if total_eth > 0:
            return total_eth
        
        # Step 3: Proximity matching (within 5 blocks)
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            return 0.0
        
        proximity_values = []
        for transfer in outgoing_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                if abs(transfer_block_num - target_block_num) <= 5:
                    asset = transfer.get("asset", "")
                    if asset in spending_currencies:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * spending_currencies[asset]
                        if 0.0001 <= eth_equivalent <= 50.0:
                            proximity_values.append(eth_equivalent)
            except (ValueError, TypeError):
                continue
        
        return sum(proximity_values)
    
    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """Calculate ETH received for sells - unified approach"""
        if not incoming_transfers:
            return 0.0
        
        # Currencies representing ETH value received
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
        
        # Step 1: Exact transaction match
        total_eth = 0.0
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
        
        # Step 2: Block proximity matching (expanded range for sells)
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            return 0.0
        
        proximity_values = []
        for transfer in incoming_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                if abs(transfer_block_num - target_block_num) <= 10:  # Expanded range for sells
                    asset = transfer.get("asset", "")
                    if asset in receiving_currencies:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * receiving_currencies[asset]
                        if 0.00001 <= eth_equivalent <= 100.0:  # Lower threshold for sells
                            proximity_values.append(eth_equivalent)
            except (ValueError, TypeError):
                continue
        
        if proximity_values:
            return sum(proximity_values)
        
        # Step 3: Fallback - find any ETH receipts
        all_eth_received = []
        for transfer in incoming_transfers:
            asset = transfer.get("asset", "")
            if asset in receiving_currencies:
                try:
                    amount = float(transfer.get("value", "0"))
                    eth_equivalent = amount * receiving_currencies[asset]
                    if 0.000001 <= eth_equivalent <= 50.0:
                        all_eth_received.append(eth_equivalent)
                except (ValueError, TypeError):
                    continue
        
        if all_eth_received:
            # Use the largest receipt as most likely sell proceeds
            return max(all_eth_received)
        
        return 0.0
    
    # ============================================================================
    # MAIN PROCESSING METHODS - UNIFIED
    # ============================================================================
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """Process transfers to purchases - unified BUY analysis"""
        purchases = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for BUY analysis on {network}")
        logger.info(f"Storage: {'ENABLED' if store_data else 'DISABLED'}")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            incoming = transfers.get('incoming', [])
            outgoing = transfers.get('outgoing', [])
            
            # Process INCOMING transfers as potential BUYS
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
                    
                    # Calculate ETH spent
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # Create transfer record for storage (if enabled)
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
                    
                    # Create purchase for analysis (always done)
                    if not self.is_excluded_token(asset, contract_address) and eth_spent >= 0.00001:
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
                            }
                        )
                        purchases.append(purchase)
                
                except Exception as e:
                    logger.debug(f"Error processing incoming transfer: {e}")
                    continue
        
        # Store transfers to BigQuery if enabled
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                self.stats['transfers_stored'] = stored_count
                logger.info(f"✅ Stored {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"❌ Failed to store transfers: {e}")
                self._last_stored_count = 0
        else:
            self._last_stored_count = 0
            if store_data:
                logger.warning("⚠️ Storage requested but not available")
        
        self.stats['transfers_processed'] = len(all_transfer_records)
        
        # Log contract address extraction
        ca_count = sum(1 for p in purchases if p.web3_analysis and p.web3_analysis.get('contract_address'))
        logger.info(f"BUY: {len(purchases)} purchases, {ca_count} with contract addresses")
        
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """Process transfers to sells - unified SELL analysis"""
        sells = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing {len(wallets)} wallets for SELL analysis on {network}")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            outgoing = transfers.get('outgoing', [])
            incoming = transfers.get('incoming', [])
            
            # Process OUTGOING transfers as potential SELLS
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
                    
                    # Calculate ETH received from sell
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    # Create transfer record for storage (if enabled)
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
                    
                    # Create sell for analysis (always done)
                    if not self.is_excluded_token(asset, contract_address) and eth_received >= 0.000001:
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
                
                except Exception as e:
                    logger.debug(f"Error processing sell transfer: {e}")
                    continue
        
        # Store transfers if enabled
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ Stored {stored_count} sell transfer records")
            except Exception as e:
                logger.error(f"❌ Failed to store sell transfers: {e}")
                self._last_stored_count = 0
        
        logger.info(f"SELL: {len(sells)} sells identified")
        return sells
    
    # ============================================================================
    # ANALYSIS METHODS - UNIFIED
    # ============================================================================
    
    async def analyze_purchases_enhanced(self, purchases: List, analysis_type: str) -> Dict:
        """Enhanced analysis with AI - unified approach with lazy loading"""
        if not purchases:
            return self._create_empty_result(analysis_type)
        
        logger.info(f"🔍 Analyzing {len(purchases)} {analysis_type} transactions")
        
        # Step 1: Try AI analysis if available (lazy loaded)
        ai_engine = self._get_ai_engine()
        if ai_engine:
            try:
                logger.info("🤖 Running AI-enhanced analysis...")
                result = await ai_engine.complete_ai_analysis(purchases, analysis_type)
                
                if result.get('enhanced'):
                    enhanced_result = self._create_result_with_contracts(result, purchases, analysis_type)
                    self.stats['ai_enhanced_tokens'] = len(result.get('scores', {}))
                    logger.info(f"✅ AI analysis complete: {self.stats['ai_enhanced_tokens']} tokens enhanced")
                    return enhanced_result
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
        
        # Step 2: Fallback to basic analysis
        logger.info("📊 Using basic analysis...")
        basic_result = self._analyze_purchases_basic(purchases, analysis_type)
        return self._create_result_with_contracts(basic_result, purchases, analysis_type)
    
    def _analyze_purchases_basic(self, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Basic analysis fallback - efficient pandas approach"""
        try:
            # Create DataFrame efficiently
            data = []
            for p in purchases:
                eth_value = p.amount_received if analysis_type == 'sell' else p.eth_spent
                contract_address = ''
                if p.web3_analysis:
                    contract_address = p.web3_analysis.get('contract_address', '') or p.web3_analysis.get('ca', '')
                
                data.append({
                    'token': p.token_bought,
                    'eth_value': eth_value,
                    'wallet': p.wallet_address,
                    'score': p.sophistication_score or 0,
                    'contract': contract_address
                })
            
            df = pd.DataFrame(data)
            
            # Aggregate by token
            token_stats = df.groupby('token').agg({
                'eth_value': ['sum', 'mean', 'count'],
                'wallet': 'nunique',
                'score': 'mean'
            }).round(4)
            
            token_stats.columns = ['total_value', 'mean_value', 'tx_count', 'unique_wallets', 'avg_score']
            
            # Calculate scores
            scores = {}
            for token in token_stats.index:
                stats = token_stats.loc[token]
                
                if analysis_type == 'sell':
                    volume_score = min(stats['total_value'] * 100, 60)
                    diversity_score = min(stats['unique_wallets'] * 10, 25)
                    quality_score = min((stats['avg_score'] / 100) * 15, 15)
                else:
                    volume_score = min(stats['total_value'] * 50, 50)
                    diversity_score = min(stats['unique_wallets'] * 8, 30)
                    quality_score = min((stats['avg_score'] / 100) * 20, 20)
                
                total_score = volume_score + diversity_score + quality_score
                
                scores[token] = {
                    'total_score': float(total_score),
                    'volume_score': float(volume_score),
                    'diversity_score': float(diversity_score),
                    'quality_score': float(quality_score),
                    'ai_enhanced': False,
                    'confidence': 0.75
                }
            
            return {
                'scores': scores,
                'analysis_type': analysis_type,
                'enhanced': False
            }
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            return {'scores': {}, 'analysis_type': analysis_type, 'enhanced': False}
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def _parse_timestamp(self, transfer: Dict, block_number: int = None) -> datetime:
        """Parse timestamp from transfer - simplified"""
        if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata']:
            try:
                return datetime.fromisoformat(transfer['metadata']['blockTimestamp'].replace('Z', '+00:00'))
            except:
                pass
        
        # Fallback to current time
        return datetime.utcnow()
    
    def _create_result_with_contracts(self, analysis_results: Dict, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Create result with contract addresses - unified approach"""
        if not analysis_results or not analysis_results.get('scores'):
            return self._create_empty_result(analysis_type)
        
        scores = analysis_results['scores']
        ranked_tokens = []
        
        # Build contract lookup
        contract_lookup = {}
        purchase_stats = {}
        
        for purchase in purchases:
            token = purchase.token_bought
            
            # Contract lookup
            if purchase.web3_analysis:
                ca = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                if ca:
                    contract_lookup[token] = ca
            
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
            
            # Add AI enhancement data
            token_data.update({
                'ai_enhanced': score_data.get('ai_enhanced', False),
                'confidence': score_data.get('confidence', 0.75),
                'platforms': ['DEX'],
                'web3_data': {
                    'contract_address': contract_address,
                    'ca': contract_address,
                    'token_symbol': token,
                    'network': 'ethereum'  # Default
                }
            })
            
            ranked_tokens.append((token, token_data, score_data['total_score'], score_data))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate totals
        if analysis_type == 'sell':
            total_eth = sum(p.amount_received for p in purchases)
        else:
            total_eth = sum(p.eth_spent for p in purchases)
        
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        result = {
            'network': 'ethereum',  # Default
            'analysis_type': analysis_type,
            'total_transactions': len(purchases),
            'unique_tokens': unique_tokens,
            'total_eth_value': total_eth,
            'ranked_tokens': ranked_tokens,
            'performance_metrics': {
                **self.get_processing_stats(),
                'contract_addresses_extracted': sum(1 for _, data, _, _ in ranked_tokens if data.get('contract_address'))
            },
            'enhanced': analysis_results.get('enhanced', False),
            'scores': scores
        }
        
        return self._convert_numpy_types(result)
    
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
            'scores': {}
        }
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'transfers_processed': self.stats.get('transfers_processed', 0),
            'transfers_stored': self.stats.get('transfers_stored', 0),
            'last_stored_count': self._last_stored_count,
            'ai_enhanced_tokens': self.stats.get('ai_enhanced_tokens', 0),
            'ai_available': self._ai_enabled if self._ai_enabled is not None else False,
            'web3_available': self.web3_enabled,
            'excluded_assets_count': len(self.excluded_assets),
            'processing_mode': 'unified'
        }
    
    # ============================================================================
    # VALIDATION AND LOGGING METHODS
    # ============================================================================
    
    async def validate_data_quality(self, purchases: List[Purchase]) -> Dict:
        """Validate data quality and return quality report"""
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
        
        # Validate each purchase
        for purchase in purchases:
            is_valid = True
            
            # Check required fields
            if not purchase.token_bought:
                warnings.append(f"Purchase missing token_bought")
                is_valid = False
                
            if not purchase.transaction_hash:
                warnings.append(f"Purchase missing transaction_hash")
                is_valid = False
                
            # Check ETH values
            if hasattr(purchase, 'eth_spent') and purchase.eth_spent < 0:
                warnings.append(f"Negative ETH spent for {purchase.token_bought}")
                is_valid = False
                
            if hasattr(purchase, 'amount_received') and purchase.amount_received < 0:
                warnings.append(f"Negative amount received for {purchase.token_bought}")
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        # Calculate quality score
        quality_score = valid_count / len(purchases) if purchases else 0
        
        # Store for later reference
        self._last_quality_score = quality_score
        
        return {
            'data_quality_score': quality_score,
            'warnings': warnings[:10],  # Limit warnings
            'total_purchases': len(purchases),
            'valid_purchases': valid_count,
            'invalid_purchases': len(purchases) - valid_count,
            'quality_level': 'high' if quality_score >= 0.9 else 'medium' if quality_score >= 0.7 else 'low'
        }
    
    def log_token_analysis_summary(self, purchases: List[Purchase], analysis_type: str):
        """Log comprehensive token analysis summary"""
        if not purchases:
            logger.info(f"No {analysis_type} data to summarize")
            return
        
        # Group by token for summary
        token_groups = {}
        total_eth = 0
        
        for purchase in purchases:
            token = purchase.token_bought
            if token not in token_groups:
                token_groups[token] = {
                    'count': 0,
                    'eth_value': 0,
                    'wallets': set(),
                    'contracts': set()
                }
            
            token_groups[token]['count'] += 1
            token_groups[token]['wallets'].add(purchase.wallet_address)
            
            # ETH value based on analysis type
            if analysis_type == 'sell':
                eth_value = purchase.amount_received
            else:
                eth_value = purchase.eth_spent
                
            token_groups[token]['eth_value'] += eth_value
            total_eth += eth_value
            
            # Contract address
            if purchase.web3_analysis and purchase.web3_analysis.get('contract_address'):
                token_groups[token]['contracts'].add(purchase.web3_analysis['contract_address'])
        
        # Log summary
        logger.info(f"=== {analysis_type.upper()} ANALYSIS SUMMARY ===")
        logger.info(f"Total transactions: {len(purchases)}")
        logger.info(f"Unique tokens: {len(token_groups)}")
        logger.info(f"Total ETH {'received' if analysis_type == 'sell' else 'spent'}: {total_eth:.6f}")
        
        # Top tokens by ETH value
        top_tokens = sorted(token_groups.items(), key=lambda x: x[1]['eth_value'], reverse=True)[:5]
        logger.info("Top tokens by ETH volume:")
        for i, (token, data) in enumerate(top_tokens[:5]):
            logger.info(f"  {i+1}. {token}: {data['eth_value']:.6f} ETH, {data['count']} txs, {len(data['wallets'])} wallets")
        
        # Contract address stats
        tokens_with_contracts = sum(1 for data in token_groups.values() if data['contracts'])
        logger.info(f"Tokens with contract addresses: {tokens_with_contracts}/{len(token_groups)}")
        
        logger.info("=" * (len(f"{analysis_type.upper()} ANALYSIS SUMMARY") + 6))


# Export both names for backward compatibility
Web3DataProcessor = UnifiedDataProcessor  # Alias for backward compatibility
__all__ = ['UnifiedDataProcessor', 'Web3DataProcessor']