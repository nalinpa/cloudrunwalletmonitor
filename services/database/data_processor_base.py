import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from services.database.bigquery_client import BigQueryTransferService

logger = logging.getLogger(__name__)

# Import AI scoring if available
try:
    from core.analysis.ai_system import AdvancedCryptoAI
    AI_COMPLETE = True
    print("🚀 Advanced AI with pandas-ta + TextBlob loaded successfully!")
except ImportError as e:
    AI_COMPLETE = False
    print(f"⚠️ Advanced AI not available: {e}")

AI_SCORING_AVAILABLE = AI_COMPLETE

class DataProcessor:
    """Enhanced data processor with contract address extraction and AI-powered alpha scoring"""
    
    def __init__(self):
        # Token exclusion lists
        self.EXCLUDED_ASSETS = frozenset({
            'ETH', 'WETH', 'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'LUSD', 'USDC.E'
        })
        
        self.EXCLUDED_CONTRACTS = frozenset({
            '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
            '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
            '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
            '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH
            '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC on Base
        })
        
        # BigQuery transfer service for storing transfer records
        self.bigquery_transfer_service: BigQueryTransferService = None
        self._last_stored_count = 0
        
        # AI Enhancement - simplified
        self._enhanced_scoring_enabled = AI_COMPLETE
        
        if AI_COMPLETE:
            self.ai_engine = AdvancedCryptoAI()
            print("🤖 Complete AI Enhancement: ENABLED")
            print("✅ Features: Technical Analysis, Sentiment, ML, Whale Detection")
        else:
            self.ai_engine = None
            print("📊 Basic analysis only")
    
    def set_transfer_service(self, transfer_service: BigQueryTransferService):
        """Set the BigQuery transfer service for database operations"""
        self.bigquery_transfer_service = transfer_service
    
    def is_excluded_token(self, asset: str, contract_address: str = None) -> bool:
        """Check if token should be excluded"""
        if not asset:
            return True
            
        asset_upper = asset.upper()
        if asset_upper in self.EXCLUDED_ASSETS:
            return True
        
        if contract_address and contract_address.lower() in self.EXCLUDED_CONTRACTS:
            return True
        
        if len(asset) <= 6 and any(stable in asset_upper for stable in ['USD', 'DAI']):
            return True
        
        return False
    
    def extract_contract_address(self, transfer: Dict) -> str:
        """Enhanced contract address extraction from transfer data"""
        # Try multiple locations for contract address
        contract_address = ""
        
        # Method 1: Direct from rawContract
        raw_contract = transfer.get("rawContract", {})
        if raw_contract and isinstance(raw_contract, dict):
            contract_address = raw_contract.get("address", "")
        
        # Method 2: From contractAddress field
        if not contract_address:
            contract_address = transfer.get("contractAddress", "")
        
        # Method 3: From to/from addresses (for ERC20)
        if not contract_address:
            to_address = transfer.get("to", "")
            from_address = transfer.get("from", "")
            # Use 'to' address for incoming transfers as it might be the token contract
            if to_address and to_address.lower() != "0x0000000000000000000000000000000000000000":
                contract_address = to_address
        
        # Clean and validate
        if contract_address:
            contract_address = contract_address.strip().lower()
            if not contract_address.startswith('0x'):
                contract_address = '0x' + contract_address
            
            # Validate length (Ethereum addresses are 42 characters including 0x)
            if len(contract_address) == 42:
                return contract_address
        
        return ""
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """Process transfers to identify BUY transactions with enhanced contract address extraction"""
        purchases = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        storage_status = "ENABLED" if store_data else "DISABLED"
        logger.info(f"Processing transfers for BUY analysis on {network} (Storage: {storage_status})")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            incoming = transfers.get('incoming', [])
            outgoing = transfers.get('outgoing', [])
            
            # Process INCOMING transfers as potential BUYS
            for transfer in incoming:
                try:
                    asset = transfer.get("asset")
                    contract_address = self.extract_contract_address(transfer)
                    amount = float(transfer.get("value", "0"))
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    # Skip ETH transfers and invalid data
                    if not asset or asset == "ETH" or amount <= 0:
                        continue
                    
                    # Calculate ETH spent for this purchase
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # Create transfer record for ALL incoming ERC20 transfers (conditionally stored)
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
                    
                    # Create purchase for analysis (always done regardless of storage)
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
                                "ca": contract_address,  # Alternative field name
                                "token_symbol": asset,
                                "network": network
                            }
                        )
                        purchases.append(purchase)
                        
                except Exception as e:
                    logger.debug(f"Error processing incoming transfer: {e}")
                    continue
            
            # Process OUTGOING transfers for completeness (conditionally stored)
            for transfer in outgoing:
                try:
                    asset = transfer.get("asset")
                    contract_address = self.extract_contract_address(transfer)
                    amount = float(transfer.get("value", "0"))
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    # Skip ETH transfers and invalid data
                    if not asset or asset == "ETH" or amount <= 0:
                        continue
                    
                    # Calculate ETH received for sells
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    # Create transfer record for ALL outgoing ERC20 transfers (conditionally stored)
                    if store_data:
                        transfer_record = Transfer(
                            wallet_address=address,
                            token_address=contract_address,
                            transfer_type=TransferType.SELL,
                            timestamp=self._parse_timestamp(transfer, block_number),
                            cost_in_eth=eth_received,
                            transaction_hash=tx_hash,
                            block_number=block_number,
                            token_amount=amount,
                            token_symbol=asset,
                            network=network,
                            platform="Transfer",
                            wallet_sophistication_score=wallet_scores.get(address, 0)
                        )
                        all_transfer_records.append(transfer_record)
                    
                except Exception as e:
                    logger.debug(f"Error processing outgoing transfer: {e}")
                    continue
        
        # Store all transfers to BigQuery ONLY if store_data is True
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ STORED {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"❌ Failed to store transfer records to BigQuery: {e}")
                self._last_stored_count = 0
        elif store_data:
            logger.warning("⚠️ Storage requested but no transfer service available or no records")
            self._last_stored_count = 0
        else:
            logger.info(f"📊 STORAGE SKIPPED - store_data=False (analyzed {len(all_transfer_records)} transfer records)")
            self._last_stored_count = 0
        
        logger.info(f"BUY: Created {len(purchases)} purchases from {len(all_transfer_records)} total transfers")
        
        # Log contract address extraction stats
        purchases_with_ca = sum(1 for p in purchases if p.web3_analysis and p.web3_analysis.get('contract_address'))
        logger.info(f"Contract addresses extracted: {purchases_with_ca}/{len(purchases)} purchases")
        
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """Process transfers to identify SELL transactions with enhanced contract address extraction"""
        sells = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        storage_status = "ENABLED" if store_data else "DISABLED"
        logger.info(f"Processing transfers for SELL analysis on {network} (Storage: {storage_status})")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            outgoing = transfers.get('outgoing', [])
            incoming = transfers.get('incoming', [])
            
            # Process OUTGOING ERC20 transfers as potential SELLS
            for transfer in outgoing:
                try:
                    asset = transfer.get("asset")
                    contract_address = self.extract_contract_address(transfer)
                    amount_sold = float(transfer.get("value", "0"))
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    # Skip ETH transfers and invalid data
                    if not asset or asset == "ETH" or amount_sold <= 0:
                        continue
                    
                    # Calculate ETH received from sell
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    # Create transfer record for ALL outgoing ERC20 transfers (conditionally stored)
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
                    
                    # Create sell for analysis (always done regardless of storage)
                    if not self.is_excluded_token(asset, contract_address) and eth_received >= 0.000001:
                        sell = Purchase(
                            transaction_hash=tx_hash,
                            token_bought=asset,  # Token that was sold
                            amount_received=eth_received,  # ETH received from sell
                            eth_spent=0,  # Not applicable for sells
                            wallet_address=address,
                            platform="Transfer",
                            block_number=block_number,
                            timestamp=self._parse_timestamp(transfer, block_number),
                            sophistication_score=wallet_scores.get(address, 0),
                            web3_analysis={
                                "contract_address": contract_address,
                                "ca": contract_address,  # Alternative field name
                                "amount_sold": amount_sold,
                                "is_sell": True,
                                "token_symbol": asset,
                                "network": network
                            }
                        )
                        sells.append(sell)
                        
                        # DEBUG LOG to see what's being detected
                        logger.info(f"SELL DETECTED: {asset} amount_sold={amount_sold} eth_received={eth_received:.6f} CA={contract_address[:10]}... tx={tx_hash[:10]}...")
                        
                except Exception as e:
                    logger.debug(f"Error processing sell transfer: {e}")
                    continue
            
            # Also process incoming transfers for completeness (conditionally stored)
            for transfer in incoming:
                try:
                    asset = transfer.get("asset")
                    contract_address = self.extract_contract_address(transfer)
                    amount = float(transfer.get("value", "0"))
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    block_number = int(block_num, 16) if block_num != "0x0" else 0
                    
                    # Skip ETH transfers and invalid data
                    if not asset or asset == "ETH" or amount <= 0:
                        continue
                    
                    # Calculate ETH cost for this transaction
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # Create transfer record for ALL incoming ERC20 transfers (conditionally stored)
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
                    logger.debug(f"Error processing incoming transfer in sell analysis: {e}")
                    continue
        
        # Store all transfers to BigQuery ONLY if store_data is True
        if store_data and self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"✅ STORED {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"❌ Failed to store transfer records to BigQuery: {e}")
                self._last_stored_count = 0
        elif store_data:
            logger.warning("⚠️ Storage requested but no transfer service available or no records")
            self._last_stored_count = 0
        else:
            logger.info(f"📊 STORAGE SKIPPED - store_data=False (analyzed {len(all_transfer_records)} transfer records)")
            self._last_stored_count = 0
        
        logger.info(f"SELL: Created {len(sells)} sells from {len(all_transfer_records)} total transfers")
        
        # Log contract address extraction stats
        sells_with_ca = sum(1 for s in sells if s.web3_analysis and s.web3_analysis.get('contract_address'))
        logger.info(f"Contract addresses extracted: {sells_with_ca}/{len(sells)} sells")
        
        # Additional debug info for sells
        if sells:
            total_eth_received = sum(s.amount_received for s in sells)
            unique_tokens = set(s.token_bought for s in sells)
            logger.info(f"SELL SUMMARY: {total_eth_received:.6f} ETH received from selling {len(unique_tokens)} unique tokens")
            
            # Log top sells with contract addresses
            sells_sorted = sorted(sells, key=lambda x: x.amount_received, reverse=True)
            logger.info("Top 5 sells by ETH received:")
            for i, sell in enumerate(sells_sorted[:5]):
                ca = sell.web3_analysis.get('contract_address', 'No CA')[:10] if sell.web3_analysis else 'No CA'
                logger.info(f"  {i+1}. {sell.token_bought}: {sell.amount_received:.6f} ETH (CA: {ca}...)")
        else:
            logger.warning("No sells detected - check if _calculate_eth_received is working properly")
        
        return sells
    
    def _create_enhanced_result_with_ca(self, analysis_results: Dict, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Create enhanced analysis result ensuring contract addresses are included"""
        logger.info("Creating enhanced AI analysis result with contract addresses...")
        
        if not analysis_results:
            logger.error("Cannot create result - no analysis results")
            return self._create_empty_result(analysis_type)
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        is_enhanced = analysis_results.get('enhanced', False)
        
        logger.info(f"Token stats available: {token_stats is not None}")
        logger.info(f"Scores available: {len(scores)} tokens")
        logger.info(f"AI enhanced: {is_enhanced}")
        
        # Create enhanced ranked tokens with contract addresses
        ranked_tokens = []
        
        # Build contract address lookup from purchases
        contract_lookup = {}
        for p in purchases:
            token = p.token_bought
            if p.web3_analysis:
                ca = p.web3_analysis.get('contract_address', '')
                if not ca:
                    ca = p.web3_analysis.get('ca', '')
                if ca and len(ca) > 10:
                    contract_lookup[token] = ca
        
        logger.info(f"Contract lookup created for {len(contract_lookup)} tokens")
        
        if len(scores) > 0:
            logger.info("Processing token rankings with contract addresses...")
            
            # Create purchase/sell stats for missing token_stats
            purchase_stats = {}
            for purchase in purchases:
                token = purchase.token_bought
                if token not in purchase_stats:
                    purchase_stats[token] = {
                        'total_eth': 0,
                        'count': 0,
                        'wallets': set(),
                        'scores': [],
                        'tokens_amount': 0
                    }
                
                if analysis_type == 'sell':
                    purchase_stats[token]['total_eth'] += purchase.amount_received
                    if purchase.web3_analysis:
                        purchase_stats[token]['tokens_amount'] += purchase.web3_analysis.get('amount_sold', 0)
                else:
                    purchase_stats[token]['total_eth'] += purchase.eth_spent
                    purchase_stats[token]['tokens_amount'] += purchase.amount_received
                
                purchase_stats[token]['count'] += 1
                purchase_stats[token]['wallets'].add(purchase.wallet_address)
                purchase_stats[token]['scores'].append(purchase.sophistication_score or 0)
            
            for token, score_data in scores.items():
                # Get stats from purchases or defaults
                pstats = purchase_stats.get(token, {
                    'total_eth': 0, 'count': 1, 'wallets': set(['unknown']), 'scores': [0], 'tokens_amount': 0
                })
                
                # Get contract address
                contract_address = contract_lookup.get(token, '')
                
                # Enhanced token data with contract address
                if analysis_type == 'sell':
                    token_data = {
                        'total_eth_received': float(pstats['total_eth']),
                        'wallet_count': len(pstats['wallets']),
                        'total_sells': int(pstats['count']),
                        'avg_wallet_score': float(sum(pstats['scores']) / len(pstats['scores']) if pstats['scores'] else 0),
                        'total_tokens_sold': float(pstats['tokens_amount']),
                        'avg_sell_size': float(pstats['total_eth'] / pstats['count'] if pstats['count'] > 0 else 0),
                        'platforms': ['Transfer'],
                        'contract_address': contract_address,
                        'ca': contract_address,  # Alternative field name
                        'sell_pressure_score': score_data['total_score'],
                        'analysis_type': 'sell'
                    }
                else:
                    token_data = {
                        'total_eth_spent': float(pstats['total_eth']),
                        'wallet_count': len(pstats['wallets']),
                        'total_purchases': int(pstats['count']),
                        'avg_wallet_score': float(sum(pstats['scores']) / len(pstats['scores']) if pstats['scores'] else 0),
                        'platforms': ['DEX'],
                        'contract_address': contract_address,
                        'ca': contract_address,  # Alternative field name
                        'alpha_score': score_data['total_score'],
                        'analysis_type': 'buy'
                    }
                
                # Add AI enhancement data
                token_data.update({
                    'ai_enhanced': score_data.get('ai_enhanced', False),
                    'confidence': score_data.get('confidence', 0.7),
                    'ai_scores': {
                        'volume': score_data.get('volume_score', 0),
                        'quality': score_data.get('quality_score', 0),
                        'momentum': score_data.get('momentum_score', 0),
                        'liquidity': score_data.get('liquidity_score', 0),
                        'risk': score_data.get('risk_score', 0),
                        'diversity': score_data.get('diversity_score', 0)
                    },
                    'web3_data': {
                        'contract_address': contract_address,
                        'ca': contract_address,
                        'token_age_hours': score_data.get('token_age_hours'),
                        'holder_count': score_data.get('holder_count'),
                        'liquidity_eth': score_data.get('liquidity_eth'),
                        'price_change_24h': score_data.get('price_change_24h'),
                        'smart_money_percentage': score_data.get('smart_money_percentage'),
                        'whale_activity': score_data.get('whale_activity')
                    },
                    'risk_factors': score_data.get('risk_factors', {}),
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'network': getattr(self, 'current_network', 'unknown')
                })
                
                # Include all enhanced data in tuple for notifications
                # Format: (token_name, token_data, score, ai_data)
                ranked_tokens.append((token, token_data, score_data['total_score'], score_data))
                
                ca_display = contract_address[:10] + '...' if len(contract_address) > 10 else 'No CA'
                logger.debug(f"Added enhanced token: {token} (Score: {score_data['total_score']:.1f}, CA: {ca_display})")
        else:
            logger.warning("No scores available for ranking")
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Final ranked tokens with contract addresses: {len(ranked_tokens)}")
        
        # Calculate totals
        if analysis_type == 'sell':
            total_eth = sum(p.amount_received for p in purchases)
        else:
            total_eth = sum(p.eth_spent for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        logger.info(f"Enhanced result summary: {len(purchases)} transactions, {unique_tokens} tokens, {total_eth:.4f} ETH")
        
        # Log contract address stats
        tokens_with_ca = sum(1 for token, data, score, ai_data in ranked_tokens if data.get('contract_address'))
        logger.info(f"Tokens with contract addresses: {tokens_with_ca}/{len(ranked_tokens)}")
        
        return {
            'network': getattr(self, 'current_network', 'unknown'),
            'analysis_type': analysis_type,
            'total_transactions': len(purchases),
            'unique_tokens': unique_tokens,
            'total_eth_value': total_eth,
            'ranked_tokens': ranked_tokens,
            'performance_metrics': {
                **self.get_processing_stats(),
                'contract_addresses_extracted': tokens_with_ca,
                'ai_enhancement_enabled': is_enhanced
            },
            'web3_enhanced': True,
            'enhanced': is_enhanced
        }
    
    # Keep all existing helper methods unchanged...
    def _parse_timestamp(self, transfer: Dict, block_number: int = None) -> datetime:
        """Parse timestamp from transfer data or estimate from block"""
        # Try to get timestamp from transfer metadata
        if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata']:
            try:
                return datetime.fromisoformat(transfer['metadata']['blockTimestamp'].replace('Z', '+00:00'))
            except:
                pass
        
        # Fallback: estimate based on block number
        if block_number and block_number > 0:
            try:
                # Base network: ~2 second blocks, Ethereum: ~12 second blocks
                block_time = 2  # Default to Base timing
                current_block = 35093118  # Update this to current block
                seconds_ago = (current_block - block_number) * block_time
                return datetime.utcnow() - timedelta(seconds=seconds_ago)
            except:
                pass
        
        return datetime.utcnow()
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH equivalent spent - YOUR EXISTING WORKING METHOD"""
        if not outgoing_transfers:
            return 0.0
        
        # Define spending currencies with conversion rates - YOUR EXISTING LOGIC
        SPENDING_CURRENCIES = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,  # ~$2400/ETH
            'USDC': 1/2400,  # ~$2400/ETH  
            'AERO': 1/4800,  # Adjust based on current rate
        }
        
        logger.debug(f"=== ETH SPENT CALCULATION ===")
        logger.debug(f"Target TX: {target_tx}")
        logger.debug(f"Target Block: {target_block}")
        logger.debug(f"Outgoing transfers to check: {len(outgoing_transfers)}")
        
        total_eth_equivalent = 0.0
        
        # STEP 1: Try exact transaction match first (original logic)
        exact_matches = []
        for transfer in outgoing_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in SPENDING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                        total_eth_equivalent += eth_equivalent
                        exact_matches.append(f"{asset}:{amount}={eth_equivalent}ETH")
                        logger.debug(f"EXACT MATCH: {asset} {amount} = {eth_equivalent} ETH")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing {asset} value: {e}")
        
        if total_eth_equivalent > 0:
            logger.debug(f"FOUND EXACT TX MATCHES: {exact_matches}, Total: {total_eth_equivalent} ETH")
            return total_eth_equivalent
        
        # STEP 2: Block-based matching (original logic)
        block_matches = []
        matched_values = []
        for transfer in outgoing_transfers:
            if transfer.get("blockNum") == target_block:
                asset = transfer.get("asset", "")
                if asset in SPENDING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                        
                        if 0.0001 <= eth_equivalent <= 50.0:
                            matched_values.append(eth_equivalent)
                            block_matches.append(f"{asset}:{amount}={eth_equivalent}ETH")
                            logger.debug(f"BLOCK MATCH: {asset} {amount} = {eth_equivalent} ETH")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing {asset} value in block match: {e}")
        
        block_result = sum(matched_values)
        if block_result > 0:
            logger.debug(f"FOUND BLOCK MATCHES: {block_matches}, Total: {block_result} ETH")
            return block_result
        
        # STEP 3: Proximity-based matching for multi-transaction DEX trades
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            logger.debug(f"Could not parse target block: {target_block}")
            return 0.0
        
        # Look for spending within +/- 5 blocks
        proximity_matches = []
        proximity_values = []
        
        for transfer in outgoing_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                block_diff = abs(transfer_block_num - target_block_num)
                
                if block_diff <= 5:
                    asset = transfer.get("asset", "")
                    if asset in SPENDING_CURRENCIES:
                        try:
                            amount = float(transfer.get("value", "0"))
                            eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                            
                            if 0.0001 <= eth_equivalent <= 50.0:
                                proximity_values.append(eth_equivalent)
                                proximity_matches.append(f"{asset}:{amount}={eth_equivalent}ETH(±{block_diff})")
                                logger.debug(f"PROXIMITY MATCH: {asset} {amount} = {eth_equivalent} ETH (±{block_diff} blocks)")
                        except (ValueError, TypeError):
                            continue
            except (ValueError, TypeError):
                continue
        
        proximity_result = sum(proximity_values)
        if proximity_result > 0:
            logger.debug(f"FOUND PROXIMITY MATCHES: {proximity_matches}, Total: {proximity_result} ETH")
            return proximity_result
        
        logger.debug(f"NO MATCHES FOUND - returning 0.0")
        return 0.0
    
    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """ENHANCED: Calculate ETH equivalent received for SELL transactions"""
        if not incoming_transfers:
            return 0.0
        
        # EXPANDED currencies that represent ETH value received
        RECEIVING_CURRENCIES = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,  # Convert to ETH equivalent  
            'USDC': 1/2400,
            'USDC.E': 1/2400,
            'DAI': 1/2400,
            'AERO': 1/4800,  # Base network token
            'FRAX': 1/2400,
            'LUSD': 1/2400,
        }
        
        logger.debug(f"=== SELL ETH RECEIVED CALCULATION ===")
        logger.debug(f"Target TX: {target_tx}")
        logger.debug(f"Target Block: {target_block}")
        logger.debug(f"Incoming transfers to check: {len(incoming_transfers)}")
        
        total_eth_equivalent = 0.0
        
        # STEP 1: Try exact transaction match first
        exact_matches = []
        for transfer in incoming_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in RECEIVING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * RECEIVING_CURRENCIES[asset]
                        total_eth_equivalent += eth_equivalent
                        exact_matches.append(f"{asset}:{amount}={eth_equivalent}ETH")
                        logger.debug(f"SELL EXACT: {asset} {amount} = {eth_equivalent} ETH")
                    except (ValueError, TypeError):
                        continue
        
        if total_eth_equivalent > 0:
            logger.debug(f"SELL: Found exact matches: {exact_matches}, Total: {total_eth_equivalent} ETH")
            return total_eth_equivalent
        
        # STEP 2: Block-based matching - EXPANDED RANGE
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            logger.debug(f"Could not parse target block: {target_block}")
            return 0.0
        
        # STEP 3: Look for ETH/stablecoin receipts within +/- 10 blocks (expanded from 2)
        block_matches = []
        matched_values = []
        
        for transfer in incoming_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                block_diff = abs(transfer_block_num - target_block_num)
                
                # EXPANDED from ±2 to ±10 blocks for better sell detection
                if block_diff <= 10:
                    asset = transfer.get("asset", "")
                    if asset in RECEIVING_CURRENCIES:
                        try:
                            amount = float(transfer.get("value", "0"))
                            eth_equivalent = amount * RECEIVING_CURRENCIES[asset]
                            
                            # LOWERED minimum threshold for sells
                            if 0.00001 <= eth_equivalent <= 100.0:  # Was too restrictive
                                matched_values.append(eth_equivalent)
                                block_matches.append(f"{asset}:{amount}={eth_equivalent}ETH(±{block_diff})")
                                logger.debug(f"SELL BLOCK: {asset} {amount} = {eth_equivalent} ETH (±{block_diff} blocks)")
                        except (ValueError, TypeError):
                            continue
                
            except (ValueError, TypeError):
                continue
        
        block_result = sum(matched_values)
        if block_result > 0:
            logger.debug(f"SELL: Found block matches: {block_matches}, Total: {block_result} ETH")
            return block_result
        
        # STEP 4: EXPANDED proximity search - check ALL incoming transfers for any ETH-like receipts
        all_eth_received = []
        
        logger.debug(f"SELL: Checking ALL {len(incoming_transfers)} incoming transfers for ETH receipts...")
        
        for transfer in incoming_transfers:
            asset = transfer.get("asset", "")
            if asset in RECEIVING_CURRENCIES:
                try:
                    amount = float(transfer.get("value", "0"))
                    eth_equivalent = amount * RECEIVING_CURRENCIES[asset]
                    
                    # Even lower threshold - catch small sells
                    if 0.000001 <= eth_equivalent <= 50.0:
                        all_eth_received.append(eth_equivalent)
                        logger.debug(f"SELL GENERAL: {asset} {amount} = {eth_equivalent} ETH")
                except (ValueError, TypeError):
                    continue
        
        # If we found ETH receipts, use the largest one (most likely to be the sell proceeds)
        if all_eth_received:
            # Sort and take the largest value as the most likely sell proceeds
            all_eth_received.sort(reverse=True)
            largest_receipt = all_eth_received[0]
            
            # If there are multiple similar values, sum them (multi-part sale)
            similar_receipts = [x for x in all_eth_received if abs(x - largest_receipt) / largest_receipt < 0.1]
            if len(similar_receipts) > 1:
                result = sum(similar_receipts)
            else:
                result = largest_receipt
            
            logger.debug(f"SELL FALLBACK: Found {len(all_eth_received)} ETH receipts, using: {result} ETH")
            return result
        
        logger.debug(f"SELL: NO ETH RECEIVED FOUND")
        return 0.0
    
    def analyze_purchases(self, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Basic analyze purchases using pandas - fallback method"""
        if not purchases:
            return {}
        
        try:
            logger.info(f"Analyzing {len(purchases)} {analysis_type} transactions")
            
            # Convert to DataFrame with memory optimization
            data = []
            for p in purchases:
                # For sells, use amount_received as the ETH value
                # For buys, use eth_spent as the ETH value
                eth_value = p.amount_received if analysis_type == 'sell' else p.eth_spent
                
                # Extract contract address
                contract_address = ''
                if p.web3_analysis:
                    contract_address = p.web3_analysis.get('contract_address', '')
                    if not contract_address:
                        contract_address = p.web3_analysis.get('ca', '')
                
                data.append({
                    'token': p.token_bought,
                    'eth_value': eth_value,
                    'amount': p.amount_received if analysis_type == 'buy' else p.web3_analysis.get('amount_sold', 0) if p.web3_analysis else 0,
                    'wallet': p.wallet_address,
                    'score': p.sophistication_score or 0,
                    'contract': contract_address
                })
            
            df = pd.DataFrame(data)
            
            # Basic aggregation optimized for cloud functions
            token_stats = df.groupby('token').agg({
                'eth_value': ['sum', 'mean', 'count'],
                'wallet': 'nunique',
                'score': 'mean',
                'amount': 'sum'
            }).round(4)
            
            # Flatten columns
            token_stats.columns = ['total_value', 'mean_value', 'tx_count', 
                                 'unique_wallets', 'avg_score', 'total_amount']
            
            # Calculate scores based on analysis type
            scores = {}
            for token in token_stats.index:
                stats_row = token_stats.loc[token]
                
                if analysis_type == 'sell':
                    # For sells, higher values = more selling pressure
                    volume_score = min(stats_row['total_value'] * 100, 60)  # Higher weight for sells
                    diversity_score = min(stats_row['unique_wallets'] * 10, 25)
                    quality_score = min((stats_row['avg_score'] / 100) * 15, 15)
                else:
                    # For buys, standard scoring
                    volume_score = min(stats_row['total_value'] * 50, 50)
                    diversity_score = min(stats_row['unique_wallets'] * 8, 30)
                    quality_score = min((stats_row['avg_score'] / 100) * 20, 20)
                
                total_score = volume_score + diversity_score + quality_score
                
                scores[token] = {
                    'total_score': float(total_score),
                    'volume_score': float(volume_score),
                    'diversity_score': float(diversity_score),
                    'quality_score': float(quality_score)
                }
            
            logger.info(f"Basic analysis complete: {len(scores)} tokens scored")
            
            return {
                'token_stats': token_stats,
                'scores': scores,
                'analysis_type': analysis_type,
                'enhanced': False
            }
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    async def analyze_purchases_enhanced(self, purchases: List, analysis_type: str) -> Dict:
        """COMPLETE AI-Enhanced analysis with contract address extraction"""
        
        # Store current network for result creation
        if purchases and len(purchases) > 0:
            first_purchase = purchases[0]
            if hasattr(first_purchase, 'web3_analysis') and first_purchase.web3_analysis:
                self.current_network = first_purchase.web3_analysis.get('network', 'unknown')
            else:
                self.current_network = 'unknown'
        else:
            self.current_network = 'unknown'
        
        if self._enhanced_scoring_enabled and self.ai_engine:
            try:
                logger.info(f"🚀 LAUNCHING COMPLETE AI ANALYSIS WITH CONTRACT ADDRESSES...")
                
                # Run complete AI analysis
                result = await self.ai_engine.complete_ai_analysis(purchases, analysis_type)
                
                if result.get('enhanced'):
                    # Enhance result with contract addresses
                    enhanced_result = self._create_enhanced_result_with_ca(result, purchases, analysis_type)
                    
                    # Log amazing results
                    summary = enhanced_result.get('analysis_summary', result.get('analysis_summary', {}))
                    logger.info(f"🎉 COMPLETE AI SUCCESS WITH CONTRACT ADDRESSES!")
                    logger.info(f"🔍 AI Patterns Detected: {summary.get('ai_patterns_detected', 0)}")
                    logger.info(f"🎯 High Confidence Tokens: {summary.get('high_confidence_tokens', 0)}")
                    logger.info(f"⚠️ Risk Alerts: {summary.get('risk_alerts', 0)}")
                    logger.info(f"📈 Bullish Sentiment: {summary.get('sentiment_bullish', 0)}")
                    logger.info(f"🚀 Pump Signals: {summary.get('pump_signals', 0)}")
                    logger.info(f"📋 Contract Addresses: {enhanced_result['performance_metrics'].get('contract_addresses_extracted', 0)}")
                    
                    return enhanced_result
                    
            except Exception as e:
                logger.error(f"Complete AI failed: {e}")
        
        # Fallback to basic analysis with contract address extraction
        logger.info("📊 Using basic analysis fallback with contract addresses")
        basic_result = self.analyze_purchases(purchases, analysis_type)
        
        if basic_result:
            # Enhance basic result with contract addresses
            return self._create_enhanced_result_with_ca(basic_result, purchases, analysis_type)
        
        return {}
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Create empty result with contract address support"""
        return {
            'network': getattr(self, 'current_network', 'unknown'),
            'analysis_type': analysis_type,
            'total_transactions': 0,
            'unique_tokens': 0,
            'total_eth_value': 0.0,
            'ranked_tokens': [],
            'performance_metrics': {
                **self.get_processing_stats(),
                'contract_addresses_extracted': 0
            },
            'web3_enhanced': True,
            'enhanced': False,
            'error': 'No data to analyze'
        }
    
    def get_ai_insights_summary(self, result: Dict) -> str:
        """Get human-readable AI insights summary with contract address info"""
        if not result.get('enhanced'):
            return "Basic analysis - no AI insights available"
        
        summary = result.get('analysis_summary', {})
        ai_analyses = result.get('ai_analyses', {})
        
        insights = []
        insights.append(f"🤖 AI Analysis Complete!")
        insights.append(f"📊 {summary.get('total_tokens', 0)} tokens analyzed")
        insights.append(f"🔍 {summary.get('ai_patterns_detected', 0)} patterns detected")
        
        # Contract address info
        ca_count = result.get('performance_metrics', {}).get('contract_addresses_extracted', 0)
        if ca_count > 0:
            insights.append(f"📋 {ca_count} contract addresses extracted")
        
        # Whale coordination
        whale_coord = ai_analyses.get('whale_coordination', {})
        if whale_coord.get('detected'):
            insights.append(f"🐋 Whale coordination detected: {whale_coord.get('evidence_strength', 'UNKNOWN')} evidence")
        
        # Pump signals
        pump_signals = ai_analyses.get('pump_signals', {})
        if pump_signals.get('detected'):
            insights.append(f"🚀 Pump signal: {pump_signals.get('phase', 'UNKNOWN')} phase")
        
        # Technical strength
        technical = ai_analyses.get('technical_indicators', {})
        tech_strength = technical.get('strength', 0)
        if tech_strength > 0.6:
            insights.append(f"📈 Strong technical signals: {tech_strength:.0%}")
        
        # Smart money
        smart_money = ai_analyses.get('smart_money_flow', {})
        direction = smart_money.get('flow_direction', 'NEUTRAL')
        if direction != 'NEUTRAL':
            insights.append(f"🧠 Smart money: {direction}")
        
        # Risk assessment
        risk = ai_analyses.get('risk_assessment', {})
        risk_level = risk.get('risk_level', 'UNKNOWN')
        if risk_level in ['HIGH', 'EXTREME']:
            insights.append(f"⚠️ Risk level: {risk_level}")
        
        return "\n".join(insights)
    
    def _get_contract_address(self, token_purchases: List[Purchase]) -> str:
        """Extract contract address from purchases"""
        for purchase in token_purchases:
            if purchase.web3_analysis:
                contract = purchase.web3_analysis.get('contract_address', '')
                if not contract:
                    contract = purchase.web3_analysis.get('ca', '')
                if contract and len(contract) > 10:
                    return contract
        return ""
    
    def _get_network(self, token_purchases: List[Purchase]) -> str:
        """Extract network from purchases"""
        for purchase in token_purchases:
            if purchase.web3_analysis and purchase.web3_analysis.get('network'):
                return purchase.web3_analysis['network']
        return getattr(self, 'current_network', 'unknown')
    
    # Additional utility methods for debugging and monitoring
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics with contract address info"""
        return {
            'last_stored_count': self._last_stored_count,
            'enhanced_scoring_enabled': self._enhanced_scoring_enabled,
            'ai_scoring_available': AI_SCORING_AVAILABLE,
            'excluded_assets_count': len(self.EXCLUDED_ASSETS),
            'excluded_contracts_count': len(self.EXCLUDED_CONTRACTS),
            'contract_address_extraction': 'enhanced'
        }
    
    def log_token_analysis_summary(self, purchases: List[Purchase], analysis_type: str):
        """Log comprehensive analysis summary with contract address stats"""
        if not purchases:
            logger.info(f"No {analysis_type} transactions to analyze")
            return
        
        # Basic statistics
        total_eth = sum(p.eth_spent if analysis_type == 'buy' else p.amount_received for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        unique_wallets = len(set(p.wallet_address for p in purchases))
        
        # Contract address statistics
        purchases_with_ca = sum(1 for p in purchases if p.web3_analysis and p.web3_analysis.get('contract_address'))
        ca_extraction_rate = (purchases_with_ca / len(purchases)) * 100 if purchases else 0
        
        # Token distribution
        token_counts = {}
        for p in purchases:
            token_counts[p.token_bought] = token_counts.get(p.token_bought, 0) + 1
        
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"=== {analysis_type.upper()} ANALYSIS SUMMARY ===")
        logger.info(f"Total transactions: {len(purchases)}")
        logger.info(f"Total ETH volume: {total_eth:.6f}")
        logger.info(f"Unique tokens: {unique_tokens}")
        logger.info(f"Unique wallets: {unique_wallets}")
        logger.info(f"Contract addresses extracted: {purchases_with_ca}/{len(purchases)} ({ca_extraction_rate:.1f}%)")
        logger.info(f"Top tokens by transaction count:")
        for token, count in top_tokens:
            logger.info(f"  - {token}: {count} transactions")
        logger.info(f"Enhanced AI scoring: {'✓' if self._enhanced_scoring_enabled else '✗'}")
        logger.info("=" * 40)
    
    async def validate_data_quality(self, purchases: List[Purchase]) -> Dict:
        """Validate data quality including contract address extraction"""
        if not purchases:
            return {'valid': False, 'reason': 'No purchases provided'}
        
        validation_results = {
            'total_purchases': len(purchases),
            'valid_purchases': 0,
            'zero_eth_purchases': 0,
            'missing_contracts': 0,
            'missing_timestamps': 0,
            'duplicate_transactions': 0,
            'contract_address_rate': 0.0,
            'valid': True,
            'warnings': []
        }
        
        seen_hashes = set()
        purchases_with_ca = 0
        
        for purchase in purchases:
            # Check for valid ETH values
            eth_value = purchase.eth_spent if hasattr(purchase, 'eth_spent') else purchase.amount_received
            if eth_value <= 0:
                validation_results['zero_eth_purchases'] += 1
            else:
                validation_results['valid_purchases'] += 1
            
            # Check for contract addresses
            has_contract = False
            if purchase.web3_analysis:
                ca = purchase.web3_analysis.get('contract_address', '')
                if not ca:
                    ca = purchase.web3_analysis.get('ca', '')
                if ca and len(ca) > 10:
                    has_contract = True
                    purchases_with_ca += 1
            
            if not has_contract:
                validation_results['missing_contracts'] += 1
            
            # Check for timestamps
            if not purchase.timestamp:
                validation_results['missing_timestamps'] += 1
            
            # Check for duplicates
            if purchase.transaction_hash in seen_hashes:
                validation_results['duplicate_transactions'] += 1
            else:
                seen_hashes.add(purchase.transaction_hash)
        
        # Calculate contract address extraction rate
        validation_results['contract_address_rate'] = purchases_with_ca / len(purchases) if purchases else 0
        
        # Generate warnings
        if validation_results['zero_eth_purchases'] > len(purchases) * 0.5:
            validation_results['warnings'].append("High percentage of zero ETH transactions")
        
        if validation_results['missing_contracts'] > len(purchases) * 0.3:
            validation_results['warnings'].append("Many transactions missing contract addresses")
        
        if validation_results['contract_address_rate'] < 0.7:
            validation_results['warnings'].append(f"Low contract address extraction rate: {validation_results['contract_address_rate']:.1%}")
        
        if validation_results['duplicate_transactions'] > 0:
            validation_results['warnings'].append(f"{validation_results['duplicate_transactions']} duplicate transactions found")
        
        # Overall validity
        validation_results['data_quality_score'] = validation_results['valid_purchases'] / len(purchases)
        
        if validation_results['data_quality_score'] < 0.5:
            validation_results['valid'] = False
            validation_results['warnings'].append("Poor data quality - less than 50% valid transactions")
        
        logger.info(f"Data validation: {validation_results['valid_purchases']}/{len(purchases)} valid, quality score: {validation_results['data_quality_score']:.2f}")
        logger.info(f"Contract address extraction: {purchases_with_ca}/{len(purchases)} ({validation_results['contract_address_rate']:.1%})")
        
        return validation_results