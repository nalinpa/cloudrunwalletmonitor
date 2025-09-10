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
    """Enhanced data processor with AI-powered alpha scoring and optional storage"""
    
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
    
    async def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                           all_transfers: Dict, network: str,
                                           store_data: bool = False) -> List[Purchase]:
        """Process transfers to identify BUY transactions with optional storage"""
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
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
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
                            web3_analysis={"contract_address": contract_address}
                        )
                        purchases.append(purchase)
                        
                except Exception as e:
                    logger.debug(f"Error processing incoming transfer: {e}")
                    continue
            
            # Process OUTGOING transfers for completeness (conditionally stored)
            for transfer in outgoing:
                try:
                    asset = transfer.get("asset")
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
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
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str,
                                       store_data: bool = False) -> List[Purchase]:
        """Process transfers to identify SELL transactions with optional storage"""
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
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
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
                                "amount_sold": amount_sold,
                                "is_sell": True
                            }
                        )
                        sells.append(sell)
                        
                        # DEBUG LOG to see what's being detected
                        logger.info(f"SELL DETECTED: {asset} amount_sold={amount_sold} eth_received={eth_received:.6f} tx={tx_hash[:10]}...")
                        
                except Exception as e:
                    logger.debug(f"Error processing sell transfer: {e}")
                    continue
            
            # Also process incoming transfers for completeness (conditionally stored)
            for transfer in incoming:
                try:
                    asset = transfer.get("asset")
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
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
        
        # Additional debug info for sells
        if sells:
            total_eth_received = sum(s.amount_received for s in sells)
            unique_tokens = set(s.token_bought for s in sells)
            logger.info(f"SELL SUMMARY: {total_eth_received:.6f} ETH received from selling {len(unique_tokens)} unique tokens")
            
            # Log top sells
            sells_sorted = sorted(sells, key=lambda x: x.amount_received, reverse=True)
            logger.info("Top 5 sells by ETH received:")
            for i, sell in enumerate(sells_sorted[:5]):
                logger.info(f"  {i+1}. {sell.token_bought}: {sell.amount_received:.6f} ETH (amount_sold: {sell.web3_analysis.get('amount_sold', 0)})")
        else:
            logger.warning("No sells detected - check if _calculate_eth_received is working properly")
        
        return sells
    
    # Keep all your existing helper methods unchanged...
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
                
                data.append({
                    'token': p.token_bought,
                    'eth_value': eth_value,
                    'amount': p.amount_received if analysis_type == 'buy' else p.web3_analysis.get('amount_sold', 0),
                    'wallet': p.wallet_address,
                    'score': p.sophistication_score or 0,
                    'contract': p.web3_analysis.get('contract_address', '') if p.web3_analysis else ''
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
        """COMPLETE AI-Enhanced analysis with all features"""
        
        if self._enhanced_scoring_enabled and self.ai_engine:
            try:
                logger.info(f"🚀 LAUNCHING COMPLETE AI ANALYSIS...")
                
                # Run complete AI analysis
                result = await self.ai_engine.complete_ai_analysis(purchases, analysis_type)
                
                if result.get('enhanced'):
                    # Log amazing results
                    summary = result.get('analysis_summary', {})
                    logger.info(f"🎉 COMPLETE AI SUCCESS!")
                    logger.info(f"🔍 AI Patterns Detected: {summary.get('ai_patterns_detected', 0)}")
                    logger.info(f"🎯 High Confidence Tokens: {summary.get('high_confidence_tokens', 0)}")
                    logger.info(f"⚠️ Risk Alerts: {summary.get('risk_alerts', 0)}")
                    logger.info(f"📈 Bullish Sentiment: {summary.get('sentiment_bullish', 0)}")
                    logger.info(f"🚀 Pump Signals: {summary.get('pump_signals', 0)}")
                    
                    return result
                    
            except Exception as e:
                logger.error(f"Complete AI failed: {e}")
        
        # Fallback to your existing basic analysis
        logger.info("📊 Using basic analysis fallback")
        return self.analyze_purchases(purchases, analysis_type)
    
    def get_ai_insights_summary(self, result: Dict) -> str:
        """Get human-readable AI insights summary"""
        if not result.get('enhanced'):
            return "Basic analysis - no AI insights available"
        
        summary = result.get('analysis_summary', {})
        ai_analyses = result.get('ai_analyses', {})
        
        insights = []
        insights.append(f"🤖 AI Analysis Complete!")
        insights.append(f"📊 {summary.get('total_tokens', 0)} tokens analyzed")
        insights.append(f"🔍 {summary.get('ai_patterns_detected', 0)} patterns detected")
        
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
            if purchase.web3_analysis and purchase.web3_analysis.get('contract_address'):
                contract = purchase.web3_analysis['contract_address']
                if contract and len(contract) > 10:
                    return contract
        return ""
    
    def _get_network(self, token_purchases: List[Purchase]) -> str:
        """Extract network from purchases"""
        # Try to get from transfer records or use default
        # You can enhance this based on your data structure
        return "ethereum"  # Default fallback
    
    # Additional utility methods for debugging and monitoring
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'last_stored_count': self._last_stored_count,
            'enhanced_scoring_enabled': self._enhanced_scoring_enabled,
            'ai_scoring_available': AI_SCORING_AVAILABLE,
            'excluded_assets_count': len(self.EXCLUDED_ASSETS),
            'excluded_contracts_count': len(self.EXCLUDED_CONTRACTS)
        }
    
    def log_token_analysis_summary(self, purchases: List[Purchase], analysis_type: str):
        """Log comprehensive analysis summary"""
        if not purchases:
            logger.info(f"No {analysis_type} transactions to analyze")
            return
        
        # Basic statistics
        total_eth = sum(p.eth_spent if analysis_type == 'buy' else p.amount_received for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        unique_wallets = len(set(p.wallet_address for p in purchases))
        
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
        logger.info(f"Top tokens by transaction count:")
        for token, count in top_tokens:
            logger.info(f"  - {token}: {count} transactions")
        logger.info(f"Enhanced AI scoring: {'✓' if self._enhanced_scoring_enabled else '✗'}")
        logger.info("=" * 40)
    
    async def validate_data_quality(self, purchases: List[Purchase]) -> Dict:
        """Validate data quality and return metrics"""
        if not purchases:
            return {'valid': False, 'reason': 'No purchases provided'}
        
        validation_results = {
            'total_purchases': len(purchases),
            'valid_purchases': 0,
            'zero_eth_purchases': 0,
            'missing_contracts': 0,
            'missing_timestamps': 0,
            'duplicate_transactions': 0,
            'valid': True,
            'warnings': []
        }
        
        seen_hashes = set()
        
        for purchase in purchases:
            # Check for valid ETH values
            eth_value = purchase.eth_spent if hasattr(purchase, 'eth_spent') else purchase.amount_received
            if eth_value <= 0:
                validation_results['zero_eth_purchases'] += 1
            else:
                validation_results['valid_purchases'] += 1
            
            # Check for contract addresses
            if not purchase.web3_analysis or not purchase.web3_analysis.get('contract_address'):
                validation_results['missing_contracts'] += 1
            
            # Check for timestamps
            if not purchase.timestamp:
                validation_results['missing_timestamps'] += 1
            
            # Check for duplicates
            if purchase.transaction_hash in seen_hashes:
                validation_results['duplicate_transactions'] += 1
            else:
                seen_hashes.add(purchase.transaction_hash)
        
        # Generate warnings
        if validation_results['zero_eth_purchases'] > len(purchases) * 0.5:
            validation_results['warnings'].append("High percentage of zero ETH transactions")
        
        if validation_results['missing_contracts'] > len(purchases) * 0.3:
            validation_results['warnings'].append("Many transactions missing contract addresses")
        
        if validation_results['duplicate_transactions'] > 0:
            validation_results['warnings'].append(f"{validation_results['duplicate_transactions']} duplicate transactions found")
        
        # Overall validity
        validation_results['data_quality_score'] = validation_results['valid_purchases'] / len(purchases)
        
        if validation_results['data_quality_score'] < 0.5:
            validation_results['valid'] = False
            validation_results['warnings'].append("Poor data quality - less than 50% valid transactions")
        
        logger.info(f"Data validation: {validation_results['valid_purchases']}/{len(purchases)} valid, quality score: {validation_results['data_quality_score']:.2f}")
        
        return validation_results