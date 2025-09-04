import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from services.database.bigquery_client import BigQueryTransferService

logger = logging.getLogger(__name__)

class DataProcessor:
    """Enhanced data processor that stores all transfers to BigQuery"""
    
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
                                           all_transfers: Dict, network: str) -> List[Purchase]:
        """Process transfers to identify purchases and store all transfers to BigQuery"""
        purchases = []
        all_transfer_records = []  # Collect all transfers for batch storage
        wallet_scores = {w.address: w.score for w in wallets}
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            incoming = transfers.get('incoming', [])
            outgoing = transfers.get('outgoing', [])
            
            # Debug log to see what currencies are available
            all_assets = set()
            for transfer in incoming + outgoing:
                all_assets.add(transfer.get("asset"))
            logger.debug(f"Wallet {address} has transfers in: {all_assets}")
            
            # Process incoming transfers (potential buys)
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
                    
                    # Calculate ETH cost using multiple currencies
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # Debug log to see what we're finding
                    if eth_spent > 0:
                        logger.debug(f"Found ETH cost {eth_spent} for token {asset} in tx {tx_hash}")
                    else:
                        # Check what spending currencies are available
                        spending_breakdown = self._get_spending_breakdown(outgoing, tx_hash, block_num)
                        if spending_breakdown:
                            logger.warning(f"No ETH cost calculated but found spending: {spending_breakdown} for tx {tx_hash}")
                    
                    # Create transfer record for ALL incoming ERC20 transfers
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
                    
                    # Only create purchases for non-excluded tokens with sufficient ETH spent
                    if not self.is_excluded_token(asset, contract_address) and eth_spent >= 0.0005:
                        purchase = Purchase(
                            transaction_hash=tx_hash,
                            token_bought=asset,
                            amount_received=amount,
                            eth_spent=eth_spent,
                            wallet_address=address,
                            platform="DEX",
                            block_number=block_number,
                            timestamp=transfer_record.timestamp,
                            sophistication_score=wallet_scores.get(address, 0),
                            web3_analysis={"contract_address": contract_address}
                        )
                        purchases.append(purchase)
                        
                except Exception as e:
                    logger.debug(f"Error processing incoming transfer: {e}")
                    continue
            
            # Process outgoing transfers (potential sells or token movements)
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
                    
                    # Calculate ETH received for this transaction (for sells)
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    
                    # Create transfer record for ALL outgoing ERC20 transfers
                    transfer_record = Transfer(
                        wallet_address=address,
                        token_address=contract_address,
                        transfer_type=TransferType.SELL,
                        timestamp=self._parse_timestamp(transfer, block_number),
                        cost_in_eth=eth_received if eth_received > 0 else 0.0,  # Don't use arbitrary estimates
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
        
        # Store all transfers to BigQuery
        if self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"Stored {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"Failed to store transfer records to BigQuery: {e}")
                self._last_stored_count = 0
        
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                       all_transfers: Dict, network: str) -> List[Purchase]:
        """Process transfers to identify sells and store all transfers to BigQuery"""
        sells = []
        all_transfer_records = []  # Collect all transfers for batch storage
        wallet_scores = {w.address: w.score for w in wallets}
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            outgoing = transfers.get('outgoing', [])
            incoming = transfers.get('incoming', [])
            
            # Process outgoing ERC20 transfers as potential sells
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
                    
                    # Create transfer record for ALL outgoing ERC20 transfers
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
                    
                    # Only create sells for non-excluded tokens with sufficient value
                    if not self.is_excluded_token(asset, contract_address) and eth_received >= 0.001:
                        sell = Purchase(
                            transaction_hash=tx_hash,
                            token_bought=asset,  # Token that was sold
                            amount_received=eth_received,  # ETH received from sell
                            eth_spent=0,  # This is a sell, not a purchase
                            wallet_address=address,
                            platform="Transfer",
                            block_number=block_number,
                            timestamp=transfer_record.timestamp,
                            sophistication_score=wallet_scores.get(address, 0),
                            web3_analysis={
                                "contract_address": contract_address,
                                "amount_sold": amount_sold,
                                "is_sell": True
                            }
                        )
                        sells.append(sell)
                        
                except Exception as e:
                    logger.debug(f"Error processing sell transfer: {e}")
                    continue
            
            # Also process incoming transfers for completeness
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
                    
                    # Create transfer record for ALL incoming ERC20 transfers
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
        
        # Store all transfers to BigQuery
        if self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"Stored {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"Failed to store transfer records to BigQuery: {e}")
                self._last_stored_count = 0
        
        return sells
    
    def _parse_timestamp(self, transfer: Dict, block_number: int = None) -> datetime:
        """Parse timestamp from transfer data or estimate from block"""
        # Try to get timestamp from transfer metadata
        if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata']:
            try:
                return datetime.fromisoformat(transfer['metadata']['blockTimestamp'].replace('Z', '+00:00'))
            except:
                pass
        
        # Fallback: estimate based on block number (very rough)
        if block_number and block_number > 0:
            try:
                # Rough estimate: ~12 second blocks for ETH, ~2 seconds for Base
                # Adjust based on your network
                block_time = 2 if 'base' in str(transfer.get('network', '')).lower() else 12
                current_block = 35093118  # Update this to current block
                seconds_ago = (current_block - block_number) * block_time
                return datetime.utcnow() - timedelta(seconds=seconds_ago)
            except:
                pass
        
        # Final fallback
        return datetime.utcnow()
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH equivalent spent in a transaction - checks multiple currencies"""
        if not target_tx or not outgoing_transfers:
            return 0.0
        
        # Define spending currencies with their ETH conversion rates
        SPENDING_CURRENCIES = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,  # Update based on current ETH price
            'USDC': 1/2400,  # Update based on current ETH price  
            'AERO': 1/4800,  # Adjust based on current AERO/ETH rate
        }
        
        total_eth_equivalent = 0.0
        
        # Look for exact transaction match first
        for transfer in outgoing_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in SPENDING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                        total_eth_equivalent += eth_equivalent
                        # Log what we found for debugging
                        if eth_equivalent > 0:
                            logger.debug(f"Found {asset} spend: {amount} = {eth_equivalent} ETH in tx {target_tx}")
                    except (ValueError, TypeError):
                        continue
        
        # If we found spending in the exact transaction, return it
        if total_eth_equivalent > 0:
            return total_eth_equivalent
        
        # Fallback to block-based matching
        matched_values = []
        for transfer in outgoing_transfers:
            if transfer.get("blockNum") == target_block:
                asset = transfer.get("asset", "")
                if asset in SPENDING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                        
                        # Reasonable spending range check
                        if 0.0001 <= eth_equivalent <= 50.0:
                            matched_values.append(eth_equivalent)
                            logger.debug(f"Block match - {asset}: {amount} = {eth_equivalent} ETH")
                    except (ValueError, TypeError):
                        continue
        
        return sum(matched_values)

    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """Calculate ETH equivalent received from a sell - checks multiple currencies"""
        if not target_tx or not incoming_transfers:
            return 0.0
        
        # Define receiving currencies with their ETH conversion rates
        RECEIVING_CURRENCIES = {
            'ETH': 1.0,
            'WETH': 1.0,
            'USDT': 1/2400,  # Update based on current ETH price
            'USDC': 1/2400,  # Update based on current ETH price
            'AERO': 1/4800,  # Adjust based on current AERO/ETH rate
        }
        
        total_eth_equivalent = 0.0
        
        # Look for exact transaction match first
        for transfer in incoming_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in RECEIVING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * RECEIVING_CURRENCIES[asset]
                        total_eth_equivalent += eth_equivalent
                        # Log what we found for debugging
                        if eth_equivalent > 0:
                            logger.debug(f"Received {asset}: {amount} = {eth_equivalent} ETH in tx {target_tx}")
                    except (ValueError, TypeError):
                        continue
        
        # If we found receiving in the exact transaction, return it
        if total_eth_equivalent > 0:
            return total_eth_equivalent
        
        # Fallback to block-based matching
        matched_values = []
        for transfer in incoming_transfers:
            if transfer.get("blockNum") == target_block:
                asset = transfer.get("asset", "")
                if asset in RECEIVING_CURRENCIES:
                    try:
                        amount = float(transfer.get("value", "0"))
                        eth_equivalent = amount * RECEIVING_CURRENCIES[asset]
                        
                        # Reasonable receiving range check
                        if 0.001 <= eth_equivalent <= 50.0:
                            matched_values.append(eth_equivalent)
                            logger.debug(f"Block match received - {asset}: {amount} = {eth_equivalent} ETH")
                    except (ValueError, TypeError):
                        continue
        
        return sum(matched_values)

    def _get_spending_breakdown(self, outgoing_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> Dict:
        """Get detailed breakdown of what was spent in a transaction"""
        if not target_tx or not outgoing_transfers:
            return {}
        
        spending_breakdown = {}
        
        # Check exact transaction first
        for transfer in outgoing_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in ['ETH', 'WETH', 'USDT', 'USDC', 'AERO']:
                    try:
                        amount = float(transfer.get("value", "0"))
                        if amount > 0:
                            spending_breakdown[asset] = spending_breakdown.get(asset, 0) + amount
                    except (ValueError, TypeError):
                        continue
        
        # If no exact match, try block matching
        if not spending_breakdown:
            for transfer in outgoing_transfers:
                if transfer.get("blockNum") == target_block:
                    asset = transfer.get("asset", "")
                    if asset in ['ETH', 'WETH', 'USDT', 'USDC', 'AERO']:
                        try:
                            amount = float(transfer.get("value", "0"))
                            if amount > 0:
                                spending_breakdown[asset] = spending_breakdown.get(asset, 0) + amount
                        except (ValueError, TypeError):
                            continue
        
        return spending_breakdown
    
    def analyze_purchases(self, purchases: List[Purchase], analysis_type: str) -> Dict:
        """Analyze purchases using pandas - optimized for cloud functions"""
        if not purchases:
            return {}
        
        try:
            # Convert to DataFrame with memory optimization
            data = []
            for p in purchases:
                data.append({
                    'token': p.token_bought,
                    'eth_value': p.eth_spent if analysis_type == 'buy' else p.amount_received,
                    'amount': p.amount_received,
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
            
            # Calculate scores (simplified for cloud function)
            scores = {}
            for token in token_stats.index:
                stats_row = token_stats.loc[token]
                
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
            
            return {
                'token_stats': token_stats,
                'scores': scores,
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}