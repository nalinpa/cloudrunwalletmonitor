import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime
from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from services.database.transfer_service import BigQueryTransferService

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
                    
                    # Calculate ETH cost for this transaction
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    
                    # Debug logging for ETH calculation
                    if eth_spent == 0.0 and asset not in self.EXCLUDED_ASSETS:
                        logger.debug(f"Zero ETH cost for {asset} transfer - tx: {tx_hash[:10] if tx_hash else 'none'}, "
                                   f"block: {block_num}, outgoing_count: {len(outgoing)}")
                        # Log a sample of outgoing transfers for debugging
                        if len(outgoing) > 0:
                            sample_outgoing = outgoing[:2]  # First 2 transfers
                            for i, out_transfer in enumerate(sample_outgoing):
                                out_asset = out_transfer.get("asset", "unknown")
                                out_value = out_transfer.get("value", "0")
                                out_hash = out_transfer.get("hash", "none")
                                logger.debug(f"  Outgoing {i}: {out_asset}={out_value}, hash={out_hash[:10] if out_hash != 'none' else 'none'}")
                    
                    elif eth_spent > 0:
                        logger.debug(f"Found ETH cost for {asset}: {eth_spent}")
                    
                    
                    # Create transfer record for ALL incoming ERC20 transfers
                    transfer_record = Transfer(
                        wallet_address=address,
                        token_address=contract_address,
                        transfer_type=TransferType.BUY,
                        timestamp=self._parse_timestamp(transfer),
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
                        timestamp=self._parse_timestamp(transfer),
                        cost_in_eth=eth_received if eth_received > 0 else min(amount * 0.00001, 1.0),  # Estimate if no ETH received
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
                    if eth_received < 0.001:
                        eth_received = min(amount_sold * 0.00001, 1.0)  # Estimate
                    
                    # Create transfer record for ALL outgoing ERC20 transfers
                    transfer_record = Transfer(
                        wallet_address=address,
                        token_address=contract_address,
                        transfer_type=TransferType.SELL,
                        timestamp=self._parse_timestamp(transfer),
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
                        timestamp=self._parse_timestamp(transfer),
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
    
    def _parse_timestamp(self, transfer: Dict) -> datetime:
        """Parse timestamp from transfer data"""
        # Most Alchemy responses don't include timestamp in the transfer object
        # For now, use current time - in production you might want to fetch block timestamps
        return datetime.utcnow()
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH spent in a transaction - IMPROVED VERSION"""
        if not target_tx or not outgoing_transfers:
            return 0.0
        
        # Strategy 1: Look for exact transaction match first
        exact_matches = []
        for transfer in outgoing_transfers:
            if (transfer.get("hash") == target_tx and 
                transfer.get("asset") == "ETH"):
                try:
                    eth_value = float(transfer.get("value", "0"))
                    if eth_value > 0:
                        exact_matches.append(eth_value)
                except (ValueError, TypeError):
                    continue
        
        if exact_matches:
            total_exact = sum(exact_matches)
            logger.debug(f"Found exact ETH match for tx {target_tx[:10]}: {total_exact}")
            return total_exact
        
        # Strategy 2: Look for WETH transfers (common in DEX swaps)
        weth_addresses = {
            '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH on Ethereum
            '0x4200000000000000000000000000000000000006'   # WETH on Base
        }
        
        weth_matches = []
        for transfer in outgoing_transfers:
            contract_address = transfer.get("rawContract", {}).get("address", "").lower()
            if (transfer.get("blockNum") == target_block and 
                contract_address in [addr.lower() for addr in weth_addresses]):
                try:
                    weth_value = float(transfer.get("value", "0"))
                    if weth_value > 0:
                        weth_matches.append(weth_value)
                        logger.debug(f"Found WETH transfer: {weth_value}")
                except (ValueError, TypeError):
                    continue
        
        if weth_matches:
            return sum(weth_matches)
        
        # Strategy 3: Improved block-based matching with better filtering
        block_matches = []
        eth_transfers_in_block = []
        
        for transfer in outgoing_transfers:
            if transfer.get("blockNum") == target_block:
                if transfer.get("asset") == "ETH":
                    eth_transfers_in_block.append(transfer)
        
        # If only one ETH transfer in block, likely related
        if len(eth_transfers_in_block) == 1:
            try:
                eth_value = float(eth_transfers_in_block[0].get("value", "0"))
                if 0.000001 <= eth_value <= 100.0:  # Broader range
                    logger.debug(f"Single ETH transfer in block: {eth_value}")
                    return eth_value
            except (ValueError, TypeError):
                pass
        
        # If multiple ETH transfers, use heuristics
        elif len(eth_transfers_in_block) > 1:
            for transfer in eth_transfers_in_block:
                try:
                    eth_amount = float(transfer.get("value", "0"))
                    # More permissive range for DEX transactions
                    if 0.000001 <= eth_amount <= 100.0:
                        block_matches.append(eth_amount)
                except (ValueError, TypeError):
                    continue
            
            if block_matches:
                # Take the largest ETH transfer as most likely to be the swap
                largest_transfer = max(block_matches)
                logger.debug(f"Multiple ETH transfers, using largest: {largest_transfer}")
                return largest_transfer
        
        # Strategy 4: Fallback - estimate based on typical DEX patterns
        # For very small amounts or dust, provide minimal estimate
        try:
            token_amount = float(target_tx) if target_tx.replace('.', '').isdigit() else 0
            if token_amount > 1000:  # Large token amounts might be low-value tokens
                estimated_eth = min(token_amount * 0.000001, 0.01)  # Very conservative estimate
                if estimated_eth > 0:
                    logger.debug(f"Using fallback estimation: {estimated_eth}")
                    return estimated_eth
        except:
            pass
        
        logger.debug(f"No ETH cost found for tx {target_tx[:10] if target_tx else 'none'}, block {target_block}")
        return 0.0
    
    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """Calculate ETH received from a sell - IMPROVED VERSION"""
        if not target_tx or not incoming_transfers:
            return 0.0
        
        # Strategy 1: Look for exact transaction match first
        for transfer in incoming_transfers:
            if (transfer.get("hash") == target_tx and 
                transfer.get("asset") == "ETH"):
                try:
                    eth_value = float(transfer.get("value", "0"))
                    if eth_value > 0:
                        logger.debug(f"Found exact ETH received match: {eth_value}")
                        return eth_value
                except (ValueError, TypeError):
                    continue
        
        # Strategy 2: Look for WETH received (DEX sells often involve WETH)
        weth_addresses = {
            '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH on Ethereum  
            '0x4200000000000000000000000000000000000006'   # WETH on Base
        }
        
        for transfer in incoming_transfers:
            contract_address = transfer.get("rawContract", {}).get("address", "").lower()
            if (transfer.get("blockNum") == target_block and 
                contract_address in [addr.lower() for addr in weth_addresses]):
                try:
                    weth_value = float(transfer.get("value", "0"))
                    if weth_value > 0:
                        logger.debug(f"Found WETH received: {weth_value}")
                        return weth_value
                except (ValueError, TypeError):
                    continue
        
        # Strategy 3: Block-based matching with improved logic
        eth_transfers_in_block = []
        for transfer in incoming_transfers:
            if (transfer.get("blockNum") == target_block and 
                transfer.get("asset") == "ETH"):
                eth_transfers_in_block.append(transfer)
        
        # If single ETH transfer in block, likely the sell proceeds
        if len(eth_transfers_in_block) == 1:
            try:
                eth_value = float(eth_transfers_in_block[0].get("value", "0"))
                if 0.000001 <= eth_value <= 100.0:  # Broader range
                    logger.debug(f"Single ETH received in block: {eth_value}")
                    return eth_value
            except (ValueError, TypeError):
                pass
        
        # If multiple ETH transfers, take the largest (most likely the sell)
        elif len(eth_transfers_in_block) > 1:
            valid_amounts = []
            for transfer in eth_transfers_in_block:
                try:
                    eth_amount = float(transfer.get("value", "0"))
                    if 0.000001 <= eth_amount <= 100.0:
                        valid_amounts.append(eth_amount)
                except (ValueError, TypeError):
                    continue
            
            if valid_amounts:
                largest = max(valid_amounts)
                logger.debug(f"Multiple ETH received, using largest: {largest}")
                return largest
        
        logger.debug(f"No ETH received found for tx {target_tx[:10] if target_tx else 'none'}, block {target_block}")
        return 0.0
    
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