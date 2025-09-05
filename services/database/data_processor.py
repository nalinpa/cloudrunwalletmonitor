import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from api.models.data_models import WalletInfo, Purchase, Transfer, TransferType
from services.database.bigquery_client import BigQueryTransferService

logger = logging.getLogger(__name__)

class DataProcessor:
    """Enhanced data processor with extensive debug logging"""
    
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
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        # Debug: Log first wallet's transfer structure
        if wallets and all_transfers:
            first_wallet = wallets[0]
            first_address = first_wallet.address
            first_transfers = all_transfers.get(first_address, {"incoming": [], "outgoing": []})
            
            logger.info(f"=== DEBUG: FIRST WALLET TRANSFER STRUCTURE ===")
            logger.info(f"Wallet: {first_address}")
            logger.info(f"Incoming transfers: {len(first_transfers.get('incoming', []))}")
            logger.info(f"Outgoing transfers: {len(first_transfers.get('outgoing', []))}")
            
            if first_transfers.get('outgoing'):
                sample_out = first_transfers['outgoing'][0]
                logger.info(f"Sample outgoing transfer: {sample_out}")
                logger.info(f"  - Asset: {sample_out.get('asset')}")
                logger.info(f"  - Value: {sample_out.get('value')}")
                logger.info(f"  - Hash: {sample_out.get('hash')}")
                logger.info(f"  - BlockNum: {sample_out.get('blockNum')}")
            
            if first_transfers.get('incoming'):
                sample_in = first_transfers['incoming'][0]
                logger.info(f"Sample incoming transfer: {sample_in}")
                logger.info(f"  - Asset: {sample_in.get('asset')}")
                logger.info(f"  - Value: {sample_in.get('value')}")
                logger.info(f"  - Hash: {sample_in.get('hash')}")
                logger.info(f"  - BlockNum: {sample_in.get('blockNum')}")
            logger.info(f"=== END DEBUG STRUCTURE ===")
        
        for wallet in wallets:
            address = wallet.address
            transfers = all_transfers.get(address, {"incoming": [], "outgoing": []})
            
            incoming = transfers.get('incoming', [])
            outgoing = transfers.get('outgoing', [])
            
            # Debug log currencies for first few wallets
            if len(all_transfer_records) < 50:  # Only for first batch
                all_assets = set()
                for transfer in incoming + outgoing:
                    all_assets.add(transfer.get("asset"))
                logger.debug(f"Wallet {address[-8:]} currencies: {all_assets}")
            
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
                    
                    # Debug log for first few transfers
                    if len(all_transfer_records) < 10:
                        logger.info(f"=== PURCHASE DEBUG ===")
                        logger.info(f"Token: {asset}, Amount: {amount}")
                        logger.info(f"TX: {tx_hash}")
                        logger.info(f"ETH spent calculated: {eth_spent}")
                        logger.info(f"Excluded: {self.is_excluded_token(asset, contract_address)}")
                        spending_breakdown = self._get_spending_breakdown(outgoing, tx_hash, block_num)
                        logger.info(f"Spending breakdown: {spending_breakdown}")
                        logger.info(f"=== END PURCHASE DEBUG ===")
                    
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
                    
                    # LOWERED THRESHOLD FOR TESTING
                    if not self.is_excluded_token(asset, contract_address) and eth_spent >= 0.00001:
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
                        logger.debug(f"Added purchase: {asset} for {eth_spent} ETH")
                        
                except Exception as e:
                    logger.debug(f"Error processing incoming transfer: {e}")
                    continue
            
            # Process outgoing transfers
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
                        cost_in_eth=eth_received if eth_received > 0 else 0.0,
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
        
        logger.info(f"Created {len(purchases)} purchases from {len(all_transfer_records)} total transfers")
        return purchases
    
    async def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                   all_transfers: Dict, network: str) -> List[Purchase]:
        """Process transfers to identify SELL transactions - Complete function with improved detection"""
        sells = []
        all_transfer_records = []
        wallet_scores = {w.address: w.score for w in wallets}
        
        logger.info(f"Processing transfers for SELL analysis on {network}")
        
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
                    
                    # Calculate ETH received from sell - USE YOUR EXISTING METHOD (but with the improved version)
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
                    
                    # Only create sell if not excluded and has meaningful ETH value received
                    if not self.is_excluded_token(asset, contract_address) and eth_received >= 0.000001:
                        sell = Purchase(
                            transaction_hash=tx_hash,
                            token_bought=asset,  # Token that was sold
                            amount_received=eth_received,  # ETH received from sell
                            eth_spent=0,  # Not applicable for sells
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
                        
                        # ADD DEBUG LOG to see what's being detected
                        logger.info(f"SELL DETECTED: {asset} amount_sold={amount_sold} eth_received={eth_received:.6f} tx={tx_hash[:10]}...")
                        
                except Exception as e:
                    logger.debug(f"Error processing sell transfer: {e}")
                    continue
            
            # Also process incoming transfers for completeness (same as your buy logic)
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
                    
                    # Calculate ETH cost for this transaction - USE YOUR EXISTING METHOD
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
        
        # Store all transfers to BigQuery - USE YOUR EXISTING STORAGE LOGIC
        if self.bigquery_transfer_service and all_transfer_records:
            try:
                stored_count = await self.bigquery_transfer_service.store_transfers_batch(all_transfer_records)
                self._last_stored_count = stored_count
                logger.info(f"Stored {stored_count} transfer records to BigQuery")
            except Exception as e:
                logger.error(f"Failed to store transfer records to BigQuery: {e}")
                self._last_stored_count = 0
        
        logger.info(f"SELL: Created {len(sells)} sells from {len(all_transfer_records)} total transfers")
        
        # Additional debug info
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
                # Base network: ~2 second blocks
                block_time = 2
                current_block = 35093118  # Update this to current block
                seconds_ago = (current_block - block_number) * block_time
                return datetime.utcnow() - timedelta(seconds=seconds_ago)
            except:
                pass
        
        return datetime.utcnow()
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                       target_tx: str, target_block: str) -> float:
        """Calculate ETH equivalent spent - handles multi-transaction DEX trades"""
        if not outgoing_transfers:
            return 0.0
        
        # Define spending currencies with conversion rates
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
        
        # STEP 3: NEW - Time-based matching for multi-transaction DEX trades
        # Parse target block number for proximity matching
        try:
            target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
        except (ValueError, TypeError):
            logger.debug(f"Could not parse target block: {target_block}")
            return 0.0
        
        # Look for spending within +/- 5 blocks (about 10 seconds on Base)
        proximity_matches = []
        proximity_values = []
        
        for transfer in outgoing_transfers:
            transfer_block = transfer.get("blockNum", "0x0")
            try:
                transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
                block_diff = abs(transfer_block_num - target_block_num)
                
                # Within 5 blocks
                if block_diff <= 5:
                    asset = transfer.get("asset", "")
                    if asset in SPENDING_CURRENCIES:
                        try:
                            amount = float(transfer.get("value", "0"))
                            eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                            
                            # Reasonable spending range
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
        
        # STEP 4: NEW - Wallet activity correlation 
        # If no proximity matches, look for any recent spending activity
        recent_spending = []
        recent_values = []
        
        for transfer in outgoing_transfers:
            asset = transfer.get("asset", "")
            if asset in SPENDING_CURRENCIES:
                try:
                    amount = float(transfer.get("value", "0"))
                    eth_equivalent = amount * SPENDING_CURRENCIES[asset]
                    
                    # Any reasonable spending amount
                    if 0.001 <= eth_equivalent <= 10.0:  # Higher threshold for correlation
                        recent_values.append(eth_equivalent)
                        recent_spending.append(f"{asset}:{amount}={eth_equivalent}ETH")
                        logger.debug(f"RECENT SPENDING: {asset} {amount} = {eth_equivalent} ETH")
                except (ValueError, TypeError):
                    continue
        
        # If we found multiple small spends, take the median to avoid outliers
        if len(recent_values) > 0:
            if len(recent_values) == 1:
                correlation_result = recent_values[0]
            else:
                # Take median of recent spending as estimate
                correlation_result = sorted(recent_values)[len(recent_values) // 2]
            
            logger.debug(f"CORRELATION ESTIMATE: {recent_spending}, Using: {correlation_result} ETH")
            return correlation_result
        
        logger.debug(f"NO MATCHES FOUND - returning 0.0")
        return 0.0

    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                          target_tx: str, target_block: str) -> float:
        """Calculate ETH equivalent received for SELL transactions"""
        if not incoming_transfers:
            return 0.0
        
        # EXPANDED currencies that represent ETH value received (more than your current list)
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
        
        # STEP 3: Look for ETH/stablecoin receipts within +/- 10 blocks (was too restrictive)
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
        # This catches multi-transaction DEX trades that your current logic misses
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

    def _get_receiving_breakdown(self, incoming_transfers: List[Dict], 
                               target_tx: str, target_block: str) -> Dict:
        """Get detailed breakdown of what was received in a transaction"""
        if not target_tx or not incoming_transfers:
            return {}
        
        receiving_breakdown = {}
        
        # Check exact transaction first
        for transfer in incoming_transfers:
            if transfer.get("hash") == target_tx:
                asset = transfer.get("asset", "")
                if asset in ['ETH', 'WETH', 'USDT', 'USDC', 'AERO']:
                    try:
                        amount = float(transfer.get("value", "0"))
                        if amount > 0:
                            receiving_breakdown[asset] = receiving_breakdown.get(asset, 0) + amount
                    except (ValueError, TypeError):
                        continue
        
        return receiving_breakdown
    
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