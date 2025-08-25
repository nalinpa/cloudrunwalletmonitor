import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime
from api.models.data_models import WalletInfo, Purchase

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and analysis logic"""
    
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
    
    def process_transfers_to_purchases(self, wallets: List[WalletInfo], 
                                     all_transfers: Dict, network: str) -> List[Purchase]:
        """Process transfers to identify purchases"""
        purchases = []
        wallet_scores = {w.address: w.score for w in wallets}
        
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
                    
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
                    
                    if self.is_excluded_token(asset, contract_address):
                        continue
                    
                    amount = float(transfer.get("value", "0"))
                    if amount <= 0:
                        continue
                    
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    
                    # Calculate ETH spent
                    eth_spent = self._calculate_eth_spent(outgoing, tx_hash, block_num)
                    if eth_spent < 0.0005:
                        continue
                    
                    purchase = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,
                        amount_received=amount,
                        eth_spent=eth_spent,
                        wallet_address=address,
                        platform="DEX",
                        block_number=int(block_num, 16) if block_num != "0x0" else 0,
                        timestamp=datetime.now(),
                        sophistication_score=wallet_scores.get(address, 0),
                        web3_analysis={"contract_address": contract_address}
                    )
                    
                    purchases.append(purchase)
                    
                except Exception as e:
                    logger.debug(f"Error processing transfer: {e}")
                    continue
        
        return purchases
    
    def process_transfers_to_sells(self, wallets: List[WalletInfo], 
                                 all_transfers: Dict, network: str) -> List[Purchase]:
        """Process transfers to identify sells"""
        sells = []
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
                    if not asset or asset == "ETH":
                        continue
                    
                    contract_info = transfer.get("rawContract", {})
                    contract_address = contract_info.get("address", "").lower()
                    
                    if self.is_excluded_token(asset, contract_address):
                        continue
                    
                    amount_sold = float(transfer.get("value", "0"))
                    if amount_sold <= 0:
                        continue
                    
                    tx_hash = transfer.get("hash", "")
                    block_num = transfer.get("blockNum", "0x0")
                    
                    # Calculate ETH received from sell
                    eth_received = self._calculate_eth_received(incoming, tx_hash, block_num)
                    if eth_received < 0.001:
                        eth_received = min(amount_sold * 0.00001, 1.0)  # Estimate
                    
                    if eth_received < 0.001:
                        continue
                    
                    sell = Purchase(
                        transaction_hash=tx_hash,
                        token_bought=asset,  # Token that was sold
                        amount_received=eth_received,  # ETH received from sell
                        eth_spent=0,  # This is a sell, not a purchase
                        wallet_address=address,
                        platform="Transfer",
                        block_number=int(block_num, 16) if block_num != "0x0" else 0,
                        timestamp=datetime.now(),
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
        
        return sells
    
    def _calculate_eth_spent(self, outgoing_transfers: List[Dict], 
                           target_tx: str, target_block: str) -> float:
        """Calculate ETH spent in a transaction"""
        if not target_tx or not outgoing_transfers:
            return 0.0
        
        # Look for exact transaction match first
        for transfer in outgoing_transfers:
            if (transfer.get("hash") == target_tx and 
                transfer.get("asset") == "ETH"):
                try:
                    return float(transfer.get("value", "0"))
                except (ValueError, TypeError):
                    continue
        
        # Fallback to block-based matching
        matched_values = []
        for transfer in outgoing_transfers:
            if (transfer.get("blockNum") == target_block and 
                transfer.get("asset") == "ETH"):
                try:
                    eth_amount = float(transfer.get("value", "0"))
                    if 0.0001 <= eth_amount <= 50.0:
                        matched_values.append(eth_amount)
                except (ValueError, TypeError):
                    continue
        
        return sum(matched_values)
    
    def _calculate_eth_received(self, incoming_transfers: List[Dict], 
                              target_tx: str, target_block: str) -> float:
        """Calculate ETH received from a sell"""
        if not target_tx or not incoming_transfers:
            return 0.0
        
        # Look for exact transaction match first
        for transfer in incoming_transfers:
            if (transfer.get("hash") == target_tx and 
                transfer.get("asset") == "ETH"):
                try:
                    return float(transfer.get("value", "0"))
                except (ValueError, TypeError):
                    continue
        
        # Fallback to block-based matching
        matched_values = []
        for transfer in incoming_transfers:
            if (transfer.get("blockNum") == target_block and 
                transfer.get("asset") == "ETH"):
                try:
                    eth_amount = float(transfer.get("value", "0"))
                    if 0.001 <= eth_amount <= 50.0:
                        matched_values.append(eth_amount)
                except (ValueError, TypeError):
                    continue
        
        return sum(matched_values)
    
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