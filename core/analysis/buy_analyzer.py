import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from .base_analyzer import BaseAnalyzer
from api.models.data_models import AnalysisResult, WalletInfo, Purchase
from services.database.data_processor import DataProcessor

class CloudBuyAnalyzer(BaseAnalyzer):
    """Cloud-optimized Buy analyzer"""
    
    def __init__(self, network: str):
        super().__init__(network)
        self.data_processor = DataProcessor()
    
    def _get_analysis_type(self) -> str:
        return "buy"
    
    async def _process_data(self, wallets: List[WalletInfo], 
                          all_transfers: Dict) -> AnalysisResult:
        """Process transfers to identify buy transactions"""
        
        # Convert transfers to purchases
        purchases = self.data_processor.process_transfers_to_purchases(
            wallets, all_transfers, self.network
        )
        
        if not purchases:
            return self._empty_result()
        
        # Analyze purchases using pandas
        analysis_results = self.data_processor.analyze_purchases(purchases, "buy")
        
        # Create result
        return self._create_result(analysis_results, purchases)
    
    def _create_result(self, analysis_results: Dict, 
                      purchases: List[Purchase]) -> AnalysisResult:
        """Create analysis result"""
        
        if not analysis_results:
            return self._empty_result()
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        
        # Create ranked tokens
        ranked_tokens = []
        contract_lookup = {p.token_bought: p.web3_analysis.get('contract_address', '') 
                          for p in purchases if p.web3_analysis}
        
        if token_stats is not None:
            for token in scores.keys():
                if token in token_stats.index:
                    stats_data = token_stats.loc[token]
                    score_data = scores[token]
                    
                    token_data = {
                        'total_eth_spent': float(stats_data['total_value']),
                        'wallet_count': int(stats_data['unique_wallets']),
                        'total_purchases': int(stats_data['tx_count']),
                        'avg_wallet_score': float(stats_data['avg_score']),
                        'platforms': ['DEX'],
                        'contract_address': contract_lookup.get(token, ''),
                        'alpha_score': score_data['total_score'],
                        'is_base_native': self.network == 'base'
                    }
                    
                    ranked_tokens.append((token, token_data, score_data['total_score']))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate totals
        total_eth = sum(p.eth_spent for p in purchases)
        unique_tokens = len(set(p.token_bought for p in purchases))
        
        return AnalysisResult(
            network=self.network,
            analysis_type="buy",
            total_transactions=len(purchases),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=self.stats,
            web3_enhanced=True
        )
