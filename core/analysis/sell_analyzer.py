import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from .base_analyzer import BaseAnalyzer
from api.models.data_models import AnalysisResult, WalletInfo, Purchase
from services.database.data_processor import DataProcessor

class CloudSellAnalyzer(BaseAnalyzer):
    """Cloud-optimized Sell analyzer with BigQuery transfer storage"""
    
    def __init__(self, network: str):
        super().__init__(network)
        self.data_processor = DataProcessor()
    
    async def initialize(self):
        """Initialize services and set BigQuery transfer service in data processor"""
        await super().initialize()
        # Connect the data processor to the BigQuery transfer service
        self.data_processor.set_transfer_service(self.bigquery_transfer_service)
    
    def _get_analysis_type(self) -> str:
        return "sell"
    
    async def _process_data(self, wallets: List[WalletInfo], 
                          all_transfers: Dict) -> AnalysisResult:
        """Process transfers to identify sell transactions and store all transfers to BigQuery"""
        
        # Convert transfers to sells - this will now also store all transfers to BigQuery
        sells = await self.data_processor.process_transfers_to_sells(
            wallets, all_transfers, self.network
        )
        
        if not sells:
            return self._empty_result()
        
        # Analyze sells using pandas
        analysis_results = self.data_processor.analyze_purchases(sells, "sell")
        
        # Update stats with BigQuery transfer storage info
        self.stats["transfers_stored"] = getattr(self.data_processor, '_last_stored_count', 0)
        
        # Create result
        return self._create_result(analysis_results, sells)
    
    def _create_result(self, analysis_results: Dict, 
                      sells: List[Purchase]) -> AnalysisResult:
        """Create analysis result for sells"""
        
        if not analysis_results:
            return self._empty_result()
        
        token_stats = analysis_results.get('token_stats')
        scores = analysis_results.get('scores', {})
        
        # Create ranked tokens
        ranked_tokens = []
        contract_lookup = {s.token_bought: s.web3_analysis.get('contract_address', '') 
                          for s in sells if s.web3_analysis}
        
        if token_stats is not None:
            for token in scores.keys():
                if token in token_stats.index:
                    stats_data = token_stats.loc[token]
                    score_data = scores[token]
                    
                    token_data = {
                        'total_eth_value': float(stats_data['total_value']),
                        'wallet_count': int(stats_data['unique_wallets']),
                        'total_sells': int(stats_data['tx_count']),
                        'avg_wallet_score': float(stats_data['avg_score']),
                        'methods': ['Transfer'],
                        'contract_address': contract_lookup.get(token, ''),
                        'sell_pressure_score': score_data['total_score'],
                        'is_base_native': self.network == 'base'
                    }
                    
                    ranked_tokens.append((token, token_data, score_data['total_score']))
        
        # Sort by score
        ranked_tokens.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate totals
        total_eth = sum(s.amount_received for s in sells)
        unique_tokens = len(set(s.token_bought for s in sells))
        
        return AnalysisResult(
            network=self.network,
            analysis_type="sell",
            total_transactions=len(sells),
            unique_tokens=unique_tokens,
            total_eth_value=total_eth,
            ranked_tokens=ranked_tokens,
            performance_metrics=self.stats,
            web3_enhanced=True
        )