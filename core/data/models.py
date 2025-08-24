from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class Purchase:
    transaction_hash: str
    token_bought: str
    amount_received: float
    eth_spent: float
    wallet_address: str
    platform: str
    block_number: int
    timestamp: datetime
    sophistication_score: Optional[float] = None
    web3_analysis: Optional[Dict] = None

@dataclass
class AnalysisResult:
    network: str
    analysis_type: str
    total_transactions: int
    unique_tokens: int
    total_eth_value: float
    ranked_tokens: List[tuple]
    performance_metrics: Dict[str, Any]
    web3_enhanced: bool = False
