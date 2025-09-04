from dataclasses import dataclass, asdict
from enum import Enum
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class WalletInfo:
    address: str
    score: float
    network: str
    
class TransferType(Enum):
    BUY = "buy"
    SELL = "sell"
    
@dataclass
class Transfer:
    """Data model for individual transfer records"""
    wallet_address: str
    token_address: str
    transfer_type: TransferType  # buy or sell
    timestamp: datetime
    cost_in_eth: float
    transaction_hash: str
    block_number: int
    token_amount: float
    token_symbol: Optional[str] = None
    network: str = "ethereum"
    platform: Optional[str] = None
    created_at: datetime = None
    wallet_sophistication_score: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        # Convert enum to string
        data['transfer_type'] = self.transfer_type.value
        # Keep datetime objects as is for MongoDB
        data['timestamp'] = self.timestamp
        data['created_at'] = self.created_at
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transfer':
        """Create Transfer from dictionary"""
        # Convert string back to enum
        if isinstance(data.get('transfer_type'), str):
            data['transfer_type'] = TransferType(data['transfer_type'])
        
        # Convert ISO strings to datetime if necessary
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].rstrip('Z'))
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].rstrip('Z'))
        
        return cls(**data)