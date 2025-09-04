import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict
from utils.config import Config
from services.database.database_client import DatabaseService
from services.database.bigquery_client import BigQueryTransferService
from services.blockchain.alchemy_client import AlchemyService
from api.models.data_models import AnalysisResult, WalletInfo

logger = logging.getLogger(__name__)

class BaseAnalyzer(ABC):
    """Base class for all analyzers with BigQuery transfer storage capability"""
    
    def __init__(self, network: str):
        self.network = network
        self.config = Config()
        self.db_service = DatabaseService(self.config)  # MongoDB for wallet data
        self.bigquery_transfer_service = BigQueryTransferService(self.config)  # BigQuery for transfer data
        self.alchemy_service = AlchemyService(self.config)
        
        # Performance tracking
        self.stats = {
            "analysis_time": 0.0,
            "wallets_processed": 0,
            "transfers_processed": 0,
            "transfers_stored": 0,
            "memory_used_mb": 0.0
        }
    
    async def initialize(self):
        """Initialize all services"""
        # Initialize MongoDB service for wallet data
        await self.db_service.initialize()
        
        # Initialize BigQuery service for transfer data
        await self.bigquery_transfer_service.initialize()
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    async def analyze(self, num_wallets: int, days_back: float) -> AnalysisResult:
        """Main analysis method with BigQuery transfer storage"""
        start_time = time.time()
        logger.info(f"Starting {self.__class__.__name__} for {self.network}")
        
        try:
            # Get wallets from MongoDB
            wallets = await self.db_service.get_top_wallets(self.network, num_wallets)
            if not wallets:
                logger.warning(f"No wallets found for {self.network}")
                return self._empty_result()
            
            self.stats["wallets_processed"] = len(wallets)
            
            # Get block range
            start_block, end_block = await self.alchemy_service.get_block_range(self.network, days_back)
            if start_block == 0:
                logger.error("Failed to get block range")
                return self._empty_result()
            
            # Get transfers
            wallet_addresses = [w.address for w in wallets]
            all_transfers = await self.alchemy_service.get_transfers_batch(
                self.network, wallet_addresses, start_block, end_block
            )
            
            # Count total transfers for stats
            total_transfers = 0
            for wallet_transfers in all_transfers.values():
                total_transfers += len(wallet_transfers.get('incoming', []))
                total_transfers += len(wallet_transfers.get('outgoing', []))
            
            self.stats["transfers_processed"] = total_transfers
            logger.info(f"Processing {total_transfers} transfers from {len(wallets)} wallets")
            
            # Process data (implemented by subclasses) - this will now store transfers to BigQuery
            result = await self._process_data(wallets, all_transfers)
            
            self.stats["analysis_time"] = time.time() - start_time
            logger.info(f"Analysis complete in {self.stats['analysis_time']:.2f}s")
            logger.info(f"Stored {self.stats.get('transfers_stored', 0)} transfer records to BigQuery")
            
            return result
            
        except Exception as e:
            self.stats["analysis_time"] = time.time() - start_time
            logger.error(f"Analysis failed: {e}")
            return self._empty_result()
    
    @abstractmethod
    async def _process_data(self, wallets: List[WalletInfo], 
                          all_transfers: Dict) -> AnalysisResult:
        """Process the transfer data - implemented by subclasses"""
        pass
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty result"""
        return AnalysisResult(
            network=self.network,
            analysis_type=self._get_analysis_type(),
            total_transactions=0,
            unique_tokens=0,
            total_eth_value=0.0,
            ranked_tokens=[],
            performance_metrics=self.stats,
            web3_enhanced=True
        )
    
    @abstractmethod
    def _get_analysis_type(self) -> str:
        """Get the analysis type - implemented by subclasses"""
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.db_service.cleanup()
        await self.bigquery_transfer_service.cleanup()
    
    async def get_transfer_stats(self, days_back: int = 7) -> Dict:
        """Get transfer statistics for this network from BigQuery"""
        try:
            return await self.bigquery_transfer_service.get_transfer_stats(
                network=self.network, 
                days_back=days_back
            )
        except Exception as e:
            logger.error(f"Failed to get transfer stats: {e}")
            return {}
    
    async def get_top_tokens_by_volume(self, transfer_type, days_back: int = 7, limit: int = 20) -> List[Dict]:
        """Get top tokens by volume for this network from BigQuery"""
        try:
            from api.models.data_models import TransferType
            transfer_type_enum = TransferType.BUY if transfer_type == 'buy' else TransferType.SELL
            
            return await self.bigquery_transfer_service.get_top_tokens_by_volume(
                transfer_type=transfer_type_enum,
                network=self.network,
                days_back=days_back,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get top tokens: {e}")
            return []