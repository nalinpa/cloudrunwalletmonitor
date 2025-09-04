import logging
import asyncio
from typing import List, Dict, Optional
from google.cloud import bigquery
from api.models.data_models import WalletInfo
from utils.config import Config
import concurrent.futures

logger = logging.getLogger(__name__)

class DatabaseService:
    """BigQuery-only database service - replaces MongoDB completely"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[bigquery.Client] = None
        self.dataset_id = config.bigquery_dataset_id
        self.table_id = "smart_wallets"
        self.full_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.table_id}"
        self._initialized = False
        
        # Thread executor for sync BigQuery operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize BigQuery client"""
        try:
            # Run sync initialization in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            
            self._initialized = True
            logger.info("BigQuery database service initialized successfully")
            
        except Exception as e:
            logger.error(f"BigQuery database service initialization failed: {e}")
            self._initialized = False
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization for BigQuery"""
        # Initialize client
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        logger.info(f"BigQuery client initialized for project: {self.config.bigquery_project_id}")
        
        # Test connection
        try:
            query = f"SELECT COUNT(*) as count FROM `{self.full_table_id}` LIMIT 1"
            query_job = self.client.query(query)
            list(query_job.result())
            logger.info(f"Successfully connected to table: {self.full_table_id}")
        except Exception as e:
            logger.error(f"Failed to connect to table {self.full_table_id}: {e}")
            raise
    
    async def get_top_wallets(self, network: str, limit: int = 50) -> List[WalletInfo]:
        """Get top wallets from BigQuery smart_wallets table"""
        if not self._initialized or not self.client:
            logger.error("BigQuery client not initialized")
            return []
        
        try:
            loop = asyncio.get_event_loop()
            wallets = await loop.run_in_executor(
                self.executor,
                self._sync_get_top_wallets,
                network,
                limit
            )
            return wallets
            
        except Exception as e:
            logger.error(f"Error getting top wallets: {e}")
            return []
    
    def _sync_get_top_wallets(self, network: str, limit: int) -> List[WalletInfo]:
        """Synchronously get top wallets from BigQuery"""
        try:
            # Query your table: address (STRING), score (INTEGER), is_active (BOOLEAN)
            query = f"""
            SELECT 
                address,
                score
            FROM `{self.full_table_id}`
            WHERE address IS NOT NULL
              AND address LIKE '0x%'
              AND LENGTH(address) = 42
              AND score IS NOT NULL
              AND score > 0
              AND COALESCE(is_active, TRUE) = TRUE
            ORDER BY score DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("limit", "INT64", limit)
                ]
            )
            
            logger.info(f"Fetching top {limit} wallets for network: {network}")
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            wallets = []
            for row in results:
                wallet = WalletInfo(
                    address=row.address,
                    score=float(row.score) if row.score else 0.0,
                    network=network  # Assign network from request
                )
                wallets.append(wallet)

            logger.info(f"DB: Retrieved {len(wallets)} wallets from BigQuery")

            if len(wallets) == 0:
                logger.warning("No wallets found in smart_wallets table")
                logger.info("Check that your table has:")
                logger.info("- Valid hex addresses (0x...)")
                logger.info("- Scores > 0")
                logger.info("- is_active = TRUE")
            
            return wallets
            
        except Exception as e:
            logger.error(f"Sync get top wallets failed: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test BigQuery connection"""
        try:
            if not self.client:
                return False
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_test_connection
            )
            return result
            
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False
    
    def _sync_test_connection(self) -> bool:
        """Synchronous connection test"""
        try:
            # Simple query test
            query = f"SELECT COUNT(*) as count FROM `{self.full_table_id}` LIMIT 1"
            query_job = self.client.query(query)
            results = list(query_job.result())
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Connection test query failed: {e}")
            return False
    
    async def get_wallet_stats(self) -> Dict:
        """Get wallet statistics from BigQuery"""
        if not self._initialized or not self.client:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.executor,
                self._sync_get_wallet_stats
            )
            return stats
            
        except Exception as e:
            logger.error(f"Error getting wallet stats: {e}")
            return {}
    
    def _sync_get_wallet_stats(self) -> Dict:
        """Get wallet statistics"""
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_wallets,
                COUNT(CASE WHEN COALESCE(is_active, TRUE) = TRUE THEN 1 END) as active_wallets,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                COUNT(CASE WHEN score >= 50 THEN 1 END) as high_score_wallets,
                COUNT(CASE WHEN score >= 20 THEN 1 END) as medium_score_wallets
            FROM `{self.full_table_id}`
            WHERE address IS NOT NULL 
              AND address LIKE '0x%'
              AND LENGTH(address) = 42
              AND score IS NOT NULL
            """
            
            query_job = self.client.query(query)
            results = list(query_job.result())
            
            if results:
                row = results[0]
                return {
                    "total_wallets": int(row.total_wallets or 0),
                    "active_wallets": int(row.active_wallets or 0),
                    "avg_score": float(row.avg_score or 0),
                    "min_score": int(row.min_score or 0),
                    "max_score": int(row.max_score or 0),
                    "high_score_wallets": int(row.high_score_wallets or 0),
                    "medium_score_wallets": int(row.medium_score_wallets or 0)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Wallet stats query failed: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup BigQuery client"""
        if self.client:
            try:
                self.client.close()
                logger.info("BigQuery database connection closed")
            except Exception as e:
                logger.error(f"Error closing BigQuery connection: {e}")
            finally:
                self.client = None
                self._initialized = False
        
        if self.executor:
            self.executor.shutdown(wait=True)