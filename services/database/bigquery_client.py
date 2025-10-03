# services/database/bigquery_client.py - COMPLETE WITH WALLET RETRIEVAL
import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from api.models.data_models import Transfer, TransferType, WalletInfo
from utils.config import Config
import concurrent.futures

logger = logging.getLogger(__name__)

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("BigQuery library not available")

class BigQueryTransferService:
    """Complete BigQuery service for wallets AND transfers - No MongoDB needed"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[bigquery.Client] = None
        self.dataset_id = config.bigquery_dataset_id
        
        # Tables
        self.transfers_table = config.bigquery_transfers_table
        self.wallets_table = "smart_wallets"  # Wallet storage
        
        # Full table IDs
        self.transfers_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.transfers_table}"
        self.wallets_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.wallets_table}"
        
        self._initialized = False
        
        # Thread executor for sync BigQuery operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize BigQuery client and setup tables"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            
            self._initialized = True
            logger.info("BigQuery service initialized (wallets + transfers)")
            
        except Exception as e:
            logger.error(f"BigQuery service initialization failed: {e}")
            self._initialized = False
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization for BigQuery"""
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery library not available")
            
        # Initialize client
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        logger.info(f"BigQuery client initialized for project: {self.config.bigquery_project_id}")
        
        # Ensure dataset exists
        self._sync_ensure_dataset_exists()
        
        # Ensure tables exist
        self._sync_ensure_wallets_table_exists()
        self._sync_ensure_transfers_table_exists()
    
    def _sync_ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            dataset_ref = f"{self.config.bigquery_project_id}.{self.dataset_id}"
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_ref} exists")
        except NotFound:
            logger.info(f"Creating dataset {dataset_ref}")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.config.bigquery_location
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset.dataset_id}")
        except Exception as e:
            logger.error(f"Error with dataset: {e}")
            raise
    
    def _sync_ensure_wallets_table_exists(self):
        """Check if smart_wallets table exists - don't create if schema differs"""
        try:
            table = self.client.get_table(self.wallets_table_id)
            logger.info(f"Wallets table exists: {self.wallets_table_id}")
            
            # Log existing schema for reference
            schema_fields = [field.name for field in table.schema]
            logger.info(f"Existing schema: {', '.join(schema_fields)}")
            
            # Verify required fields exist
            required_fields = ['address', 'score']
            missing = [f for f in required_fields if f not in schema_fields]
            
            if missing:
                logger.error(f"Missing required fields in wallets table: {missing}")
                raise ValueError(f"Wallets table missing required fields: {missing}")
            
            logger.info("✓ Wallets table has required fields (address, score)")
            
        except NotFound:
            logger.warning(f"Wallets table not found: {self.wallets_table_id}")
            logger.info(f"Creating new wallets table with standard schema...")
            
            schema = [
                bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("score", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("network", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("is_active", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("last_activity", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("total_transactions", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("total_volume_eth", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            
            table = bigquery.Table(self.wallets_table_id, schema=schema)
            table.clustering_fields = ["score"]
            
            table = self.client.create_table(table, timeout=30)
            logger.info(f"Created wallets table: {table.table_id}")
    
    def _sync_ensure_transfers_table_exists(self):
        """Ensure the transfers table exists"""
        try:
            self.client.get_table(self.transfers_table_id)
            logger.info(f"Transfers table exists: {self.transfers_table_id}")
        except NotFound:
            logger.info(f"Creating transfers table: {self.transfers_table_id}")
            
            schema = [
                bigquery.SchemaField("wallet_address", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("token_address", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("transfer_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("cost_in_eth", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("transaction_hash", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("block_number", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("token_amount", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("token_symbol", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("network", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("platform", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("wallet_sophistication_score", "FLOAT64", mode="NULLABLE"),
            ]
            
            table = bigquery.Table(self.transfers_table_id, schema=schema)
            
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            
            table.clustering_fields = ["network", "transfer_type", "wallet_address"]
            
            table = self.client.create_table(table, timeout=30)
            logger.info(f"Created transfers table: {table.table_id}")
    
    # ========================================================================
    # WALLET OPERATIONS - Replaces MongoDB
    # ========================================================================
    
    async def get_top_wallets(self, network: str, limit: int = 50) -> List[WalletInfo]:
        """Get top wallets from BigQuery - Replaces MongoDB"""
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
            logger.error(f"Error getting top wallets from BigQuery: {e}")
            return []
    
    def _sync_get_top_wallets(self, network: str, limit: int) -> List[WalletInfo]:
        """Synchronously get top wallets from BigQuery - FIXED for existing schema"""
        try:
            # UPDATED QUERY - Works with your existing schema (address, score, id, created_at, last_updated, is_active)
            query = f"""
            SELECT 
                address,
                score
            FROM `{self.wallets_table_id}`
            WHERE address IS NOT NULL
              AND address LIKE '0x%'
              AND LENGTH(address) = 42
              AND score IS NOT NULL
              AND score > 0
              AND COALESCE(is_active, TRUE) = TRUE
            ORDER BY score DESC
            LIMIT @row_limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("row_limit", "INT64", limit)
                ]
            )
            
            logger.info(f"BigQuery: Fetching top {limit} wallets (no network filter)")
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            wallets = []
            for row in results:
                wallet = WalletInfo(
                    address=row.address,
                    score=float(row.score) if row.score else 0.0,
                    network=network  # Use requested network since table doesn't have network column
                )
                wallets.append(wallet)

            logger.info(f"✓ BigQuery: Retrieved {len(wallets)} wallets")
            
            return wallets
            
        except Exception as e:
            logger.error(f"BigQuery wallet query failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def store_wallet(self, address: str, score: float, network: str) -> bool:
        """Store or update a wallet in BigQuery"""
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self._sync_store_wallet,
                address,
                score,
                network
            )
            return success
        except Exception as e:
            logger.error(f"Failed to store wallet: {e}")
            return False
    
    def _sync_store_wallet(self, address: str, score: float, network: str) -> bool:
        """Synchronously store wallet"""
        try:
            now = datetime.utcnow().isoformat()
            
            row_data = {
                "address": address,
                "score": float(score),
                "network": network,
                "is_active": True,
                "created_at": now,
                "updated_at": now
            }
            
            table = self.client.get_table(self.wallets_table_id)
            errors = self.client.insert_rows_json(table, [row_data])
            
            if not errors:
                logger.info(f"Stored wallet: {address} (score: {score})")
                return True
            else:
                logger.error(f"Error storing wallet: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Wallet storage failed: {e}")
            return False
    
    async def get_wallet_stats(self, network: str = None) -> Dict:
        """Get wallet statistics from BigQuery"""
        if not self._initialized or not self.client:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.executor,
                self._sync_get_wallet_stats,
                network
            )
            return stats
            
        except Exception as e:
            logger.error(f"Error getting wallet stats: {e}")
            return {}
    
    def _sync_get_wallet_stats(self, network: str = None) -> Dict:
        """Get wallet statistics"""
        try:
            where_clause = f"WHERE network = '{network}'" if network else ""
            
            query = f"""
            SELECT 
                COUNT(*) as total_wallets,
                COUNT(CASE WHEN COALESCE(is_active, TRUE) = TRUE THEN 1 END) as active_wallets,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                COUNT(CASE WHEN score >= 50 THEN 1 END) as high_score_wallets,
                COUNT(CASE WHEN score >= 20 THEN 1 END) as medium_score_wallets
            FROM `{self.wallets_table_id}`
            {where_clause}
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
                    "medium_score_wallets": int(row.medium_score_wallets or 0),
                    "network": network
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Wallet stats query failed: {e}")
            return {}
    
    # ========================================================================
    # TRANSFER OPERATIONS
    # ========================================================================
    
    async def store_transfers_batch(self, transfers: List[Transfer]) -> int:
        """Store multiple transfer records"""
        if not self._initialized or self.client is None:
            logger.error("BigQuery client not initialized")
            return 0
        
        if not transfers:
            return 0
        
        try:
            rows_to_insert = []
            for transfer in transfers:
                row_data = self._transfer_to_bigquery_row(transfer)
                rows_to_insert.append(row_data)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._sync_insert_rows, 
                rows_to_insert
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error storing transfers batch: {e}")
            return 0
    
    def _sync_insert_rows(self, rows_to_insert: List[Dict]) -> int:
        """Synchronous batch insert"""
        try:
            table = self.client.get_table(self.transfers_table_id)
            
            chunk_size = 1000
            total_inserted = 0
            
            for i in range(0, len(rows_to_insert), chunk_size):
                chunk = rows_to_insert[i:i + chunk_size]
                
                errors = self.client.insert_rows_json(
                    table, 
                    chunk,
                    ignore_unknown_values=True,
                    skip_invalid_rows=False
                )
                
                if errors:
                    logger.warning(f"Insert errors in chunk {i//chunk_size + 1}: {errors}")
                    successful_inserts = len(chunk) - len(errors)
                    total_inserted += max(0, successful_inserts)
                else:
                    total_inserted += len(chunk)
            
            logger.info(f"✓ BigQuery: Inserted {total_inserted}/{len(rows_to_insert)} transfers")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Sync insert rows failed: {e}")
            return 0
    
    def _transfer_to_bigquery_row(self, transfer: Transfer) -> Dict[str, Any]:
        """Convert Transfer object to BigQuery row format"""
        return {
            "wallet_address": transfer.wallet_address,
            "token_address": transfer.token_address,
            "transfer_type": transfer.transfer_type.value,
            "timestamp": transfer.timestamp.isoformat() if transfer.timestamp else datetime.utcnow().isoformat(),
            "cost_in_eth": float(transfer.cost_in_eth),
            "transaction_hash": transfer.transaction_hash,
            "block_number": int(transfer.block_number),
            "token_amount": float(transfer.token_amount),
            "token_symbol": transfer.token_symbol,
            "network": transfer.network,
            "platform": transfer.platform,
            "created_at": transfer.created_at.isoformat() if transfer.created_at else datetime.utcnow().isoformat(),
            "wallet_sophistication_score": float(transfer.wallet_sophistication_score) if transfer.wallet_sophistication_score else None
        }
    
    async def get_transfer_stats(self, network: str = None, days_back: int = 30) -> Dict:
        """Get transfer statistics"""
        if not self._initialized or self.client is None:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_get_transfer_stats,
                network,
                days_back
            )
            return result
            
        except Exception as e:
            logger.error(f"Error getting transfer stats: {e}")
            return {}
    
    def _sync_get_transfer_stats(self, network: str, days_back: int) -> Dict:
        """Synchronous transfer stats query"""
        try:
            conditions = ["timestamp >= @cutoff_date"]
            params = [
                bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", 
                                             datetime.utcnow() - timedelta(days=days_back))
            ]
            
            if network:
                conditions.append("network = @network")
                params.append(bigquery.ScalarQueryParameter("network", "STRING", network))
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                transfer_type,
                network,
                COUNT(*) as count,
                COALESCE(SUM(cost_in_eth), 0) as total_eth,
                COUNT(DISTINCT wallet_address) as unique_wallets,
                COUNT(DISTINCT token_address) as unique_tokens,
                COALESCE(AVG(cost_in_eth), 0) as avg_eth
            FROM `{self.transfers_table_id}`
            WHERE {where_clause}
            GROUP BY transfer_type, network
            ORDER BY transfer_type, network
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            stats = {
                "total_transfers": 0,
                "total_eth_volume": 0.0,
                "by_type": {},
                "by_network": {},
                "days_analyzed": days_back
            }
            
            for row in results:
                transfer_type = row.transfer_type
                network_name = row.network
                count = int(row.count)
                total_eth = float(row.total_eth or 0)
                
                stats["total_transfers"] += count
                stats["total_eth_volume"] += total_eth
                
                if transfer_type not in stats["by_type"]:
                    stats["by_type"][transfer_type] = {"count": 0, "total_eth": 0.0, "networks": {}}
                
                if network_name not in stats["by_network"]:
                    stats["by_network"][network_name] = {"count": 0, "total_eth": 0.0, "types": {}}
                
                stats["by_type"][transfer_type]["count"] += count
                stats["by_type"][transfer_type]["total_eth"] += total_eth
                stats["by_type"][transfer_type]["networks"][network_name] = {
                    "count": count,
                    "total_eth": total_eth,
                    "unique_wallets": int(row.unique_wallets),
                    "unique_tokens": int(row.unique_tokens),
                    "avg_eth": float(row.avg_eth)
                }
                
                stats["by_network"][network_name]["count"] += count
                stats["by_network"][network_name]["total_eth"] += total_eth
            
            return stats
            
        except Exception as e:
            logger.error(f"Transfer stats query failed: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup BigQuery client"""
        if self.client:
            try:
                self.client.close()
                logger.info("BigQuery connection closed")
            except Exception as e:
                logger.error(f"Error closing BigQuery: {e}")
            finally:
                self.client = None
                self._initialized = False
        
        if self.executor:
            self.executor.shutdown(wait=True)