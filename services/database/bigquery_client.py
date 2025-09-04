import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from api.models.data_models import Transfer, TransferType
from utils.config import Config
import concurrent.futures

logger = logging.getLogger(__name__)

# Check if BigQuery is available
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("BigQuery library not available")

class BigQueryTransferService:
    """Service for managing transfer records in BigQuery - Fixed SQL syntax errors"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[bigquery.Client] = None
        self.dataset_id = config.bigquery_dataset_id
        self.table_id = config.bigquery_transfers_table
        self.full_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.table_id}"
        self._initialized = False
        # Thread executor for sync BigQuery operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize BigQuery client and setup table"""
        try:
            # Run sync initialization in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            
            self._initialized = True
            logger.info("BigQuery transfer service initialized successfully")
            
        except Exception as e:
            logger.error(f"BigQuery transfer service initialization failed: {e}")
            self._initialized = False
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization for BigQuery"""
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery library not available")
            
        # Initialize client with the project
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        logger.info(f"BigQuery client initialized for project: {self.config.bigquery_project_id}")
        
        # Ensure dataset exists
        self._sync_ensure_dataset_exists()
        
        # Ensure table exists with proper schema
        self._sync_ensure_table_exists()
    
    def _sync_ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            dataset_ref = f"{self.config.bigquery_project_id}.{self.dataset_id}"
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_ref} exists")
        except NotFound:
            logger.info(f"Creating dataset {dataset_ref} in {self.config.bigquery_location}")
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = self.config.bigquery_location
                dataset = self.client.create_dataset(dataset, timeout=30)
                logger.info(f"Created dataset {dataset.dataset_id}")
            except Exception as create_error:
                logger.error(f"Failed to create dataset: {create_error}")
                raise create_error
        except Exception as e:
            logger.error(f"Error checking dataset existence: {e}")
            raise e
    
    def _sync_ensure_table_exists(self):
        """Ensure the transfers table exists with proper schema"""
        try:
            self.client.get_table(self.full_table_id)
            logger.info(f"Table {self.full_table_id} exists")
        except NotFound:
            logger.info(f"Creating table {self.full_table_id}")
            
            # Define schema for transfers table
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
            
            table = bigquery.Table(self.full_table_id, schema=schema)
            
            # Set table properties
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            
            # Add clustering for better query performance
            table.clustering_fields = ["network", "transfer_type", "wallet_address"]
            
            table = self.client.create_table(table, timeout=30)
            logger.info(f"Created table {table.table_id}")
    
    async def get_transfer_stats(self, network: str = None, 
                               days_back: int = 30) -> Dict:
        """Get transfer statistics - FIXED SQL syntax"""
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
        """Synchronous transfer stats query - FIXED"""
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
            
            # FIXED: Proper SQL with all required fields
            query = f"""
            SELECT 
                transfer_type,
                network,
                COUNT(*) as count,
                COALESCE(SUM(cost_in_eth), 0) as total_eth,
                COUNT(DISTINCT wallet_address) as unique_wallets,
                COUNT(DISTINCT token_address) as unique_tokens,
                COALESCE(AVG(cost_in_eth), 0) as avg_eth
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            GROUP BY transfer_type, network
            ORDER BY transfer_type, network
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            
            # Log the actual query for debugging
            logger.info(f"Executing BigQuery stats query: {query}")
            logger.info(f"Query parameters: {[(p.name, p.value) for p in params]}")
            
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
                unique_wallets = int(row.unique_wallets)
                unique_tokens = int(row.unique_tokens)
                avg_eth = float(row.avg_eth or 0)
                
                stats["total_transfers"] += count
                stats["total_eth_volume"] += total_eth
                
                if transfer_type not in stats["by_type"]:
                    stats["by_type"][transfer_type] = {
                        "count": 0,
                        "total_eth": 0.0,
                        "networks": {}
                    }
                
                if network_name not in stats["by_network"]:
                    stats["by_network"][network_name] = {
                        "count": 0,
                        "total_eth": 0.0,
                        "types": {}
                    }
                
                # Update type stats
                stats["by_type"][transfer_type]["count"] += count
                stats["by_type"][transfer_type]["total_eth"] += total_eth
                stats["by_type"][transfer_type]["networks"][network_name] = {
                    "count": count,
                    "total_eth": total_eth,
                    "unique_wallets": unique_wallets,
                    "unique_tokens": unique_tokens,
                    "avg_eth": avg_eth
                }
                
                # Update network stats
                stats["by_network"][network_name]["count"] += count
                stats["by_network"][network_name]["total_eth"] += total_eth
                stats["by_network"][network_name]["types"][transfer_type] = {
                    "count": count,
                    "total_eth": total_eth,
                    "unique_wallets": unique_wallets,
                    "unique_tokens": unique_tokens,
                    "avg_eth": avg_eth
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Sync get transfer stats failed: {e}")
            return {}
    
    async def get_top_tokens_by_volume(self, transfer_type: TransferType,
                                     network: str = None,
                                     days_back: int = 7,
                                     limit: int = 50) -> List[Dict]:
        """Get top tokens by ETH volume - FIXED SQL"""
        if not self._initialized or self.client is None:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_get_top_tokens,
                transfer_type,
                network,
                days_back,
                limit
            )
            return result
            
        except Exception as e:
            logger.error(f"Error getting top tokens: {e}")
            return []
    
    def _sync_get_top_tokens(self, transfer_type: TransferType, network: str, 
                           days_back: int, limit: int) -> List[Dict]:
        """Synchronous get top tokens query - FIXED"""
        try:
            conditions = [
                "transfer_type = @transfer_type",
                "timestamp >= @cutoff_date",
                "cost_in_eth > 0"  # Only include tokens with actual ETH value
            ]
            params = [
                bigquery.ScalarQueryParameter("transfer_type", "STRING", transfer_type.value),
                bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", 
                                             datetime.utcnow() - timedelta(days=days_back)),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
            
            if network:
                conditions.append("network = @network")
                params.append(bigquery.ScalarQueryParameter("network", "STRING", network))
            
            where_clause = " AND ".join(conditions)
            
            # FIXED: Proper SQL with all required fields and COALESCE for NULL handling
            query = f"""
            SELECT 
                token_address,
                COALESCE(token_symbol, 'UNKNOWN') as token_symbol,
                COALESCE(SUM(cost_in_eth), 0) as total_eth_volume,
                COUNT(*) as transfer_count,
                COUNT(DISTINCT wallet_address) as unique_wallets_count,
                COALESCE(AVG(cost_in_eth), 0) as avg_eth_per_transfer
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            GROUP BY token_address, token_symbol
            HAVING COALESCE(SUM(cost_in_eth), 0) > 0
            ORDER BY total_eth_volume DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            
            # Log the actual query for debugging
            logger.info(f"Executing BigQuery top tokens query: {query}")
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            top_tokens = []
            for row in results:
                token_data = {
                    "token_address": row.token_address,
                    "token_symbol": row.token_symbol,
                    "total_eth_volume": float(row.total_eth_volume),
                    "transfer_count": int(row.transfer_count),
                    "unique_wallets_count": int(row.unique_wallets_count),
                    "avg_eth_per_transfer": float(row.avg_eth_per_transfer)
                }
                top_tokens.append(token_data)
            
            logger.info(f"BIGQ: Retrieved {len(top_tokens)} top tokens")
            return top_tokens
            
        except Exception as e:
            logger.error(f"Sync get top tokens failed: {e}")
            return []
    
    # Add other missing methods with proper SQL syntax
    async def store_transfer(self, transfer: Transfer) -> bool:
        """Store a single transfer record"""
        if not self._initialized or self.client is None:
            logger.error("BigQuery client not initialized")
            return False
        
        return await self.store_transfers_batch([transfer]) > 0
    
    async def store_transfers_batch(self, transfers: List[Transfer]) -> int:
        """Store multiple transfer records efficiently using streaming inserts"""
        if not self._initialized or self.client is None:
            logger.error("BigQuery client not initialized")
            return 0
        
        if not transfers:
            return 0
        
        try:
            # Convert transfers to BigQuery rows
            rows_to_insert = []
            for transfer in transfers:
                row_data = self._transfer_to_bigquery_row(transfer)
                rows_to_insert.append(row_data)
            
            # Use thread pool for sync BigQuery operations
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
            table = self.client.get_table(self.full_table_id)
            
            # Process in chunks for better performance
            chunk_size = 1000
            total_inserted = 0
            
            for i in range(0, len(rows_to_insert), chunk_size):
                chunk = rows_to_insert[i:i + chunk_size]
                
                # Use streaming insert for real-time data
                errors = self.client.insert_rows_json(
                    table, 
                    chunk,
                    ignore_unknown_values=True,
                    skip_invalid_rows=False
                )
                
                if errors:
                    logger.warning(f"Insert errors in chunk {i//chunk_size + 1}: {errors}")
                    # Count successful inserts (total - errors)
                    successful_inserts = len(chunk) - len(errors)
                    total_inserted += max(0, successful_inserts)
                else:
                    total_inserted += len(chunk)
            
            logger.info(f"Batch inserted {total_inserted} transfers from {len(rows_to_insert)} records")
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
    
    async def cleanup(self):
        """Cleanup BigQuery client"""
        if self.client:
            self.client.close()
            self._initialized = False
        
        if self.executor:
            self.executor.shutdown(wait=True)