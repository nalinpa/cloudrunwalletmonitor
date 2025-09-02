import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from api.models.data_models import Transfer, TransferType
from utils.config import Config

logger = logging.getLogger(__name__)

class BigQueryTransferService:
    """Service for managing transfer records in BigQuery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[bigquery.Client] = None
        self.dataset_id = config.bigquery_dataset_id
        self.table_id = config.bigquery_transfers_table
        self.full_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.table_id}"
        self._initialized = False
    
    async def initialize(self):
        """Initialize BigQuery client and setup table"""
        try:
            self.client = bigquery.Client(project=self.config.bigquery_project_id)
            
            # Ensure dataset exists
            await self._ensure_dataset_exists()
            
            # Ensure table exists with proper schema
            await self._ensure_table_exists()
            
            self._initialized = True
            logger.info("BigQuery transfer service initialized successfully")
            
        except Exception as e:
            logger.error(f"BigQuery transfer service initialization failed: {e}")
            self._initialized = False
            raise
    
    async def _ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            self.client.get_dataset(self.dataset_id)
            logger.info(f"Dataset {self.dataset_id} exists")
        except NotFound:
            logger.info(f"Creating dataset {self.dataset_id}")
            dataset = bigquery.Dataset(f"{self.config.bigquery_project_id}.{self.dataset_id}")
            dataset.location = self.config.bigquery_location
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset.dataset_id}")
    
    async def _ensure_table_exists(self):
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
                bigquery.SchemaField("transfer_type", "STRING", mode="REQUIRED"),  # 'buy' or 'sell'
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
    
    async def store_transfer(self, transfer: Transfer) -> bool:
        """Store a single transfer record"""
        if not self._initialized or self.client is None:
            logger.error("BigQuery client not initialized")
            return False
        
        try:
            transfer_data = self._transfer_to_bigquery_row(transfer)
            
            # Use MERGE statement to handle duplicates
            query = f"""
            MERGE `{self.full_table_id}` T
            USING (SELECT * FROM UNNEST([{self._format_row_for_merge(transfer_data)}])) S
            ON T.transaction_hash = S.transaction_hash
            WHEN NOT MATCHED THEN
              INSERT (wallet_address, token_address, transfer_type, timestamp, cost_in_eth,
                      transaction_hash, block_number, token_amount, token_symbol, network,
                      platform, created_at, wallet_sophistication_score)
              VALUES (S.wallet_address, S.token_address, S.transfer_type, S.timestamp, S.cost_in_eth,
                      S.transaction_hash, S.block_number, S.token_amount, S.token_symbol, S.network,
                      S.platform, S.created_at, S.wallet_sophistication_score)
            """
            
            job = self.client.query(query)
            job.result(timeout=30)
            
            logger.debug(f"Stored transfer: {transfer.wallet_address} -> {transfer.token_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing transfer: {e}")
            return False
    
    async def store_transfers_batch(self, transfers: List[Transfer]) -> int:
        """Store multiple transfer records efficiently using batch insert"""
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
            
            # Use batch insert with ignore_unknown_values and create_disposition
            table = self.client.get_table(self.full_table_id)
            
            # Configure job to ignore duplicates based on transaction_hash
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                ignore_unknown_values=True,
                max_bad_records=10  # Allow some bad records but still complete the job
            )
            
            # For large batches, use streaming inserts
            if len(rows_to_insert) > 1000:
                # Process in chunks for better performance
                chunk_size = 1000
                total_inserted = 0
                
                for i in range(0, len(rows_to_insert), chunk_size):
                    chunk = rows_to_insert[i:i + chunk_size]
                    errors = self.client.insert_rows_json(table, chunk)
                    
                    if errors:
                        logger.warning(f"Insert errors in chunk {i//chunk_size + 1}: {errors}")
                    else:
                        total_inserted += len(chunk)
                
                logger.info(f"Batch inserted {total_inserted} transfers from {len(transfers)} records")
                return total_inserted
            
            else:
                # Use streaming insert for smaller batches
                errors = self.client.insert_rows_json(table, rows_to_insert)
                
                if errors:
                    logger.error(f"Insert errors: {errors}")
                    return 0
                
                logger.info(f"Batch inserted {len(transfers)} transfer records")
                return len(transfers)
                
        except Exception as e:
            logger.error(f"Error storing transfers batch: {e}")
            return 0
    
    def _transfer_to_bigquery_row(self, transfer: Transfer) -> Dict[str, Any]:
        """Convert Transfer object to BigQuery row format"""
        return {
            "wallet_address": transfer.wallet_address,
            "token_address": transfer.token_address,
            "transfer_type": transfer.transfer_type.value,
            "timestamp": transfer.timestamp.isoformat() if transfer.timestamp else None,
            "cost_in_eth": transfer.cost_in_eth,
            "transaction_hash": transfer.transaction_hash,
            "block_number": transfer.block_number,
            "token_amount": transfer.token_amount,
            "token_symbol": transfer.token_symbol,
            "network": transfer.network,
            "platform": transfer.platform,
            "created_at": transfer.created_at.isoformat() if transfer.created_at else datetime.utcnow().isoformat(),
            "wallet_sophistication_score": transfer.wallet_sophistication_score
        }
    
    def _format_row_for_merge(self, row_data: Dict[str, Any]) -> str:
        """Format row data for MERGE statement"""
        # This is a simplified version - in production, you'd want proper SQL escaping
        values = []
        for key, value in row_data.items():
            if value is None:
                values.append("NULL")
            elif isinstance(value, str):
                values.append(f"'{value}'")
            elif isinstance(value, (int, float)):
                values.append(str(value))
            else:
                values.append(f"'{str(value)}'")
        
        return f"STRUCT({', '.join(values)})"
    
    async def get_transfers_by_wallet(self, wallet_address: str, 
                                    limit: int = 100, 
                                    transfer_type: Optional[TransferType] = None,
                                    days_back: Optional[int] = None) -> List[Transfer]:
        """Get transfers for a specific wallet"""
        if not self._initialized or self.client is None:
            return []
        
        try:
            conditions = ["wallet_address = @wallet_address"]
            params = {"wallet_address": wallet_address}
            
            if transfer_type:
                conditions.append("transfer_type = @transfer_type")
                params["transfer_type"] = transfer_type.value
            
            if days_back:
                conditions.append("timestamp >= @cutoff_date")
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                params["cutoff_date"] = cutoff_date
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                wallet_address, token_address, transfer_type, timestamp, cost_in_eth,
                transaction_hash, block_number, token_amount, token_symbol, network,
                platform, created_at, wallet_sophistication_score
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT @limit
            """
            
            params["limit"] = limit
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, type_map.get(type(value), "STRING"), value)
                    for name, value in params.items()
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            transfers = []
            for row in results:
                try:
                    transfer = self._bigquery_row_to_transfer(row)
                    transfers.append(transfer)
                except Exception as e:
                    logger.warning(f"Failed to parse transfer row: {e}")
                    continue
            
            return transfers
            
        except Exception as e:
            logger.error(f"Error getting transfers for wallet {wallet_address}: {e}")
            return []
    
    async def get_transfers_by_token(self, token_address: str, 
                                   limit: int = 100,
                                   transfer_type: Optional[TransferType] = None,
                                   days_back: Optional[int] = None) -> List[Transfer]:
        """Get transfers for a specific token"""
        if not self._initialized or self.client is None:
            return []
        
        try:
            conditions = ["token_address = @token_address"]
            params = {"token_address": token_address}
            
            if transfer_type:
                conditions.append("transfer_type = @transfer_type")
                params["transfer_type"] = transfer_type.value
            
            if days_back:
                conditions.append("timestamp >= @cutoff_date")
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                params["cutoff_date"] = cutoff_date
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                wallet_address, token_address, transfer_type, timestamp, cost_in_eth,
                transaction_hash, block_number, token_amount, token_symbol, network,
                platform, created_at, wallet_sophistication_score
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT @limit
            """
            
            params["limit"] = limit
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, "STRING" if isinstance(value, str) else 
                                                       "INTEGER" if isinstance(value, int) else 
                                                       "TIMESTAMP" if isinstance(value, datetime) else "STRING", value)
                    for name, value in params.items()
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            transfers = []
            total_eth_volume = 0
            unique_wallets = set()
            
            for row in results:
                try:
                    transfer = self._bigquery_row_to_transfer(row)
                    transfers.append(transfer)
                    total_eth_volume += transfer.cost_in_eth
                    unique_wallets.add(transfer.wallet_address)
                except Exception as e:
                    logger.warning(f"Failed to parse transfer row: {e}")
                    continue
            
            return transfers
            
        except Exception as e:
            logger.error(f"Error getting transfers for token {token_address}: {e}")
            return []
    
    async def get_transfer_stats(self, network: str = None, 
                               days_back: int = 30) -> Dict:
        """Get transfer statistics"""
        if not self._initialized or self.client is None:
            return {}
        
        try:
            conditions = ["timestamp >= @cutoff_date"]
            params = {"cutoff_date": datetime.utcnow() - timedelta(days=days_back)}
            
            if network:
                conditions.append("network = @network")
                params["network"] = network
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                transfer_type,
                network,
                COUNT(*) as count,
                SUM(cost_in_eth) as total_eth,
                COUNT(DISTINCT wallet_address) as unique_wallets,
                COUNT(DISTINCT token_address) as unique_tokens,
                AVG(cost_in_eth) as avg_eth
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            GROUP BY transfer_type, network
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", params["cutoff_date"]),
                ] + ([bigquery.ScalarQueryParameter("network", "STRING", params["network"])] if network else [])
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            stats = {
                "total_transfers": 0,
                "total_eth_volume": 0,
                "by_type": {},
                "by_network": {},
                "days_analyzed": days_back
            }
            
            for row in results:
                transfer_type = row.transfer_type
                network = row.network
                
                stats["total_transfers"] += row.count
                stats["total_eth_volume"] += row.total_eth or 0
                
                if transfer_type not in stats["by_type"]:
                    stats["by_type"][transfer_type] = {
                        "count": 0,
                        "total_eth": 0,
                        "networks": {}
                    }
                
                if network not in stats["by_network"]:
                    stats["by_network"][network] = {
                        "count": 0,
                        "total_eth": 0,
                        "types": {}
                    }
                
                # Update type stats
                stats["by_type"][transfer_type]["count"] += row.count
                stats["by_type"][transfer_type]["total_eth"] += row.total_eth or 0
                stats["by_type"][transfer_type]["networks"][network] = {
                    "count": row.count,
                    "total_eth": row.total_eth or 0,
                    "unique_wallets": row.unique_wallets,
                    "unique_tokens": row.unique_tokens
                }
                
                # Update network stats
                stats["by_network"][network]["count"] += row.count
                stats["by_network"][network]["total_eth"] += row.total_eth or 0
                stats["by_network"][network]["types"][transfer_type] = {
                    "count": row.count,
                    "total_eth": row.total_eth or 0,
                    "unique_wallets": row.unique_wallets,
                    "unique_tokens": row.unique_tokens
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting transfer stats: {e}")
            return {}
    
    async def cleanup_old_transfers(self, days_to_keep: int = 90) -> int:
        """Remove old transfer records to manage table size"""
        if not self._initialized or self.client is None:
            return 0
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            query = f"""
            DELETE FROM `{self.full_table_id}`
            WHERE created_at < @cutoff_date
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", cutoff_date)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            result = query_job.result()
            
            deleted_count = query_job.num_dml_affected_rows or 0
            logger.info(f"Cleaned up {deleted_count} old transfer records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old transfers: {e}")
            return 0
    
    async def get_top_tokens_by_volume(self, transfer_type: TransferType,
                                     network: str = None,
                                     days_back: int = 7,
                                     limit: int = 50) -> List[Dict]:
        """Get top tokens by ETH volume"""
        if not self._initialized or self.client is None:
            return []
        
        try:
            conditions = [
                "transfer_type = @transfer_type",
                "timestamp >= @cutoff_date"
            ]
            params = {
                "transfer_type": transfer_type.value,
                "cutoff_date": datetime.utcnow() - timedelta(days=days_back),
                "limit": limit
            }
            
            if network:
                conditions.append("network = @network")
                params["network"] = network
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                token_address,
                token_symbol,
                SUM(cost_in_eth) as total_eth_volume,
                COUNT(*) as transfer_count,
                COUNT(DISTINCT wallet_address) as unique_wallets_count,
                AVG(cost_in_eth) as avg_eth_per_transfer
            FROM `{self.full_table_id}`
            WHERE {where_clause}
            GROUP BY token_address, token_symbol
            ORDER BY total_eth_volume DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, 
                                                 "STRING" if isinstance(value, str) else 
                                                 "TIMESTAMP" if isinstance(value, datetime) else 
                                                 "INTEGER", value)
                    for name, value in params.items()
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            top_tokens = []
            for row in results:
                token_data = {
                    "token_address": row.token_address,
                    "token_symbol": row.token_symbol,
                    "total_eth_volume": float(row.total_eth_volume),
                    "transfer_count": row.transfer_count,
                    "unique_wallets_count": row.unique_wallets_count,
                    "avg_eth_per_transfer": float(row.avg_eth_per_transfer)
                }
                top_tokens.append(token_data)
            
            return top_tokens
            
        except Exception as e:
            logger.error(f"Error getting top tokens: {e}")
            return []
    
    def _bigquery_row_to_transfer(self, row) -> Transfer:
        """Convert BigQuery row to Transfer object"""
        return Transfer(
            wallet_address=row.wallet_address,
            token_address=row.token_address,
            transfer_type=TransferType(row.transfer_type),
            timestamp=row.timestamp,
            cost_in_eth=float(row.cost_in_eth),
            transaction_hash=row.transaction_hash,
            block_number=int(row.block_number),
            token_amount=float(row.token_amount),
            token_symbol=row.token_symbol,
            network=row.network,
            platform=row.platform,
            created_at=row.created_at,
            wallet_sophistication_score=float(row.wallet_sophistication_score) if row.wallet_sophistication_score else None
        )
    
    async def cleanup(self):
        """Cleanup BigQuery client"""
        if self.client:
            self.client.close()
            self._initialized = False