import motor.motor_asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pymongo import IndexModel, ASCENDING, DESCENDING
from api.models.data_models import Transfer, TransferType
from utils.config import Config

logger = logging.getLogger(__name__)

class TransferService:
    """Service for managing transfer records in database"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self.collection_name = "transfers"
        self._connected = False
    
    async def initialize(self):
        """Initialize database connection and setup indexes"""
        try:
            if self.config.mongo_uri:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.mongo_uri)
                self.db = self.client[self.config.db_name]
                self.collection = self.db[self.collection_name]
                
                # Test the connection
                await self.client.admin.command('ping')
                self._connected = True
                
                # Setup indexes for efficient querying
                await self._setup_indexes()
                
                logger.info("Transfer service initialized successfully")
            else:
                logger.warning("No MongoDB URI provided")
                self._connected = False
        except Exception as e:
            logger.error(f"Transfer service initialization failed: {e}")
            self._connected = False
            raise
    
    async def _setup_indexes(self):
        """Setup database indexes for optimal query performance"""
        try:
            indexes = [
                # Primary query indexes
                IndexModel([("wallet_address", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("token_address", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("transfer_type", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("network", ASCENDING), ("timestamp", DESCENDING)]),
                
                # Composite indexes for common queries
                IndexModel([("wallet_address", ASCENDING), ("transfer_type", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("token_address", ASCENDING), ("transfer_type", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("network", ASCENDING), ("transfer_type", ASCENDING), ("timestamp", DESCENDING)]),
                
                # Analysis indexes
                IndexModel([("timestamp", DESCENDING), ("cost_in_eth", DESCENDING)]),
                IndexModel([("block_number", DESCENDING)]),
                IndexModel([("transaction_hash", ASCENDING)]),
                
                # Time-based indexes for cleanup
                IndexModel([("created_at", ASCENDING)]),
            ]
            
            await self.collection.create_indexes(indexes)
            logger.info(f"Created {len(indexes)} indexes for transfers collection")
            
        except Exception as e:
            logger.error(f"Failed to setup indexes: {e}")
    
    async def store_transfer(self, transfer: Transfer) -> bool:
        """Store a single transfer record"""
        if not self._connected or self.collection is None:
            logger.error("Database not connected")
            return False
        
        try:
            transfer_dict = transfer.to_dict()
            
            # Use upsert to avoid duplicates based on transaction hash
            await self.collection.update_one(
                {"transaction_hash": transfer.transaction_hash},
                {"$set": transfer_dict},
                upsert=True
            )
            
            logger.debug(f"Stored transfer: {transfer.wallet_address} -> {transfer.token_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing transfer: {e}")
            return False
    
    async def store_transfers_batch(self, transfers: List[Transfer]) -> int:
        """Store multiple transfer records efficiently"""
        if not self._connected or self.collection is None:
            logger.error("Database not connected")
            return 0
        
        if not transfers:
            return 0
        
        try:
            operations = []
            for transfer in transfers:
                transfer_dict = transfer.to_dict()
                operations.append({
                    "updateOne": {
                        "filter": {"transaction_hash": transfer.transaction_hash},
                        "update": {"$set": transfer_dict},
                        "upsert": True
                    }
                })
            
            # Batch process in chunks to avoid memory issues
            chunk_size = 1000
            total_stored = 0
            
            for i in range(0, len(operations), chunk_size):
                chunk = operations[i:i + chunk_size]
                result = await self.collection.bulk_write(chunk)
                total_stored += result.upserted_count + result.modified_count
            
            logger.info(f"Stored {total_stored} transfers from {len(transfers)} records")
            return total_stored
            
        except Exception as e:
            logger.error(f"Error storing transfers batch: {e}")
            return 0
    
    async def get_transfers_by_wallet(self, wallet_address: str, 
                                    limit: int = 100, 
                                    transfer_type: Optional[TransferType] = None,
                                    days_back: Optional[int] = None) -> List[Transfer]:
        """Get transfers for a specific wallet"""
        if not self._connected or self.collection is None:
            return []
        
        try:
            query = {"wallet_address": wallet_address}
            
            if transfer_type:
                query["transfer_type"] = transfer_type.value
            
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query["timestamp"] = {"$gte": cutoff_date}
            
            cursor = self.collection.find(query).sort("timestamp", DESCENDING).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            transfers = []
            for doc in documents:
                try:
                    transfer = Transfer.from_dict(doc)
                    transfers.append(transfer)
                except Exception as e:
                    logger.warning(f"Failed to parse transfer document: {e}")
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
        if not self._connected or self.collection is None:
            return []
        
        try:
            query = {"token_address": token_address}
            
            if transfer_type:
                query["transfer_type"] = transfer_type.value
            
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query["timestamp"] = {"$gte": cutoff_date}
            
            cursor = self.collection.find(query).sort("timestamp", DESCENDING).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            transfers = []
            for doc in documents:
                try:
                    transfer = Transfer.from_dict(doc)
                    transfers.append(transfer)
                except Exception as e:
                    logger.warning(f"Failed to parse transfer document: {e}")
                    continue
            
            return transfers
            
        except Exception as e:
            logger.error(f"Error getting transfers for token {token_address}: {e}")
            return []
    
    async def get_transfer_stats(self, network: str = None, 
                               days_back: int = 30) -> Dict:
        """Get transfer statistics"""
        if not self._connected or self.collection is None:
            return {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            match_stage = {"timestamp": {"$gte": cutoff_date}}
            
            if network:
                match_stage["network"] = network
            
            pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": {
                            "transfer_type": "$transfer_type",
                            "network": "$network"
                        },
                        "count": {"$sum": 1},
                        "total_eth": {"$sum": "$cost_in_eth"},
                        "unique_wallets": {"$addToSet": "$wallet_address"},
                        "unique_tokens": {"$addToSet": "$token_address"},
                        "avg_eth": {"$avg": "$cost_in_eth"}
                    }
                },
                {
                    "$project": {
                        "transfer_type": "$_id.transfer_type",
                        "network": "$_id.network",
                        "count": 1,
                        "total_eth": 1,
                        "unique_wallets_count": {"$size": "$unique_wallets"},
                        "unique_tokens_count": {"$size": "$unique_tokens"},
                        "avg_eth": 1
                    }
                }
            ]
            
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            stats = {
                "total_transfers": 0,
                "total_eth_volume": 0,
                "by_type": {},
                "by_network": {},
                "days_analyzed": days_back
            }
            
            for result in results:
                transfer_type = result["transfer_type"]
                network = result["network"]
                
                stats["total_transfers"] += result["count"]
                stats["total_eth_volume"] += result["total_eth"]
                
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
                stats["by_type"][transfer_type]["count"] += result["count"]
                stats["by_type"][transfer_type]["total_eth"] += result["total_eth"]
                stats["by_type"][transfer_type]["networks"][network] = {
                    "count": result["count"],
                    "total_eth": result["total_eth"],
                    "unique_wallets": result["unique_wallets_count"],
                    "unique_tokens": result["unique_tokens_count"]
                }
                
                # Update network stats
                stats["by_network"][network]["count"] += result["count"]
                stats["by_network"][network]["total_eth"] += result["total_eth"]
                stats["by_network"][network]["types"][transfer_type] = {
                    "count": result["count"],
                    "total_eth": result["total_eth"],
                    "unique_wallets": result["unique_wallets_count"],
                    "unique_tokens": result["unique_tokens_count"]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting transfer stats: {e}")
            return {}
    
    async def cleanup_old_transfers(self, days_to_keep: int = 90) -> int:
        """Remove old transfer records to manage database size"""
        if not self._connected or self.collection is None:
            return 0
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            result = await self.collection.delete_many({
                "created_at": {"$lt": cutoff_date}
            })
            
            deleted_count = result.deleted_count
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
        if not self._connected or self.collection is None:
            return []
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            match_stage = {
                "transfer_type": transfer_type.value,
                "timestamp": {"$gte": cutoff_date}
            }
            
            if network:
                match_stage["network"] = network
            
            pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": {
                            "token_address": "$token_address",
                            "token_symbol": "$token_symbol"
                        },
                        "total_eth_volume": {"$sum": "$cost_in_eth"},
                        "transfer_count": {"$sum": 1},
                        "unique_wallets": {"$addToSet": "$wallet_address"},
                        "avg_eth_per_transfer": {"$avg": "$cost_in_eth"}
                    }
                },
                {
                    "$project": {
                        "token_address": "$_id.token_address",
                        "token_symbol": "$_id.token_symbol",
                        "total_eth_volume": 1,
                        "transfer_count": 1,
                        "unique_wallets_count": {"$size": "$unique_wallets"},
                        "avg_eth_per_transfer": 1
                    }
                },
                {"$sort": {"total_eth_volume": DESCENDING}},
                {"$limit": limit}
            ]
            
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top tokens: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup database connection"""
        if self.client:
            self.client.close()
            self._connected = False