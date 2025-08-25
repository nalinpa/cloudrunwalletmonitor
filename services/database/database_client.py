import motor.motor_asyncio
import logging
from typing import List, Dict, Optional
from api.models.data_models import WalletInfo
from utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database operations for Cloud Function"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            if self.config.mongo_uri:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.mongo_uri)
                self.db = self.client[self.config.db_name]
                logger.info("Database connected successfully")
            else:
                logger.warning("No MongoDB URI provided")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def get_top_wallets(self, network: str, limit: int = 50) -> List[WalletInfo]:
        """Get top wallets from database"""
        if not self.db:
            return []
        
        try:
            collection = self.db[self.config.wallets_collection]
            cursor = collection.find(
                {"network": network},
                {"address": 1, "score": 1, "_id": 0}
            ).sort("score", -1).limit(limit)
            
            wallets_data = await cursor.to_list(length=limit)
            
            return [
                WalletInfo(
                    address=wallet['address'],
                    score=wallet.get('score', 0),
                    network=network
                )
                for wallet in wallets_data
            ]
            
        except Exception as e:
            logger.error(f"Error fetching wallets: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup database connection"""
        if self.client:
            self.client.close()