import motor.motor_asyncio
import logging
from typing import List, Dict, Optional
from api.models.data_models import WalletInfo
from utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database operations for Cloud Function - Fixed for Motor driver"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
        self._connected = False
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            if self.config.mongo_uri:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.mongo_uri)
                self.db = self.client[self.config.db_name]
                
                # Test the connection
                await self.client.admin.command('ping')
                self._connected = True
                logger.info("Database connected successfully")
            else:
                logger.warning("No MongoDB URI provided")
                self._connected = False
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._connected = False
            raise
    
    async def get_top_wallets(self, network: str, limit: int = 50) -> List[WalletInfo]:
        """Get top wallets from database - all wallets since they're not network-specific"""
        # Fix: Use self._connected instead of self.db
        if not self._connected or self.db is None:
            logger.error("Database not connected")
            return []
        
        try:
            collection = self.db[self.config.wallets_collection]
            
            # Get all wallets, sorted by score (your field names are correct: address, score)
            cursor = collection.find(
                {"address": {"$exists": True}, "score": {"$exists": True}},
                {"address": 1, "score": 1, "_id": 0}
            ).sort("score", -1).limit(limit)
            
            wallets_data = await cursor.to_list(length=limit)
            
            wallets = []
            for wallet in wallets_data:
                if wallet.get('address') and wallet.get('score') is not None:
                    wallets.append(WalletInfo(
                        address=wallet['address'],
                        score=float(wallet['score']),
                        network=network  # Network is provided as parameter for context
                    ))
            
            logger.info(f"Retrieved {len(wallets)} wallets for {network} analysis")
            return wallets
            
        except Exception as e:
            logger.error(f"Error fetching wallets: {e}")
            return []
    
    async def get_wallet_stats(self) -> Dict:
        """Get general wallet statistics for debugging"""
        # Fix: Use self._connected instead of self.db
        if not self._connected or self.db is None:
            return {"error": "Database not connected"}
        
        try:
            collection = self.db[self.config.wallets_collection]
            
            # Count total documents
            total_count = await collection.count_documents({})
            
            # Check field names
            sample_doc = await collection.find_one({})
            
            # Count documents with score/rating fields
            score_count = await collection.count_documents({"score": {"$exists": True}})
            rating_count = await collection.count_documents({"rating": {"$exists": True}})
            address_count = await collection.count_documents({"address": {"$exists": True}})
            
            # Get sample documents
            sample_docs = await collection.find({}).limit(3).to_list(length=3)
            
            return {
                "total_documents": total_count,
                "documents_with_score": score_count,
                "documents_with_rating": rating_count,
                "documents_with_address": address_count,
                "sample_document_fields": list(sample_doc.keys()) if sample_doc else [],
                "sample_documents": [
                    {
                        "fields": list(doc.keys()),
                        "has_address": "address" in doc,
                        "has_score": "score" in doc,
                        "has_rating": "rating" in doc,
                        "address_length": len(doc.get("address", "")) if doc.get("address") else 0,
                        "score_value": doc.get("score"),
                        "rating_value": doc.get("rating")
                    } for doc in sample_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting wallet stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup database connection"""
        if self.client:
            self.client.close()
            self._connected = False