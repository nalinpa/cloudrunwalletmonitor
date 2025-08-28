import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

from services.database.transfer_service import TransferService
from api.models.data_models import TransferType
from utils.config import Config

logger = logging.getLogger(__name__)

class TransferRoutes:
    """API routes for transfer data management and querying"""
    
    @staticmethod
    async def get_transfer_stats(network: str = None, days_back: int = 30) -> Dict[str, Any]:
        """Get transfer statistics"""
        try:
            config = Config()
            transfer_service = TransferService(config)
            await transfer_service.initialize()
            
            stats = await transfer_service.get_transfer_stats(network, days_back)
            
            await transfer_service.cleanup()
            
            return {
                "success": True,
                "data": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Transfer stats failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_transfer_stats_sync(network: str = None, days_back: int = 30) -> Dict[str, Any]:
        """Sync wrapper for transfer stats"""
        try:
            return asyncio.run(TransferRoutes.get_transfer_stats(network, days_back))
        except Exception as e:
            logger.error(f"Sync transfer stats failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def get_wallet_transfers(wallet_address: str, 
                                 transfer_type: str = None,
                                 days_back: int = 30, 
                                 limit: int = 100) -> Dict[str, Any]:
        """Get transfers for a specific wallet"""
        try:
            config = Config()
            transfer_service = TransferService(config)
            await transfer_service.initialize()
            
            # Convert string to enum if provided
            transfer_type_enum = None
            if transfer_type:
                try:
                    transfer_type_enum = TransferType(transfer_type.lower())
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid transfer_type: {transfer_type}. Must be 'buy' or 'sell'",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            transfers = await transfer_service.get_transfers_by_wallet(
                wallet_address, limit, transfer_type_enum, days_back
            )
            
            # Convert transfers to dict format
            transfer_data = []
            for transfer in transfers:
                transfer_dict = transfer.to_dict()
                transfer_data.append(transfer_dict)
            
            await transfer_service.cleanup()
            
            return {
                "success": True,
                "data": {
                    "wallet_address": wallet_address,
                    "transfer_count": len(transfers),
                    "transfers": transfer_data
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Wallet transfers query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_wallet_transfers_sync(wallet_address: str, 
                                transfer_type: str = None,
                                days_back: int = 30, 
                                limit: int = 100) -> Dict[str, Any]:
        """Sync wrapper for wallet transfers"""
        try:
            return asyncio.run(TransferRoutes.get_wallet_transfers(
                wallet_address, transfer_type, days_back, limit
            ))
        except Exception as e:
            logger.error(f"Sync wallet transfers failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def get_token_transfers(token_address: str, 
                                transfer_type: str = None,
                                days_back: int = 30, 
                                limit: int = 100) -> Dict[str, Any]:
        """Get transfers for a specific token"""
        try:
            config = Config()
            transfer_service = TransferService(config)
            await transfer_service.initialize()
            
            # Convert string to enum if provided
            transfer_type_enum = None
            if transfer_type:
                try:
                    transfer_type_enum = TransferType(transfer_type.lower())
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid transfer_type: {transfer_type}. Must be 'buy' or 'sell'",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            transfers = await transfer_service.get_transfers_by_token(
                token_address, limit, transfer_type_enum, days_back
            )
            
            # Convert transfers to dict format
            transfer_data = []
            total_eth_volume = 0
            unique_wallets = set()
            
            for transfer in transfers:
                transfer_dict = transfer.to_dict()
                transfer_data.append(transfer_dict)
                total_eth_volume += transfer.cost_in_eth
                unique_wallets.add(transfer.wallet_address)
            
            await transfer_service.cleanup()
            
            return {
                "success": True,
                "data": {
                    "token_address": token_address,
                    "transfer_count": len(transfers),
                    "total_eth_volume": total_eth_volume,
                    "unique_wallets": len(unique_wallets),
                    "transfers": transfer_data
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Token transfers query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_token_transfers_sync(token_address: str, 
                               transfer_type: str = None,
                               days_back: int = 30, 
                               limit: int = 100) -> Dict[str, Any]:
        """Sync wrapper for token transfers"""
        try:
            return asyncio.run(TransferRoutes.get_token_transfers(
                token_address, transfer_type, days_back, limit
            ))
        except Exception as e:
            logger.error(f"Sync token transfers failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def get_top_tokens(transfer_type: str,
                           network: str = None,
                           days_back: int = 7,
                           limit: int = 50) -> Dict[str, Any]:
        """Get top tokens by volume"""
        try:
            # Convert string to enum
            try:
                transfer_type_enum = TransferType(transfer_type.lower())
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid transfer_type: {transfer_type}. Must be 'buy' or 'sell'",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            config = Config()
            transfer_service = TransferService(config)
            await transfer_service.initialize()
            
            top_tokens = await transfer_service.get_top_tokens_by_volume(
                transfer_type_enum, network, days_back, limit
            )
            
            await transfer_service.cleanup()
            
            return {
                "success": True,
                "data": {
                    "transfer_type": transfer_type,
                    "network": network,
                    "days_analyzed": days_back,
                    "token_count": len(top_tokens),
                    "tokens": top_tokens
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Top tokens query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_top_tokens_sync(transfer_type: str,
                          network: str = None,
                          days_back: int = 7,
                          limit: int = 50) -> Dict[str, Any]:
        """Sync wrapper for top tokens"""
        try:
            return asyncio.run(TransferRoutes.get_top_tokens(
                transfer_type, network, days_back, limit
            ))
        except Exception as e:
            logger.error(f"Sync top tokens failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def cleanup_old_transfers(days_to_keep: int = 90) -> Dict[str, Any]:
        """Cleanup old transfer records"""
        try:
            config = Config()
            transfer_service = TransferService(config)
            await transfer_service.initialize()
            
            deleted_count = await transfer_service.cleanup_old_transfers(days_to_keep)
            
            await transfer_service.cleanup()
            
            return {
                "success": True,
                "message": f"Cleaned up {deleted_count} old transfer records",
                "deleted_count": deleted_count,
                "days_kept": days_to_keep,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Transfer cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def cleanup_old_transfers_sync(days_to_keep: int = 90) -> Dict[str, Any]:
        """Sync wrapper for cleanup"""
        try:
            return asyncio.run(TransferRoutes.cleanup_old_transfers(days_to_keep))
        except Exception as e:
            logger.error(f"Sync cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Helper functions for backward compatibility
def get_transfer_stats_sync(network: str = None, days_back: int = 30) -> Dict[str, Any]:
    """Sync wrapper for transfer stats"""
    try:
        return TransferRoutes.get_transfer_stats_sync(network, days_back)
    except Exception as e:
        logger.error(f"Transfer stats wrapper failed: {e}")
        return {"error": str(e), "success": False}

def get_wallet_transfers_sync(wallet_address: str, **kwargs) -> Dict[str, Any]:
    """Sync wrapper for wallet transfers"""
    try:
        return TransferRoutes.get_wallet_transfers_sync(wallet_address, **kwargs)
    except Exception as e:
        logger.error(f"Wallet transfers wrapper failed: {e}")
        return {"error": str(e), "success": False}

# Export for use in other modules
__all__ = [
    'TransferRoutes',
    'get_transfer_stats_sync', 
    'get_wallet_transfers_sync'
]