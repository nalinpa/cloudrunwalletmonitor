import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Import your notification service
from services.notifications import (
    send_test_notification,
    send_analysis_summary,
    send_token_alerts,
    send_error_notification,
    send_system_status,
    check_notification_config,
    get_notification_status,
    notification_service
)

logger = logging.getLogger(__name__)

class NotificationRoutes:
    """API routes for notification management - fixed async handling"""
    
    @staticmethod
    async def test_notification() -> Dict[str, Any]:
        """Send test notification - ASYNC VERSION"""
        try:
            if not check_notification_config():
                return {
                    "success": False,
                    "error": "Telegram not configured",
                    "message": "Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # This is now properly awaited in an async function
            success = await send_test_notification()
            
            return {
                "success": success,
                "message": "Test notification sent successfully!" if success else "Failed to send test notification",
                "timestamp": datetime.utcnow().isoformat(),
                "configured": True
            }
            
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def test_notification_sync() -> Dict[str, Any]:
        """Send test notification - SYNC VERSION for non-async contexts"""
        try:
            # Use asyncio.run to handle the async function in sync context
            result = asyncio.run(NotificationRoutes.test_notification())
            return result
            
        except Exception as e:
            logger.error(f"Sync test notification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def get_status() -> Dict[str, Any]:
        """Get notification status - ASYNC VERSION"""
        try:
            status = get_notification_status()
            
            return {
                "configured": status['configured'],
                "bot_token_set": status['bot_token_length'] > 0,
                "chat_id_set": status['chat_id_set'],
                "alert_thresholds": status['thresholds'],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_status_sync() -> Dict[str, Any]:
        """Get notification status - SYNC VERSION"""
        try:
            return asyncio.run(NotificationRoutes.get_status())
        except Exception as e:
            logger.error(f"Sync status check failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def update_thresholds(thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Update alert thresholds - ASYNC VERSION"""
        try:
            # Validate threshold values
            valid_keys = {'min_alpha_score', 'min_sell_score', 'min_eth_value', 'min_wallet_count'}
            new_thresholds = {}
            
            for key, value in thresholds.items():
                if key in valid_keys:
                    try:
                        new_thresholds[key] = float(value)
                    except (ValueError, TypeError):
                        return {
                            "success": False,
                            "error": f"Invalid value for {key}: {value}",
                            "timestamp": datetime.utcnow().isoformat()
                        }
            
            if not new_thresholds:
                return {
                    "success": False,
                    "error": "No valid thresholds provided",
                    "valid_keys": list(valid_keys),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Update thresholds
            notification_service.update_alert_thresholds(new_thresholds)
            
            return {
                "success": True,
                "message": "Alert thresholds updated successfully",
                "updated_thresholds": new_thresholds,
                "current_thresholds": notification_service.alert_thresholds,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Threshold update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def update_thresholds_sync(thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Update alert thresholds - SYNC VERSION"""
        try:
            return asyncio.run(NotificationRoutes.update_thresholds(thresholds))
        except Exception as e:
            logger.error(f"Sync threshold update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def simulate_alert(alert_type: str = "buy") -> Dict[str, Any]:
        """Simulate alert notification - ASYNC VERSION"""
        try:
            if not check_notification_config():
                return {
                    "success": False,
                    "error": "Telegram not configured",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Create simulated analysis result
            if alert_type == 'buy':
                simulated_result = {
                    'network': 'ethereum',
                    'analysis_type': 'buy',
                    'total_transactions': 25,
                    'unique_tokens': 8,
                    'total_eth_value': 1.25,
                    'top_tokens': [
                        ('PEPE', {
                            'alpha_score': 45.2,
                            'total_eth_spent': 0.85,
                            'wallet_count': 12,
                            'contract_address': '0x6982508145454ce325ddbe47a25d4ec3d2311933'
                        }, 45.2),
                        ('WOJAK', {
                            'alpha_score': 38.7,
                            'total_eth_spent': 0.40,
                            'wallet_count': 8,
                            'contract_address': '0x5026f006b85729a8b14553fae6af249ad16c9aab'
                        }, 38.7)
                    ]
                }
            else:  # sell
                simulated_result = {
                    'network': 'base',
                    'analysis_type': 'sell',
                    'total_transactions': 18,
                    'unique_tokens': 6,
                    'total_eth_value': 0.75,
                    'top_tokens': [
                        ('DEGEN', {
                            'sell_pressure_score': 32.1,
                            'total_eth_value': 0.45,
                            'wallet_count': 9,
                            'contract_address': '0x4ed4e862860bed51a9570b96d89af5e1b0efefed'
                        }, 32.1)
                    ]
                }
            
            # Send notifications
            summary_sent = await send_analysis_summary(simulated_result)
            alerts_sent = await send_token_alerts(simulated_result)
            
            return {
                "success": True,
                "message": "Simulated alerts sent successfully",
                "summary_sent": summary_sent,
                "token_alerts_sent": alerts_sent,
                "simulated_data": {
                    "alert_type": alert_type,
                    "tokens": len(simulated_result['top_tokens']),
                    "network": simulated_result['network']
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Alert simulation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def simulate_alert_sync(alert_type: str = "buy") -> Dict[str, Any]:
        """Simulate alert notification - SYNC VERSION"""
        try:
            return asyncio.run(NotificationRoutes.simulate_alert(alert_type))
        except Exception as e:
            logger.error(f"Sync alert simulation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Helper functions for backward compatibility
def send_test_notification_sync() -> bool:
    """Sync wrapper for test notification"""
    try:
        result = NotificationRoutes.test_notification_sync()
        return result.get('success', False)
    except Exception as e:
        logger.error(f"Test notification sync wrapper failed: {e}")
        return False

def get_notification_status_sync() -> Dict[str, Any]:
    """Sync wrapper for status check"""
    try:
        return NotificationRoutes.get_status_sync()
    except Exception as e:
        logger.error(f"Status check sync wrapper failed: {e}")
        return {"error": str(e), "configured": False}

# Export for use in other modules
__all__ = [
    'NotificationRoutes',
    'send_test_notification_sync', 
    'get_notification_status_sync'
]