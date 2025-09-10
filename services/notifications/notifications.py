# services/notifications/notifications.py - Updated with storage status notifications

"""
Unified Telegram Notification Service
Combines existing TelegramClient with enhanced notification features
"""

import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramClient:
    """Your existing TelegramClient class with HTTP/2 support"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        if self.bot_token and self.chat_id:
            # Configure HTTP/2 client with better timeout and retry settings
            timeout = httpx.Timeout(30.0, connect=10.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=True,  # Enable HTTP/2
                verify=True,
                follow_redirects=True
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def send_message(self, message: str) -> bool:
        if not self._client or not self.bot_token or not self.chat_id:
            logger.warning("Telegram client not configured")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            # Use HTTP/2 with proper headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "CryptoMonitor/2.0",
                "Accept": "application/json"
            }
            
            response = await self._client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except httpx.TimeoutException:
            logger.error("Telegram request timeout")
            return False
        except httpx.HTTPError as e:
            logger.error(f"Telegram HTTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

# Global telegram client instance
telegram_client = TelegramClient()

def check_notification_config():
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return bool(bot_token and chat_id and len(bot_token) > 40)

def format_alert_message(alert: dict) -> str:
    """Format alert for Telegram"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    
    if alert_type == 'new_token':
        emoji = "🆕"
        score = data.get('alpha_score', 0)
    else:
        emoji = "📉"
        score = data.get('sell_score', 0)
    
    eth_value = data.get('total_eth_spent') or data.get('total_eth_value', 0)
    
    message = f"""
{emoji} **{alert_type.replace('_', ' ').upper()}**

🪙 **Token:** `{alert['token']}`
🌐 **Network:** {alert['network'].upper()}
📊 **Score:** {score:.1f}
💰 **ETH:** {eth_value:.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🎯 **Confidence:** {alert['confidence']}

⏰ {datetime.now().strftime('%H:%M:%S')}
🚀 HTTP/2 Enabled
"""
    
    return message.strip()

async def send_alert_notifications(alerts: list):
    """Send alert notifications via Telegram"""
    if not alerts:
        return
    
    async with telegram_client:
        for alert in alerts:
            message = format_alert_message(alert)
            success = await telegram_client.send_message(message)
            if success:
                logger.info(f"Alert sent for {alert.get('token', 'unknown')}")
            else:
                logger.error(f"Failed to send alert for {alert.get('token', 'unknown')}")
            
            # Rate limiting - wait between messages
            await asyncio.sleep(1)

async def send_test_notification():
    """Send test notification"""
    test_message = f"🧪 **TEST NOTIFICATION**\n\n✅ Crypto Monitor is working!\n🕐 {datetime.now().strftime('%H:%M:%S')}\n🚀 HTTP/2 Enabled"
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

# ============================================================================
# ENHANCED TELEGRAM SERVICE - New functionality for modular architecture
# ============================================================================

class TelegramService:
    """Enhanced Telegram service that wraps the existing TelegramClient"""
    
    def __init__(self):
        # Use existing configuration check
        pass
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured using existing function"""
        return check_notification_config()
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message using existing telegram client"""
        if not self.is_configured():
            logger.warning("Telegram not configured - skipping notification")
            return False
        
        try:
            async with telegram_client:
                return await telegram_client.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Telegram connection and return detailed status"""
        if not self.is_configured():
            return {
                "configured": False,
                "error": "Bot token or chat ID missing",
                "bot_token_present": bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                "chat_id_present": bool(os.getenv('TELEGRAM_CHAT_ID'))
            }
        
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            # Test bot info
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check bot info
                bot_response = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
                
                if bot_response.status_code != 200:
                    return {
                        "configured": False,
                        "error": f"Bot API returned {bot_response.status_code}",
                        "bot_valid": False
                    }
                
                bot_data = bot_response.json()
                if not bot_data.get('ok'):
                    return {
                        "configured": False,
                        "error": f"Bot API error: {bot_data.get('description')}",
                        "bot_valid": False
                    }
                
                bot_info = bot_data.get('result', {})
                
                # Test chat access
                chat_response = await client.get(
                    f"https://api.telegram.org/bot{bot_token}/getChat",
                    params={"chat_id": chat_id}
                )
                
                chat_accessible = chat_response.status_code == 200
                chat_error = None
                
                if not chat_accessible:
                    try:
                        chat_data = chat_response.json()
                        chat_error = chat_data.get('description', 'Unknown chat error')
                    except:
                        chat_error = f"HTTP {chat_response.status_code}"
                
                return {
                    "configured": True,
                    "bot_valid": True,
                    "bot_username": bot_info.get('username'),
                    "bot_name": bot_info.get('first_name'),
                    "chat_accessible": chat_accessible,
                    "chat_error": chat_error,
                    "ready_for_notifications": chat_accessible
                }
                
        except Exception as e:
            return {
                "configured": False,
                "error": f"Connection test failed: {str(e)}",
                "exception": True
            }
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, min_alpha_score: float = 50.0):
        """Send notifications for analysis results"""
        try:
            if not result.ranked_tokens:
                await self.send_message(f"📊 **Analysis Complete** - No tokens found for {network.upper()}")
                return
            
            # Filter tokens by score
            qualifying_tokens = []
            for token_data in result.ranked_tokens:
                if len(token_data) >= 3 and token_data[2] >= min_alpha_score:
                    qualifying_tokens.append(token_data)
            
            limited_tokens = qualifying_tokens[:max_tokens]
            
            if limited_tokens:
                # Send individual notifications
                for i, token_tuple in enumerate(limited_tokens):
                    token = token_tuple[0]
                    data = token_tuple[1] 
                    score = token_tuple[2]
                    
                    if result.analysis_type == "buy":
                        message = f"""🟢 **BUY ALERT**

🪙 **Token:** `{token}`
📊 **Score:** {score:.1f}
🌐 **Network:** {network.upper()}

💰 **ETH Spent:** {data.get('total_eth_spent', 0):.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🔄 **Purchases:** {data.get('total_purchases', 0)}
⭐ **Avg Wallet Score:** {data.get('avg_wallet_score', 0):.1f}

🏆 **Rank:** #{i+1} of {len(limited_tokens)}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                    else:  # sell
                        message = f"""🔴 **SELL ALERT**

🪙 **Token:** `{token}`
📊 **Sell Pressure:** {score:.1f}
🌐 **Network:** {network.upper()}

💰 **ETH Received:** {data.get('total_eth_received', 0):.4f}
👥 **Wallets Selling:** {data.get('wallet_count', 0)}
🔄 **Sells:** {data.get('total_sells', 0)}
⭐ **Avg Wallet Score:** {data.get('avg_wallet_score', 0):.1f}

🏆 **Rank:** #{i+1} of {len(limited_tokens)}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                    
                    await self.send_message(message.strip())
                    await asyncio.sleep(1.5)  # Rate limiting
                
                # Send summary with storage info
                storage_info = ""
                if hasattr(result, 'performance_metrics'):
                    transfers_stored = result.performance_metrics.get('transfers_stored', 0)
                    if transfers_stored > 0:
                        storage_info = f"\n🗄️ **Stored:** {transfers_stored} transfer records"
                    else:
                        storage_info = f"\n🗄️ **Storage:** Disabled (no data stored)"
                
                summary_message = f"""📊 **{result.analysis_type.upper()} ANALYSIS SUMMARY**

✅ **Alerts Sent:** {len(limited_tokens)}
📈 **Total Tokens Found:** {result.unique_tokens}
💰 **Total ETH Volume:** {result.total_eth_value:.4f}
🔍 **Filtering:** min score {min_alpha_score}, max {max_tokens} tokens{storage_info}

🌐 **Network:** {network.upper()}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                
                await self.send_message(summary_message.strip())
                logger.info(f"Sent {len(limited_tokens)} notifications for {network}")
                
            else:
                # No qualifying tokens
                max_score = max([t[2] for t in result.ranked_tokens[:3]]) if result.ranked_tokens else 0
                
                # Check storage status
                storage_info = ""
                if hasattr(result, 'performance_metrics'):
                    transfers_stored = result.performance_metrics.get('transfers_stored', 0)
                    if transfers_stored > 0:
                        storage_info = f"\n🗄️ **Stored:** {transfers_stored} records"
                    else:
                        storage_info = f"\n🗄️ **Storage:** Disabled"
                
                message = f"""📊 **{result.analysis_type.upper()} ANALYSIS - NO ALERTS**

🌐 **Network:** {network.upper()}
📊 **Tokens Found:** {result.unique_tokens}
🚫 **Above {min_alpha_score} Score:** 0
📈 **Highest Score:** {max_score:.1f}{storage_info}

💡 **Tip:** Lower min_alpha_score to see more alerts
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                
                await self.send_message(message.strip())
                logger.info(f"No qualifying tokens found for {network} (max score: {max_score:.1f})")
                
        except Exception as e:
            logger.error(f"Failed to send analysis notifications: {e}")
            await self.send_message(f"❌ **Notification Error** - Analysis completed but failed to send alerts: {str(e)}")
    
    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int, 
                                    days_back: float, use_smart_timing: bool, max_tokens: int, 
                                    min_alpha_score: float, store_data: bool = False):
        """Send analysis start notification with storage status"""
        timing_info = f"⏰ Smart: {days_back}d" if use_smart_timing else f"⏰ Manual: {days_back}d"
        storage_info = f"🗄️ Storage: {'Enabled' if store_data else 'Disabled'}"
        
        start_message = f"""🚀 **ENHANCED ANALYSIS STARTED**

**Network:** {network.upper()}
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
{timing_info}
{storage_info}
**AI Enhancement:** pandas-ta Enabled
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score

⏰ {datetime.now().strftime('%H:%M:%S')}
🚀 HTTP/2 Enabled"""
        
        await self.send_message(start_message)

# Global enhanced service instance
telegram_service = TelegramService()

# Export all functions for backward compatibility
__all__ = [
    'TelegramClient',
    'TelegramService', 
    'telegram_client',
    'telegram_service',
    'send_alert_notifications',
    'send_test_notification',
    'check_notification_config',
    'format_alert_message'
]