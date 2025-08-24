import httpx
import asyncio
import logging
from typing import Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramClient:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        if self.bot_token and self.chat_id:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def send_message(self, message: str) -> bool:
        if not self._client or not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = await self._client.post(url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

telegram_client = TelegramClient()

async def send_alert_notifications(alerts: list):
    """Send alert notifications via Telegram"""
    if not alerts:
        return
    
    async with telegram_client:
        for alert in alerts:
            message = format_alert_message(alert)
            await telegram_client.send_message(message)
            await asyncio.sleep(1)  # Rate limiting

async def send_test_notification():
    """Send test notification"""
    test_message = f" **TEST NOTIFICATION**\n\n Crypto Monitor is working!\n {datetime.now().strftime('%H:%M:%S')}"
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

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
        emoji = ""
        score = data.get('alpha_score', 0)
    else:
        emoji = ""
        score = data.get('sell_score', 0)
    
    eth_value = data.get('total_eth_spent') or data.get('total_eth_value', 0)
    
    message = f"""
{emoji} **{alert_type.replace('_', ' ').upper()}**

 **Token:** `{alert['token']}`
 **Network:** {alert['network'].upper()}
 **Score:** {score:.1f}
 **ETH:** {eth_value:.4f}
 **Wallets:** {data.get('wallet_count', 0)}
 **Confidence:** {alert['confidence']}

 {datetime.now().strftime('%H:%M:%S')}
"""
    
    return message.strip()
