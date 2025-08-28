import httpx
import asyncio
import logging
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TelegramClient:
    """Enhanced Telegram client with rich formatting and better error handling"""
    
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
                http2=True,  # Enable HTTP/2 for better performance
                verify=True,
                follow_redirects=True
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to Telegram with enhanced error handling"""
        if not self._client or not self.bot_token or not self.chat_id:
            logger.warning("Telegram client not configured")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "CryptoAnalysis/2.0",
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
    
    async def send_photo(self, photo_url: str, caption: str = "") -> bool:
        """Send a photo with optional caption"""
        if not self._client or not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"{self.base_url}/sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "photo": photo_url,
                "caption": caption[:1000] if caption else "",
                "parse_mode": "Markdown"
            }
            
            response = await self._client.post(url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            return False

# Global telegram client instance
telegram_client = TelegramClient()

class NotificationFormatter:
    """Format different types of notifications for Telegram - with trading links"""
    
    @staticmethod
    def _get_trading_links(token: str, contract_address: str, network: str) -> str:
        """Generate trading and info links for the token"""
        if not contract_address or len(contract_address) < 10:
            return ""
        
        # Clean contract address
        contract = contract_address.lower()
        if not contract.startswith('0x'):
            return ""
        
        # Network-specific links
        if network.lower() == 'ethereum':
            chain_id = '1'
            uniswap_link = f"https://app.uniswap.org/swap?outputCurrency={contract}"
            dexscreener_link = f"https://dexscreener.com/ethereum/{contract}"
            etherscan_link = f"https://etherscan.io/token/{contract}"
        elif network.lower() == 'base':
            chain_id = '8453'
            uniswap_link = f"https://app.uniswap.org/swap?outputCurrency={contract}&chain=base"
            dexscreener_link = f"https://dexscreener.com/base/{contract}"
            etherscan_link = f"https://basescan.org/token/{contract}"
        else:
            return ""
        
        # Additional useful links
        dextools_link = f"https://www.dextools.io/app/en/{network}/{contract}"
        
        links = f"""
📊 [DEXScreener]({dexscreener_link}) | 🔧 [DEXTools]({dextools_link})
🦄 [Uniswap]({uniswap_link}) | 🔍 [Explorer]({etherscan_link})"""
        
        return links
    
    @staticmethod
    def format_analysis_summary(result: Dict[str, Any]) -> str:
        """Format analysis results as a summary notification"""
        network = result.get('network', 'unknown').upper()
        analysis_type = result.get('analysis_type', 'analysis')
        total_transactions = result.get('total_transactions', 0)
        unique_tokens = result.get('unique_tokens', 0)
        total_eth_value = result.get('total_eth_value', 0)
        top_tokens = result.get('top_tokens', [])
        
        emoji = "🔍" if analysis_type == "buy" else "📉"
        type_text = analysis_type.replace('_', ' ').upper()
        
        message = f"{emoji} **{type_text} ANALYSIS**\n\n"
        message += f"🌐 **{network}** | {total_transactions:,} txs | {unique_tokens} tokens\n"
        message += f"💰 **Total:** {total_eth_value:.4f} ETH\n"
        
        if top_tokens and len(top_tokens) > 0:
            message += f"\n**Top Tokens:**\n"
            for i, (token, data, score) in enumerate(top_tokens[:3], 1):
                eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
                wallet_count = data.get('wallet_count', 0)
                contract = data.get('contract_address', '')
                
                message += f"{i}. `{token}` - {score:.1f} | {eth_value:.3f} ETH | {wallet_count}W\n"
                
                # Add quick links for the top token
                if i == 1 and contract:
                    links = NotificationFormatter._get_trading_links(token, contract, network.lower())
                    if links:
                        message += links + "\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        return message.strip()
    
    @staticmethod
    def format_high_value_alert(token: str, data: Dict[str, Any], network: str) -> str:
        """Format high-value token alert with trading links"""
        score = data.get('alpha_score', data.get('sell_pressure_score', 0))
        eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
        wallet_count = data.get('wallet_count', 0)
        contract = data.get('contract_address', '')
        
        if 'alpha_score' in data:
            emoji = "🆕"
            alert_type = "NEW TOKEN"
        else:
            emoji = "📉"
            alert_type = "SELL PRESSURE"
        
        message = f"{emoji} **{alert_type}**\n\n"
        message += f"🪙 **Token:** `{token}`\n"
        message += f"🌐 **Network:** {network.upper()}\n"
        message += f"📊 **Score:** {score:.1f}\n"
        message += f"💰 **ETH:** {eth_value:.4f}\n"
        message += f"👥 **Wallets:** {wallet_count}\n"
        
        # Add trading links
        if contract:
            message += f"📝 **Contract:** `{contract[:6]}...{contract[-4:]}`\n"
            links = NotificationFormatter._get_trading_links(token, contract, network)
            if links:
                message += links + "\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        return message.strip()
    
    @staticmethod
    def format_error_alert(error_type: str, error_message: str, network: str = None) -> str:
        """Format error notification - simple style"""
        network_text = f" ({network.upper()})" if network else ""
        
        message = f"❌ **ERROR{network_text}**\n\n"
        message += f"**Type:** {error_type}\n"
        message += f"**Details:** {error_message[:200]}{'...' if len(error_message) > 200 else ''}\n"
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return message.strip()
    
    @staticmethod
    def format_system_status(status: Dict[str, Any]) -> str:
        """Format system status notification - clean style"""
        service_status = "HEALTHY" if status.get('healthy', False) else "ISSUES"
        emoji = "✅" if status.get('healthy', False) else "❌"
        
        message = f"{emoji} **SYSTEM: {service_status}**\n\n"
        
        if 'database_connected' in status:
            db_status = "✅" if status['database_connected'] else "❌"
            message += f"Database: {db_status}\n"
        
        if 'alchemy_responsive' in status:
            api_status = "✅" if status['alchemy_responsive'] else "❌"
            message += f"Alchemy: {api_status}\n"
        
        if 'total_wallets' in status:
            message += f"Wallets: {status['total_wallets']:,}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        return message.strip()
    
    @staticmethod
    def format_test_message() -> str:
        """Format test notification - simple style"""
        return f"""🧪 **TEST NOTIFICATION**

✅ Crypto Monitor is working
📱 Telegram configured correctly

⏰ {datetime.now().strftime('%H:%M:%S')}
🚀 Ready for alerts"""

class NotificationService:
    """Main notification service that handles different types of alerts"""
    
    def __init__(self):
        self.formatter = NotificationFormatter()
        self.alert_thresholds = {
            'min_alpha_score': 25.0,
            'min_sell_score': 20.0,
            'min_eth_value': 0.05,
            'min_wallet_count': 1
        }
    
    async def send_analysis_summary(self, result: Dict[str, Any]) -> bool:
        """Send analysis summary notification"""
        try:
            message = self.formatter.format_analysis_summary(result)
            
            async with telegram_client:
                success = await telegram_client.send_message(message)
                
            if success:
                logger.info(f"Analysis summary sent for {result.get('network', 'unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send analysis summary: {e}")
            return False
    
    async def send_token_alerts(self, result: Dict[str, Any]) -> int:
        """Send alerts for high-scoring tokens, returns number of alerts sent"""
        alerts_sent = 0
        network = result.get('network', 'unknown')
        top_tokens = result.get('top_tokens', [])
        
        if not top_tokens:
            return 0
        
        try:
            async with telegram_client:
                for token, data, score in top_tokens[:10]:  # Limit to top 10
                    # Check if token meets alert thresholds
                    if self._should_alert(data, score):
                        message = self.formatter.format_high_value_alert(token, data, network)
                        
                        success = await telegram_client.send_message(message)
                        if success:
                            alerts_sent += 1
                            logger.info(f"Alert sent for token {token} on {network}")
                        
                        # Rate limiting between alerts
                        await asyncio.sleep(2)
            
            return alerts_sent
            
        except Exception as e:
            logger.error(f"Failed to send token alerts: {e}")
            return alerts_sent
    
    def _should_alert(self, data: Dict[str, Any], score: float) -> bool:
        """Determine if a token meets the alert criteria"""
        # Check score thresholds
        if 'alpha_score' in data and score < self.alert_thresholds['min_alpha_score']:
            return False
        if 'sell_pressure_score' in data and score < self.alert_thresholds['min_sell_score']:
            return False
        
        # Check ETH value
        eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
        if eth_value < self.alert_thresholds['min_eth_value']:
            return False
        
        # Check wallet count
        wallet_count = data.get('wallet_count', 0)
        if wallet_count < self.alert_thresholds['min_wallet_count']:
            return False
        
        return True
    
    async def send_error_notification(self, error_type: str, error_message: str, network: str = None) -> bool:
        """Send error notification"""
        try:
            message = self.formatter.format_error_alert(error_type, error_message, network)
            
            async with telegram_client:
                success = await telegram_client.send_message(message)
            
            if success:
                logger.info(f"Error notification sent: {error_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False
    
    async def send_system_status(self, status: Dict[str, Any]) -> bool:
        """Send system status notification"""
        try:
            message = self.formatter.format_system_status(status)
            
            async with telegram_client:
                success = await telegram_client.send_message(message)
            
            if success:
                logger.info("System status notification sent")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send system status: {e}")
            return False
    
    async def send_test_notification(self) -> bool:
        """Send test notification"""
        try:
            message = self.formatter.format_test_message()
            
            async with telegram_client:
                success = await telegram_client.send_message(message)
            
            if success:
                logger.info("Test notification sent successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send test notification: {e}")
            return False
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        logger.info(f"Alert thresholds updated: {self.alert_thresholds}")

# Global notification service
notification_service = NotificationService()

# Legacy compatibility function to match your existing format with trading links
def format_alert_message(alert: dict) -> str:
    """Format alert for Telegram - matches your existing style with trading links"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    token = alert.get('token', 'UNKNOWN')
    network = alert.get('network', 'ethereum')
    contract = data.get('contract_address', '')
    
    if alert_type == 'new_token':
        emoji = "🆕"
        score = data.get('alpha_score', 0)
    else:
        emoji = "📉"
        score = data.get('sell_score', 0)
    
    eth_value = data.get('total_eth_spent') or data.get('total_eth_value', 0)
    
    message = f"""
{emoji} **{alert_type.replace('_', ' ').upper()}**

🪙 **Token:** `{token}`
🌐 **Network:** {network.upper()}
📊 **Score:** {score:.1f}
💰 **ETH:** {eth_value:.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🎯 **Confidence:** {alert['confidence']}"""

    # Add contract address if available
    if contract and len(contract) > 10:
        message += f"\n📝 **Contract:** `{contract[:6]}...{contract[-4:]}`"
        
        # Add trading links
        links = NotificationFormatter._get_trading_links(token, contract, network)
        if links:
            message += f"\n{links}"
    
    message += f"\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"
    
    return message.strip()

# Updated convenience functions with simpler message style
async def send_alert_notifications(alerts: list):
    """Send alert notifications via Telegram - matches your existing function"""
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

async def send_error_notification(error_type: str, error_message: str, network: str = None) -> bool:
    """Send error notification"""
    return await notification_service.send_error_notification(error_type, error_message, network)

async def send_system_status(status: Dict[str, Any]) -> bool:
    """Send system status notification"""
    return await notification_service.send_system_status(status)

async def send_test_notification() -> bool:
    """Send test notification"""
    return await notification_service.send_test_notification()

def check_notification_config() -> bool:
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return bool(bot_token and chat_id and len(bot_token) > 40)

def get_notification_status() -> Dict[str, Any]:
    """Get current notification configuration status"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return {
        'configured': check_notification_config(),
        'bot_token_length': len(bot_token) if bot_token else 0,
        'chat_id_set': bool(chat_id),
        'thresholds': notification_service.alert_thresholds
    }