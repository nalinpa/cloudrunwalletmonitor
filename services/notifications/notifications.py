import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

from utils.constants import NETWORK_DISPLAY, MESSAGE_TEMPLATES, NOTIFICATION_SETTINGS
from utils.web3_utils import (
    format_contract_address, generate_action_links, get_network_info,
    format_token_age, format_holder_count, get_combined_risk_indicator,
    extract_web3_data_safely, safe_float_conversion, safe_bool_conversion
)

logger = logging.getLogger(__name__)

class TelegramClient:
    """Streamlined Telegram client - consolidates duplicate HTTP logic"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        if self.is_configured():
            timeout = httpx.Timeout(30.0, connect=10.0)
            self._client = httpx.AsyncClient(
                timeout=timeout,
                http2=True,
                verify=True,
                follow_redirects=True
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    def is_configured(self) -> bool:
        """Check if properly configured"""
        return bool(self.bot_token and self.chat_id and len(self.bot_token) > 40)
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message with error handling"""
        if not self._client or not self.is_configured():
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
            
            response = await self._client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

class NotificationFormatter:
    """Consolidated message formatting - eliminates duplicate formatting logic"""
    
    @staticmethod
    def format_enhanced_alert(alert: dict) -> str:
        """✅ UPDATED: Unified alert formatting using consolidated utilities"""
        data = alert.get('data', {})
        alert_type = alert.get('alert_type', 'unknown')
        token = alert.get('token', 'UNKNOWN')
        network = alert.get('network', 'ethereum')
        confidence = alert.get('confidence', 'Unknown')
        
        # ✅ USING: Consolidated utility functions
        contract_address = (
            data.get('contract_address') or 
            data.get('ca') or 
            extract_web3_data_safely(data.get('web3_data'), 'contract_address', '')
        )
        
        # Extract Web3 data
        web3_data = data.get('web3_data', {})
        ai_data = alert.get('ai_data', {}) if len(alert) > 3 else {}
        
        # ✅ USING: Consolidated utility function
        network_info = get_network_info(network)
        
        # Build alert header
        if alert_type in ['new_token', 'buy']:
            emoji = "🟢"
            alert_title = "BUY ALERT"
            score = data.get('alpha_score', data.get('total_score', 0))
            eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
            wallet_count = data.get('wallet_count', 0)
            tx_count = data.get('total_purchases', data.get('transaction_count', 0))
        else:
            emoji = "🔴"
            alert_title = "SELL ALERT"
            score = data.get('sell_score', data.get('sell_pressure_score', data.get('total_score', 0)))
            eth_value = data.get('total_eth_received', data.get('total_eth_value', 0))
            wallet_count = data.get('wallet_count', 0)
            tx_count = data.get('total_sells', data.get('transaction_count', 0))

        # Extract token age and holder data for prominence
        token_age_hours = ai_data.get('token_age_hours') or web3_data.get('token_age_hours')
        holder_count = ai_data.get('holder_count') or web3_data.get('holder_count')
        
        # ✅ USING: Consolidated utility functions
        age_display = format_token_age(token_age_hours)
        holder_display = format_holder_count(holder_count)
        risk_indicator = get_combined_risk_indicator(token_age_hours, holder_count)

        # Build Web3 intelligence section
        web3_signals = []
        risk_signals = []
        
        # Extract Web3 signals from AI data
        if ai_data:
            # ✅ USING: Safe conversion functions
            if safe_bool_conversion(ai_data.get('is_verified')):
                web3_signals.append("✅ Contract Verified")
            else:
                risk_signals.append("⚠️ Unverified Contract")
            
            liquidity_usd = safe_float_conversion(ai_data.get('liquidity_usd'))
            if safe_bool_conversion(ai_data.get('has_liquidity')):
                if liquidity_usd > 100000:
                    web3_signals.append(f"💧 High Liquidity (${liquidity_usd:,.0f})")
                elif liquidity_usd > 10000:
                    web3_signals.append(f"💧 Good Liquidity (${liquidity_usd:,.0f})")
                else:
                    web3_signals.append("💧 Has Liquidity")
            else:
                risk_signals.append("🚨 No Liquidity Detected")
            
            if safe_bool_conversion(ai_data.get('smart_money_buying')) or safe_bool_conversion(ai_data.get('has_smart_money')):
                web3_signals.append("🧠 Smart Money Active")
            
            honeypot_risk = safe_float_conversion(ai_data.get('honeypot_risk'))
            if honeypot_risk > 0.7:
                risk_signals.append(f"🍯 HIGH Honeypot Risk ({honeypot_risk:.0%})")
            elif honeypot_risk > 0.4:
                risk_signals.append(f"⚠️ Medium Risk ({honeypot_risk:.0%})")

        # Build the message
        message_parts = [
            f"{emoji} **{alert_title}**",
            "",
            f"🪙 **Token:** `{token}`",
            f"🌐 **Network:** {network_info['name']} ({network_info['symbol']})",
            f"📊 **Score:** {score:.1f}",
            f"💰 **ETH Volume:** {eth_value:.4f}",
            f"👥 **Wallets:** {wallet_count}",
            f"🔄 **Transactions:** {tx_count}",
            age_display,
            holder_display,
            f"🎯 **Confidence:** {confidence}"
        ]

        # Add risk assessment
        if risk_indicator:
            message_parts.extend(["", risk_indicator])

        # Add Web3 intelligence (condensed)
        if web3_signals or risk_signals:
            message_parts.extend(["", "🔍 **Web3 Intelligence:**"])
            
            for signal in web3_signals[:3]:  # Limit to prevent overflow
                message_parts.append(f"  {signal}")
            
            for risk in risk_signals[:2]:
                message_parts.append(f"  {risk}")

        # ✅ USING: Consolidated utility functions for contract and links
        message_parts.extend([
            "",
            "📋 **Contract Address:**",
            format_contract_address(contract_address),
            "",
            "🔗 **Quick Actions:**",
            generate_action_links(token, contract_address, network),
            "",
            f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}",
            "🚀 Enhanced Web3 Monitoring"
        ])
        
        return "\n".join(message_parts)
    
    @staticmethod
    def format_test_message() -> str:
        """✅ USING: Consolidated message template"""
        return MESSAGE_TEMPLATES['health_check'].format(
            timestamp=datetime.now().strftime('%H:%M:%S UTC')
        )
    
    @staticmethod
    def format_analysis_start(network: str, analysis_type: str, num_wallets: int, 
                            days_back: float, store_alerts: bool) -> str:
        """✅ UPDATED: Using consolidated network info"""
        network_info = get_network_info(network)
        timing_info = f"⏰ {days_back}d back"
        storage_info = f"📊 Alert Storage: {'Enabled' if store_alerts else 'Disabled'}"
        
        return f"""🚀 **ANALYSIS STARTED**

**Network:** {network_info['name']} ({network_info['symbol']})
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
{timing_info}
{storage_info}

🚀 **Features:** Enhanced AI, Web3 Intelligence, Momentum Tracking
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
    
    @staticmethod
    def format_analysis_summary(result, network: str, alerts_sent: int, 
                              min_alpha_score: float, store_alerts: bool) -> str:
        """✅ UPDATED: Using consolidated network info"""
        network_info = get_network_info(network)
        
        alert_storage_info = "Enabled" if store_alerts else "Disabled"
        
        return f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

✅ **Alerts Sent:** {alerts_sent}
📈 **Total Tokens Found:** {result.unique_tokens}
💰 **Total ETH Volume:** {result.total_eth_value:.4f}
🔍 **Filter:** min score {min_alpha_score}
📊 **Alert Storage:** {alert_storage_info}

🌐 **Network:** {network_info['name']} ({network_info['symbol']})
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""

class TelegramService:
    """Unified Telegram service - consolidates all notification logic with BigQuery alerts"""
    
    def __init__(self, config=None):
        self.config = config
        self.client = TelegramClient()
        self.formatter = NotificationFormatter()
        self.bigquery_service = None  # Will be set externally
        
    def is_configured(self) -> bool:
        """Check if Telegram is configured"""
        return self.client.is_configured()
    
    def set_bigquery_service(self, bigquery_service):
        """Set BigQuery service for alert storage"""
        self.bigquery_service = bigquery_service
        logger.info("BigQuery service connected to Telegram notifications")
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message through client"""
        if not self.is_configured():
            logger.warning("Telegram not configured - skipping notification")
            return False
        
        try:
            async with self.client:
                return await self.client.send_message(message, parse_mode)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """✅ UPDATED: Test Telegram connection using consolidated logic"""
        if not self.is_configured():
            return {
                "configured": False,
                "error": "Bot token or chat ID missing",
                "bot_token_present": bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                "chat_id_present": bool(os.getenv('TELEGRAM_CHAT_ID'))
            }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                bot_token = self.client.bot_token
                
                # Test bot info
                bot_response = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
                
                if bot_response.status_code != 200:
                    return {"configured": False, "error": f"Bot API returned {bot_response.status_code}"}
                
                bot_data = bot_response.json()
                if not bot_data.get('ok'):
                    return {"configured": False, "error": f"Bot API error: {bot_data.get('description')}"}
                
                bot_info = bot_data.get('result', {})
                
                # Test chat access
                chat_response = await client.get(
                    f"https://api.telegram.org/bot{bot_token}/getChat",
                    params={"chat_id": self.client.chat_id}
                )
                
                chat_accessible = chat_response.status_code == 200
                
                return {
                    "configured": True,
                    "bot_valid": True,
                    "bot_username": bot_info.get('username'),
                    "bot_name": bot_info.get('first_name'),
                    "chat_accessible": chat_accessible,
                    "ready_for_notifications": chat_accessible,
                    "bigquery_connected": bool(self.bigquery_service),
                    "version": "consolidated"
                }
                
        except Exception as e:
            return {
                "configured": False,
                "error": f"Connection test failed: {str(e)}",
                "exception": True
            }
    
    async def send_test_notification(self) -> bool:
        """Send test notification"""
        message = self.formatter.format_test_message()
        return await self.send_message(message)
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, 
                                        min_alpha_score: float = 50.0, store_alerts: bool = True):
        """✅ UPDATED: Consolidated notification sending with BigQuery alert storage"""
        try:
            if not result.ranked_tokens:
                message = f"📊 **{result.analysis_type.upper()} Analysis Complete** - No tokens found for {network.upper()}"
                await self.send_message(message)
                return
            
            # Filter and prepare alerts using consolidated settings
            qualifying_alerts = self._prepare_alerts(result, network, min_alpha_score)
            
            if not qualifying_alerts:
                await self._send_no_alerts_message(result, network, min_alpha_score)
                return
            
            # Add momentum data and store alerts (if enabled and BigQuery available)
            enhanced_alerts = await self._enhance_with_momentum(qualifying_alerts, store_alerts)
            
            # Send notifications
            limited_alerts = enhanced_alerts[:max_tokens]
            
            if limited_alerts:
                await self._send_individual_alerts(limited_alerts)
                await self._send_trending_summary(network, result.analysis_type)
                await self._send_analysis_summary_message(result, network, len(limited_alerts), min_alpha_score, store_alerts)
                
                logger.info(f"Sent {len(limited_alerts)} notifications for {network}")
            else:
                await self._send_no_alerts_message(result, network, min_alpha_score)
                
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
            await self.send_message(f"❌ **Notification Error** - Analysis completed but failed to send alerts: {str(e)}")
    
    def _prepare_alerts(self, result, network: str, min_alpha_score: float) -> List[Dict]:
        """✅ UPDATED: Prepare alerts using consolidated settings"""
        qualifying_alerts = []
        
        for token_data in result.ranked_tokens:
            if len(token_data) >= 4:
                token, data, score, ai_data = token_data[0], token_data[1], token_data[2], token_data[3]
                if score >= min_alpha_score:
                    alert = {
                        'token': token,
                        'data': data,
                        'alert_type': result.analysis_type,
                        'network': network,
                        'confidence': ai_data.get('confidence', 'Unknown'),
                        'ai_data': ai_data,
                        'score': score,
                        'enhanced_score': score
                    }
                    qualifying_alerts.append(alert)
        
        return qualifying_alerts
    
    async def _enhance_with_momentum(self, alerts: List[Dict], store_alerts: bool) -> List[Dict]:
        """Add momentum data and optionally store alerts to BigQuery"""
        enhanced_alerts = []
        
        for alert in alerts:
            try:
                # Get momentum data (if BigQuery service available)
                momentum_data = {}
                if self.bigquery_service:
                    momentum_data = await self.bigquery_service.get_token_momentum(
                        alert['token'], alert['network'], days_back=5
                    )
                alert['momentum_data'] = momentum_data
                
                # Store alert to BigQuery (if enabled and service available)
                if store_alerts and self.bigquery_service:
                    try:
                        await self.bigquery_service.store_alert(alert)
                        logger.info(f"✅ Stored alert to BigQuery: {alert['token']} ({alert['alert_type']})")
                    except Exception as e:
                        logger.error(f"Failed to store alert to BigQuery: {e}")
                
                enhanced_alerts.append(alert)
                        
            except Exception as e:
                logger.error(f"Error enhancing alert for {alert['token']}: {e}")
                enhanced_alerts.append(alert)  # Include without momentum
        
        return enhanced_alerts
    
    async def _send_individual_alerts(self, alerts: List[Dict]):
        """Send individual alert notifications"""
        for alert in alerts:
            try:
                message = self.formatter.format_enhanced_alert(alert)
                await self.send_message(message)
                await asyncio.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
    
    async def _send_trending_summary(self, network: str, analysis_type: str):
        """Send trending summary if BigQuery service available"""
        if not self.bigquery_service:
            logger.info("BigQuery service not available - skipping trending summary")
            return
        
        try:
            trending = await self.bigquery_service.get_trending_tokens(
                network=network, hours_back=24, limit=7
            )
            
            if trending:
                # ✅ UPDATED: Format trending message using consolidated utilities
                message_parts = [
                    f"🔥 **24H TRENDING - {network.upper()}**",
                    ""
                ]
                
                for i, token_data in enumerate(trending[:5]):
                    net_score = token_data.get('net_momentum_score', 0)
                    emoji = "📈" if net_score > 0 else "📉" if net_score < 0 else "➡️"
                    
                    message_parts.append(
                        f"{emoji} {i+1}. **{token_data['token_symbol']}** (Net: {net_score:+.1f})"
                    )
                
                message_parts.extend([
                    "",
                    f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}"
                ])
                
                await self.send_message('\n'.join(message_parts))
                
        except Exception as e:
            logger.error(f"Trending summary failed: {e}")
    
    async def _send_analysis_summary_message(self, result, network: str, alerts_sent: int, 
                                           min_alpha_score: float, store_alerts: bool):
        """Send analysis summary"""
        message = self.formatter.format_analysis_summary(
            result, network, alerts_sent, min_alpha_score, store_alerts
        )
        await self.send_message(message)
    
    async def _send_no_alerts_message(self, result, network: str, min_alpha_score: float):
        """✅ UPDATED: Send no alerts found message using consolidated utilities"""
        network_info = get_network_info(network)
        
        message = f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

❌ **No alerts found** (min score: {min_alpha_score})
📈 **Total Tokens:** {result.unique_tokens}
💰 **Total ETH:** {result.total_eth_value:.4f}
🌐 **Network:** {network_info['name']} ({network_info['symbol']})

⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        await self.send_message(message)
    
    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int,
                                    days_back: float, use_smart_timing: bool, max_tokens: int,
                                    min_alpha_score: float, store_alerts: bool = True):
        """Send analysis start notification"""
        message = self.formatter.format_analysis_start(
            network, analysis_type, num_wallets, days_back, store_alerts
        )
        await self.send_message(message)

# Global instances for backwards compatibility
telegram_service = TelegramService()
telegram_client = TelegramClient()

# ============================================================================
# BACKWARDS COMPATIBLE FUNCTIONS - Using consolidated utilities
# ============================================================================

async def send_alert_notifications(alerts: list):
    """✅ UPDATED: Backwards compatible function using consolidated formatting"""
    async with telegram_client:
        for alert in alerts:
            try:
                message = NotificationFormatter.format_enhanced_alert(alert)
                await telegram_client.send_message(message)
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")

async def send_test_notification():
    """Backwards compatible function"""
    return await telegram_service.send_test_notification()

def check_notification_config():
    """Backwards compatible function"""
    return telegram_service.is_configured()

def format_alert_message(alert: dict) -> str:
    """✅ UPDATED: Backwards compatible function using consolidated formatting"""
    return NotificationFormatter.format_enhanced_alert(alert)

# Export consolidated interface
__all__ = [
    'TelegramService', 'TelegramClient', 'NotificationFormatter',
    'telegram_service', 'telegram_client',
    'send_alert_notifications', 'send_test_notification', 'check_notification_config',
    'format_alert_message'
]