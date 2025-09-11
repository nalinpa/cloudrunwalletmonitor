import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramClient:
    """Enhanced TelegramClient with rich formatting support"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        if self.bot_token and self.chat_id:
            timeout = httpx.Timeout(30.0, connect=10.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=True,
                verify=True,
                follow_redirects=True
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
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
                "User-Agent": "CryptoMonitor/3.0",
                "Accept": "application/json"
            }
            
            response = await self._client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Enhanced Telegram message sent successfully")
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

def get_network_info(network: str) -> Dict[str, str]:
    """Get network-specific information for links and explorers"""
    network_configs = {
        'ethereum': {
            'name': 'Ethereum',
            'symbol': 'ETH',
            'explorer': 'etherscan.io',
            'uniswap_base': 'https://app.uniswap.org/#/swap?outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/ethereum/',
            'chain_id': '1'
        },
        'base': {
            'name': 'Base',
            'symbol': 'ETH',
            'explorer': 'basescan.org',
            'uniswap_base': 'https://app.uniswap.org/#/swap?chain=base&outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/base/',
            'chain_id': '8453'
        },
        'arbitrum': {
            'name': 'Arbitrum',
            'symbol': 'ETH', 
            'explorer': 'arbiscan.io',
            'uniswap_base': 'https://app.uniswap.org/#/swap?chain=arbitrum&outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/arbitrum/',
            'chain_id': '42161'
        }
    }
    
    return network_configs.get(network.lower(), network_configs['ethereum'])

def format_contract_address(contract_address: str) -> str:
    """Format contract address for easy copying"""
    if not contract_address or len(contract_address) < 10:
        return "❓ No CA"
    
    # Clean the address
    clean_address = contract_address.strip().lower()
    if not clean_address.startswith('0x'):
        clean_address = '0x' + clean_address
    
    # Format for display with monospace font
    return f"`{clean_address}`"

def generate_action_links(token: str, contract_address: str, network: str) -> str:
    """Generate action links for trading and research"""
    if not contract_address or len(contract_address) < 10:
        return ""
    
    network_info = get_network_info(network)
    
    # Clean contract address
    clean_ca = contract_address.strip().lower()
    if not clean_ca.startswith('0x'):
        clean_ca = '0x' + clean_ca
    
    # Generate links
    links = []
    
    # Uniswap link
    uniswap_url = f"{network_info['uniswap_base']}{clean_ca}"
    links.append(f"[🦄 Buy on Uniswap]({uniswap_url})")
    
    # DexScreener link
    dexscreener_url = f"{network_info['dexscreener_base']}{clean_ca}"
    links.append(f"[📊 DexScreener]({dexscreener_url})")
    
    # Block explorer link
    explorer_url = f"https://{network_info['explorer']}/token/{clean_ca}"
    links.append(f"[🔍 Explorer]({explorer_url})")
    
    # Twitter/X search link
    twitter_search = f"https://twitter.com/search?q={token}%20{clean_ca[:8]}"
    links.append(f"[🐦 Search X]({twitter_search})")
    
    return " | ".join(links)

def format_enhanced_alert_message(alert: dict) -> str:
    """Format enhanced alert with contract address and action links"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    token = alert.get('token', 'UNKNOWN')
    network = alert.get('network', 'ethereum')
    confidence = alert.get('confidence', 'Unknown')
    
    # Get contract address from data
    contract_address = data.get('contract_address', '')
    if not contract_address:
        # Try alternative locations
        contract_address = data.get('ca', '')
        if not contract_address and 'web3_data' in data:
            contract_address = data['web3_data'].get('contract_address', '')
    
    # Get network info
    network_info = get_network_info(network)
    
    if alert_type == 'new_token' or alert_type == 'buy':
        emoji = "🟢"
        alert_title = "BUY ALERT"
        score = data.get('alpha_score', data.get('total_score', 0))
        eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
        wallet_count = data.get('wallet_count', data.get('unique_wallets', 0))
        tx_count = data.get('total_purchases', data.get('transaction_count', 0))
    else:
        emoji = "🔴"
        alert_title = "SELL ALERT"
        score = data.get('sell_score', data.get('sell_pressure_score', data.get('total_score', 0)))
        eth_value = data.get('total_eth_received', data.get('total_eth_value', 0))
        wallet_count = data.get('wallet_count', data.get('unique_wallets', 0))
        tx_count = data.get('total_sells', data.get('transaction_count', 0))
    
    # Enhanced message format
    message = f"""{emoji} **{alert_title}**

🪙 **Token:** `{token}`
🌐 **Network:** {network_info['name']} ({network_info['symbol']})
📊 **Score:** {score:.1f}
💰 **ETH Volume:** {eth_value:.4f}
👥 **Wallets:** {wallet_count}
🔄 **Transactions:** {tx_count}
🎯 **Confidence:** {confidence}

📋 **Contract Address:**
{format_contract_address(contract_address)}

🔗 **Quick Actions:**
{generate_action_links(token, contract_address, network)}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced Monitoring v3.0"""
    
    return message.strip()

def format_alert_message(alert: dict) -> str:
    """Enhanced format alert for Telegram with contract address and links"""
    return format_enhanced_alert_message(alert)

async def send_alert_notifications(alerts: list):
    """Send enhanced alert notifications via Telegram"""
    if not alerts:
        return
    
    async with telegram_client:
        for alert in alerts:
            try:
                message = format_enhanced_alert_message(alert)
                success = await telegram_client.send_message(message)
                if success:
                    token = alert.get('token', 'unknown')
                    logger.info(f"Enhanced alert sent for {token}")
                else:
                    logger.error(f"Failed to send enhanced alert for {alert.get('token', 'unknown')}")
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
            
            # Rate limiting - wait between messages
            await asyncio.sleep(1.5)

async def send_test_notification():
    """Send enhanced test notification"""
    test_message = f"""🧪 **ENHANCED TEST NOTIFICATION**

✅ Crypto Monitor v3.0 is working!
🕐 {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Features enabled:
  • Contract Address display
  • Quick action links
  • Enhanced formatting
  • Multi-network support

🔗 **Test Links:**
[🦄 Uniswap](https://app.uniswap.org) | [📊 DexScreener](https://dexscreener.com) | [🐦 Twitter](https://twitter.com)

📋 **Sample CA:** `0x1234567890abcdef1234567890abcdef12345678`"""
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

def check_notification_config():
    """Check if Telegram is properly configured using existing function"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return bool(bot_token and chat_id and len(bot_token) > 40)

# ============================================================================
# ENHANCED TELEGRAM SERVICE - Updated for v3.0
# ============================================================================

class TelegramService:
    """Enhanced Telegram service v3.0 with contract addresses and action links"""
    
    def __init__(self):
        pass
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return check_notification_config()
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message using telegram client"""
        if not self.is_configured():
            logger.warning("Telegram not configured - skipping notification")
            return False
        
        try:
            async with telegram_client:
                return await telegram_client.send_message(message, parse_mode)
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
                    "ready_for_notifications": chat_accessible,
                    "enhanced_features": True,
                    "version": "3.0"
                }
                
        except Exception as e:
            return {
                "configured": False,
                "error": f"Connection test failed: {str(e)}",
                "exception": True
            }
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, min_alpha_score: float = 50.0):
        """Send enhanced notifications for analysis results with contract addresses"""
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
            network_info = get_network_info(network)
            
            if limited_tokens:
                # Send individual enhanced notifications
                for i, token_tuple in enumerate(limited_tokens):
                    token = token_tuple[0]
                    data = token_tuple[1] 
                    score = token_tuple[2]
                    
                    # Extract contract address
                    contract_address = data.get('contract_address', '')
                    if not contract_address:
                        contract_address = data.get('ca', '')
                        if not contract_address and 'web3_data' in data:
                            contract_address = data['web3_data'].get('contract_address', '')
                    
                    if result.analysis_type == "buy":
                        emoji = "🟢"
                        alert_title = "BUY ALERT"
                        eth_value = data.get('total_eth_spent', 0)
                        volume_label = "ETH Spent"
                        tx_label = "Purchases"
                        tx_count = data.get('total_purchases', 0)
                    else:  # sell
                        emoji = "🔴"
                        alert_title = "SELL ALERT"
                        eth_value = data.get('total_eth_received', 0)
                        volume_label = "ETH Received"
                        tx_label = "Sells"
                        tx_count = data.get('total_sells', 0)
                    
                    message = f"""{emoji} **{alert_title}**

🪙 **Token:** `{token}`
🌐 **Network:** {network_info['name']} ({network_info['symbol']})
📊 **Score:** {score:.1f}
💰 **{volume_label}:** {eth_value:.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🔄 **{tx_label}:** {tx_count}
⭐ **Avg Wallet Score:** {data.get('avg_wallet_score', 0):.1f}

📋 **Contract Address:**
{format_contract_address(contract_address)}

🔗 **Quick Actions:**
{generate_action_links(token, contract_address, network)}

🏆 **Rank:** #{i+1} of {len(limited_tokens)}
⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced v3.0"""
                    
                    await self.send_message(message.strip())
                    await asyncio.sleep(2)  # Rate limiting
                
                # Send enhanced summary with storage info
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

🌐 **Network:** {network_info['name']} ({network_info['symbol']})
🚀 **Enhanced Features:** Contract addresses, action links
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
                
                await self.send_message(summary_message.strip())
                logger.info(f"Sent {len(limited_tokens)} enhanced notifications for {network}")
                
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

🌐 **Network:** {network_info['name']} ({network_info['symbol']})
📊 **Tokens Found:** {result.unique_tokens}
🚫 **Above {min_alpha_score} Score:** 0
📈 **Highest Score:** {max_score:.1f}{storage_info}

💡 **Tip:** Lower min_alpha_score to see more alerts
⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced v3.0"""
                
                await self.send_message(message.strip())
                logger.info(f"No qualifying tokens found for {network} (max score: {max_score:.1f})")
                
        except Exception as e:
            logger.error(f"Failed to send enhanced analysis notifications: {e}")
            await self.send_message(f"❌ **Notification Error** - Analysis completed but failed to send enhanced alerts: {str(e)}")
    
    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int, 
                                    days_back: float, use_smart_timing: bool, max_tokens: int, 
                                    min_alpha_score: float, store_data: bool = False):
        """Send enhanced analysis start notification"""
        timing_info = f"⏰ Smart: {days_back}d" if use_smart_timing else f"⏰ Manual: {days_back}d"
        storage_info = f"🗄️ Storage: {'Enabled' if store_data else 'Disabled'}"
        network_info = get_network_info(network)
        
        start_message = f"""🚀 **ENHANCED ANALYSIS STARTED v3.0**

**Network:** {network_info['name']} ({network_info['symbol']})
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
{timing_info}
{storage_info}
**AI Enhancement:** Enabled
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score

🚀 **Enhanced Features:**
• Contract addresses for easy copying
• Direct Uniswap trading links
• DexScreener charts
• Twitter/X search links
• Block explorer links

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced Monitoring v3.0"""
        
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
    'format_alert_message',
    'format_enhanced_alert_message',
    'get_network_info',
    'format_contract_address',
    'generate_action_links'
]