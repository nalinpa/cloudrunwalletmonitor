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
                "User-Agent": "CryptoMonitor/3.2",
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
    """Format enhanced alert with Web3 intelligence and risk signals"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    token = alert.get('token', 'UNKNOWN')
    network = alert.get('network', 'ethereum')
    confidence = alert.get('confidence', 'Unknown')
    
    # Get contract address and Web3 data
    contract_address = data.get('contract_address', '')
    if not contract_address:
        contract_address = data.get('ca', '')
        if not contract_address and 'web3_data' in data:
            contract_address = data['web3_data'].get('contract_address', '')
    
    # Extract Web3 intelligence from AI analysis
    web3_data = data.get('web3_data', {})
    ai_data = alert.get('ai_data', {}) if len(alert) > 3 else {}
    
    # Get network info
    network_info = get_network_info(network)
    
    # Build alert header
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
    
    # Build Web3 intelligence section
    web3_signals = []
    risk_signals = []
    
    # Extract Web3 signals from AI data
    if ai_data:
        # Verification status
        if ai_data.get('is_verified'):
            web3_signals.append("✅ Contract Verified")
        else:
            risk_signals.append("⚠️ Unverified Contract")
        
        # Liquidity signals
        if ai_data.get('has_liquidity'):
            web3_signals.append("💧 Has Liquidity")
        else:
            risk_signals.append("🚨 No Liquidity Detected")
        
        # Smart money signals  
        if ai_data.get('has_smart_money'):
            web3_signals.append("🧠 Smart Money Active")
        
        # Honeypot risk
        honeypot_risk = ai_data.get('honeypot_risk', 0)
        if honeypot_risk > 0.5:
            risk_signals.append(f"🍯 Honeypot Risk: {honeypot_risk:.0%}")
        elif honeypot_risk > 0.2:
            risk_signals.append(f"⚠️ Some Risk: {honeypot_risk:.0%}")
    
    # Extract additional Web3 data
    if web3_data:
        # Token age
        token_age_hours = web3_data.get('token_age_hours')
        if token_age_hours is not None:
            if token_age_hours < 1:
                risk_signals.append(f"🆕 Brand New (<1h)")
            elif token_age_hours < 24:
                web3_signals.append(f"🕐 {token_age_hours:.1f}h old")
            elif token_age_hours < 168:  # 1 week
                web3_signals.append(f"📅 {token_age_hours/24:.1f}d old")
        
        # Holder count
        holder_count = web3_data.get('holder_count')
        if holder_count is not None:
            if holder_count < 50:
                risk_signals.append(f"👥 Few Holders: {holder_count}")
            else:
                web3_signals.append(f"👥 {holder_count} Holders")
    
    # Extract AI analysis signals
    if ai_data and 'ai_analyses' in str(ai_data):
        # Whale coordination
        if ai_data.get('whale_coordination_detected'):
            web3_signals.append("🐋 Whale Coordination")
        
        # Pump signals
        if ai_data.get('pump_signals_detected'):
            web3_signals.append("🚀 Pump Signals")
        
        # Risk assessment
        risk_level = ai_data.get('risk_level', '').upper()
        if risk_level == 'HIGH':
            risk_signals.append("🚨 High Risk")
        elif risk_level == 'MEDIUM':
            risk_signals.append("⚠️ Medium Risk")
    
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
        f"🎯 **Confidence:** {confidence}"
    ]
    
    # Add Web3 intelligence section
    if web3_signals or risk_signals:
        message_parts.append("")
        message_parts.append("🔍 **Web3 Intelligence:**")
        
        # Positive signals
        if web3_signals:
            for signal in web3_signals[:4]:  # Limit to 4 signals
                message_parts.append(f"  {signal}")
        
        # Risk signals
        if risk_signals:
            for risk in risk_signals[:3]:  # Limit to 3 risks  
                message_parts.append(f"  {risk}")
    
    # Contract address section
    message_parts.extend([
        "",
        "📋 **Contract Address:**",
        format_contract_address(contract_address)
    ])
    
    # Action links
    message_parts.extend([
        "",
        "🔗 **Quick Actions:**",
        generate_action_links(token, contract_address, network)
    ])
    
    # Footer
    message_parts.extend([
        "",
        f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}",
        "🚀 Enhanced Web3 Monitoring v3.2"
    ])
    
    return "\n".join(message_parts)

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

✅ Crypto Monitor v3.2 with Momentum Tracking!
🕐 {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Features enabled:
  • Contract Address display
  • Quick action links
  • Enhanced formatting
  • Multi-network support
  • Web3 intelligence
  • Momentum tracking

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
# ENHANCED TELEGRAM SERVICE WITH MOMENTUM TRACKING - v3.2
# ============================================================================

class TelegramService:
    """Enhanced Telegram service v3.2 with momentum tracking and Web3 intelligence"""
    
    def __init__(self):
        self.momentum_tracker = None
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return check_notification_config()
    
    async def initialize_momentum_tracking(self, config):
        """Initialize momentum tracking"""
        try:
            from services.tracker.alert_momentum import AlertMomentumTracker
            self.momentum_tracker = AlertMomentumTracker(config)
            await self.momentum_tracker.initialize()
            logger.info("Alert momentum tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize momentum tracking: {e}")
            self.momentum_tracker = None
    
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
                    "momentum_tracking": bool(self.momentum_tracker),
                    "version": "3.2"
                }
                
        except Exception as e:
            return {
                "configured": False,
                "error": f"Connection test failed: {str(e)}",
                "exception": True
            }
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, min_alpha_score: float = 50.0):
        """Send notifications with fixed trending summary"""
        try:
            if not result.ranked_tokens:
                await self.send_message(f"📊 **Analysis Complete** - No tokens found for {network.upper()}")
                return
            
            # Filter tokens by score
            qualifying_tokens = []
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
                            'ai_data': ai_data
                        }
                        qualifying_tokens.append(alert)
            
            limited_tokens = qualifying_tokens[:max_tokens]
            
            if limited_tokens:
                # Send individual notifications with momentum context
                for alert in limited_tokens:
                    try:
                        # Get momentum data
                        momentum_data = {}
                        if self.momentum_tracker:
                            momentum_data = await self.momentum_tracker.get_token_momentum(
                                alert['token'], alert['network'], days_back=5
                            )
                        
                        alert['momentum_data'] = momentum_data
                        
                        # Send message with momentum
                        message = self._format_alert_with_momentum(alert)
                        await self.send_message(message)
                        
                        # Store alert for future momentum tracking
                        if self.momentum_tracker:
                            success = await self.momentum_tracker.store_alert(alert)
                            if not success:
                                logger.warning(f"Failed to store momentum alert for {alert['token']}")
                        
                        await asyncio.sleep(2)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Failed to send alert: {e}")
                
                # FIXED: Send trending summary with error handling
                if self.momentum_tracker:
                    try:
                        await self._send_trending_summary(network)
                        logger.info("Trending summary sent successfully")
                    except Exception as e:
                        logger.error(f"Failed to send trending summary: {e}")
                        logger.error(f"Trending summary error traceback: {traceback.format_exc()}")
                else:
                    logger.warning("No momentum tracker available for trending summary")
                
                # Send regular summary
                await self._send_analysis_summary(result, network, len(limited_tokens), min_alpha_score)
                
                logger.info(f"Sent {len(limited_tokens)} notifications for {network}")
                
            else:
                await self._send_no_alerts_message(result, network, min_alpha_score)
                
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
            import traceback
            logger.error(f"Notification error traceback: {traceback.format_exc()}")
            await self.send_message(f"❌ **Notification Error** - Analysis completed but failed to send alerts: {str(e)}")


    def _format_alert_with_momentum(self, alert: dict) -> str:
        """Format alert with momentum context showing net scores"""
        # Start with the existing alert format
        message = format_enhanced_alert_message(alert)
        
        # Add momentum context if available
        momentum_data = alert.get('momentum_data', {})
        if momentum_data and momentum_data.get('net_momentum_score') is not None:
            momentum_section = [
                "",
                "📊 **Momentum Analysis:**"
            ]
            
            net_score = momentum_data.get('net_momentum_score', 0)
            momentum_strength = momentum_data.get('momentum_strength', 'NEUTRAL')
            buy_momentum = momentum_data.get('buy_momentum', 0)
            sell_momentum = momentum_data.get('sell_momentum', 0)
            alert_count = momentum_data.get('alert_count', 0)
            
            # Net momentum display
            if net_score >= 50:
                momentum_section.append(f"  🚀 **Net Score: +{net_score:.1f}** (VERY BULLISH)")
            elif net_score >= 20:
                momentum_section.append(f"  📈 **Net Score: +{net_score:.1f}** (BULLISH)")
            elif net_score >= 5:
                momentum_section.append(f"  ⬆️ **Net Score: +{net_score:.1f}** (SLIGHT BUY)")
            elif net_score <= -50:
                momentum_section.append(f"  📉 **Net Score: {net_score:.1f}** (STRONG SELL PRESSURE)")
            elif net_score <= -20:
                momentum_section.append(f"  ⬇️ **Net Score: {net_score:.1f}** (BEARISH)")
            elif net_score <= -5:
                momentum_section.append(f"  📉 **Net Score: {net_score:.1f}** (SLIGHT SELL)")
            else:
                momentum_section.append(f"  ➡️ **Net Score: {net_score:.1f}** (NEUTRAL)")
            
            # Breakdown
            if alert_count > 1:
                momentum_section.append(f"  📊 {alert_count} alerts over 5 days")
                if buy_momentum > 0:
                    momentum_section.append(f"  💚 Buy Momentum: +{buy_momentum:.1f}")
                if sell_momentum > 0:
                    momentum_section.append(f"  💔 Sell Pressure: -{sell_momentum:.1f}")
            
            # Activity indicators
            velocity = momentum_data.get('momentum_velocity', 0)
            if velocity > 0.5:
                momentum_section.append(f"  ⚡ High Activity: {velocity:.0%} recent")
            
            # Special signals
            if momentum_data.get('whale_activity'):
                momentum_section.append("  🐋 Whale activity detected")
            if momentum_data.get('pump_activity'):
                momentum_section.append("  🚀 Pump signals active")
            if momentum_data.get('smart_money_activity'):
                momentum_section.append("  🧠 Smart money involved")
            
            # Trending direction
            direction = momentum_data.get('trending_direction', 'SIDEWAYS')
            if direction == 'UP':
                momentum_section.append("  📈 **Trending: UPWARD**")
            elif direction == 'DOWN':
                momentum_section.append("  📉 **Trending: DOWNWARD**")
            
            # Insert momentum section before contract address
            lines = message.split('\n')
            contract_index = next((i for i, line in enumerate(lines) if '**Contract Address:**' in line), len(lines))
            lines[contract_index:contract_index] = momentum_section
            message = '\n'.join(lines)
        else:
            # If no momentum data, add first alert note
            lines = message.split('\n')
            contract_index = next((i for i, line in enumerate(lines) if '**Contract Address:**' in line), len(lines))
            lines[contract_index:contract_index] = ["", "📊 **First Alert** - Building momentum data..."]
            message = '\n'.join(lines)
        
        return message

    async def _send_trending_summary(self, network: str):
        """Send trending summary with enhanced error handling"""
        try:
            if not self.momentum_tracker:
                logger.warning("No momentum tracker available for trending summary")
                return
            
            logger.info(f"Getting trending tokens for {network}")
            trending = await self.momentum_tracker.get_trending_tokens(
                network=network, hours_back=24, limit=5
            )
            
            if not trending:
                logger.info("No trending tokens found")
                return
            
            logger.info(f"Found {len(trending)} trending tokens")
            
            trending_lines = ["🔥 **24H MOMENTUM RANKING**", ""]
            
            for i, token_data in enumerate(trending[:3]):  # Top 3
                net_score = token_data.get('net_momentum_score', 0)
                momentum_indicator = token_data.get('momentum_indicator', 'NEUTRAL')
                buy_momentum = token_data.get('buy_momentum', 0)
                sell_momentum = token_data.get('sell_momentum', 0)
                
                trending_lines.append(
                    f"{i+1}. **{token_data['token_symbol']}** {momentum_indicator}"
                )
                trending_lines.append(
                    f"   Net: {net_score:+.1f} (💚{buy_momentum:+.1f} 💔-{sell_momentum:.1f})"
                )
                
                volume = token_data.get('total_volume', 0)
                if volume > 1.0:
                    trending_lines.append(f"   Volume: {volume:.2f} ETH")
                
                trending_lines.append("")  # Spacing
            
            trending_lines.extend([
                "📈 **Legend:**",
                "💚 = Buy momentum, 💔 = Sell pressure",
                "Net = Combined score (Buys - Sells)",
                "",
                f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}"
            ])
            
            message = '\n'.join(trending_lines)
            logger.info(f"Sending trending summary: {len(message)} characters")
            
            success = await self.send_message(message)
            if success:
                logger.info("Trending summary sent successfully")
            else:
                logger.error("Failed to send trending summary message")
                
        except Exception as e:
            logger.error(f"Trending summary failed: {e}")
            import traceback
            logger.error(f"Trending summary traceback: {traceback.format_exc()}")
           
    async def _send_analysis_summary(self, result, network: str, alerts_sent: int, min_alpha_score: float):
        """Send analysis summary"""
        network_info = get_network_info(network)
        storage_info = ""
        if hasattr(result, 'performance_metrics'):
            transfers_stored = result.performance_metrics.get('transfers_stored', 0)
            if transfers_stored > 0:
                storage_info = f"\n🗄️ **Stored:** {transfers_stored} records"
            else:
                storage_info = f"\n🗄️ **Storage:** Disabled"
        
        summary_message = f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

✅ **Momentum-Enhanced Alerts:** {alerts_sent}
📈 **Total Tokens Found:** {result.unique_tokens}
💰 **Total ETH Volume:** {result.total_eth_value:.4f}
🔍 **Filter:** min score {min_alpha_score}{storage_info}

🌐 **Network:** {network_info['name']} ({network_info['symbol']})
🚀 **Features:** Web3 intel, momentum tracking, trending analysis
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        
        await self.send_message(summary_message.strip())
    
    async def _send_no_alerts_message(self, result, network: str, min_alpha_score: float):
        """Send message when no alerts qualify"""
        network_info = get_network_info(network)
        max_score = max([t[2] for t in result.ranked_tokens[:3]]) if result.ranked_tokens else 0
        
        message = f"""📊 **{result.analysis_type.upper()} ANALYSIS - NO ALERTS**

🌐 **Network:** {network_info['name']} ({network_info['symbol']})
📊 **Tokens Found:** {result.unique_tokens}
🚫 **Above {min_alpha_score} Score:** 0
📈 **Highest Score:** {max_score:.1f}

💡 **Tip:** Lower min_alpha_score for more alerts
⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Momentum-Enhanced v3.2"""
        
        await self.send_message(message.strip())

    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int, 
                                    days_back: float, use_smart_timing: bool, max_tokens: int, 
                                    min_alpha_score: float, store_data: bool = False):
        """Send enhanced analysis start notification"""
        timing_info = f"⏰ Smart: {days_back}d" if use_smart_timing else f"⏰ Manual: {days_back}d"
        storage_info = f"🗄️ Storage: {'Enabled' if store_data else 'Disabled'}"
        network_info = get_network_info(network)
        
        start_message = f"""🚀 **ENHANCED ANALYSIS STARTED v3.2**

**Network:** {network_info['name']} ({network_info['symbol']})
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
{timing_info}
{storage_info}
**AI Enhancement:** Enabled
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score

🚀 **Enhanced Features:**
• Contract addresses with verification status
• Web3 intelligence and risk analysis
• Momentum tracking across 5 days
• Trending token detection
• Direct Uniswap trading links
• DexScreener charts and Twitter search

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced Web3 Monitoring v3.2"""
        
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