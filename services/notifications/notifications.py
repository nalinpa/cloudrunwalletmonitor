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

class DailyPrimeTimeSystem:
    """Single daily prime time window for buy and sell analysis"""
    
    def __init__(self):
        # Single daily prime time windows (NZDT timezone)
        self.daily_prime_times = {
            'buy': {
                'prime_hour': 20,        # 9:00 AM NZDT - Absolute lowest prices (Asian morning lull)
                'prime_minute': 0,      # Exact time: 9:00 AM
                'window_minutes': 45,   # ±45 minutes around prime time (8:15 AM - 9:45 AM)
                'description': 'DAILY PEAK BUY WINDOW - Lowest global prices'
            },
            'sell': {
                'prime_hour': 5,       # 6:00 PM NZDT - Peak liquidity (Asia close + Europe active)
                'prime_minute': 0,      # Exact time: 6:00 PM  
                'window_minutes': 45,   # ±45 minutes around prime time (5:15 PM - 6:45 PM)
                'description': 'DAILY PEAK SELL WINDOW - Highest global liquidity'
            }
        }
    
    def is_daily_prime_time(self, analysis_type: str, current_time: datetime = None) -> Dict[str, Any]:
        """Check if current time is within the daily prime trading window"""
        
        if current_time is None:
            current_time = datetime.now()
        
        prime_config = self.daily_prime_times.get(analysis_type, {})
        prime_hour = prime_config.get('prime_hour')
        prime_minute = prime_config.get('prime_minute', 0)
        window_minutes = prime_config.get('window_minutes', 45)
        description = prime_config.get('description', 'Prime time')
        
        if not prime_hour:
            return {'is_prime_time': False, 'priority_level': 'MEDIUM'}
        
        # Calculate current time in minutes since midnight
        current_total_minutes = current_time.hour * 60 + current_time.minute
        prime_total_minutes = prime_hour * 60 + prime_minute
        
        # Calculate difference in minutes
        minutes_diff = current_total_minutes - prime_total_minutes
        
        # Check if within prime window
        if abs(minutes_diff) <= window_minutes:
            # Determine priority level based on proximity to exact prime time
            if abs(minutes_diff) <= 15:  # Within 15 minutes of exact prime time
                priority_level = 'CRITICAL'
                window_type = 'PEAK'
            elif abs(minutes_diff) <= 30:  # Within 30 minutes
                priority_level = 'HIGH'
                window_type = 'PRIME'
            else:  # Within 45 minutes
                priority_level = 'HIGH'
                window_type = 'EXTENDED'
            
            return {
                'is_prime_time': True,
                'priority_level': priority_level,
                'window_type': window_type,
                'minutes_from_peak': minutes_diff,
                'prime_hour': prime_hour,
                'prime_minute': prime_minute,
                'window_minutes': window_minutes,
                'description': description,
                'prime_reason': f"DAILY {window_type} {analysis_type.upper()} WINDOW",
                'exact_prime_time': f"{prime_hour:02d}:{prime_minute:02d}",
                'window_start': self._format_time(prime_hour, prime_minute - window_minutes),
                'window_end': self._format_time(prime_hour, prime_minute + window_minutes)
            }
        
        else:
            # Not in prime time - calculate next occurrence
            if minutes_diff < -window_minutes:
                # Prime time is later today
                minutes_until = -minutes_diff - window_minutes
                next_occurrence = "today"
            else:
                # Prime time is tomorrow
                minutes_until = (24 * 60) - minutes_diff + window_minutes
                next_occurrence = "tomorrow"
            
            return {
                'is_prime_time': False,
                'priority_level': 'MEDIUM',
                'minutes_until_prime': minutes_until,
                'next_prime_time': f"{prime_hour:02d}:{prime_minute:02d}",
                'next_occurrence': next_occurrence,
                'description': description,
                'prime_reason': 'NON-PRIME WINDOW'
            }
    
    def _format_time(self, hour: int, minute: int) -> str:
        """Format time handling overflow/underflow"""
        # Handle minute overflow/underflow
        while minute >= 60:
            minute -= 60
            hour += 1
        while minute < 0:
            minute += 60
            hour -= 1
        
        # Handle hour overflow/underflow
        hour = hour % 24
        
        return f"{hour:02d}:{minute:02d}"
    
    def enhance_alerts_for_daily_prime_time(self, alerts: list, analysis_type: str) -> list:
        """Enhance alerts with daily prime time priority"""
        
        prime_time_info = self.is_daily_prime_time(analysis_type)
        
        if prime_time_info['is_prime_time']:
            logger.info(f"🕐 DAILY PRIME TIME ACTIVE: {prime_time_info['prime_reason']}")
            logger.info(f"🎯 Window: {prime_time_info['window_start']} - {prime_time_info['window_end']}")
            logger.info(f"⏰ Peak time: {prime_time_info['exact_prime_time']} (offset: {prime_time_info['minutes_from_peak']:+d} min)")
        
        enhanced_alerts = []
        for alert in alerts:
            # Copy alert and add prime time information
            enhanced_alert = alert.copy()
            enhanced_alert['prime_time_info'] = prime_time_info
            
            if prime_time_info['is_prime_time']:
                # PRIME TIME ENHANCEMENTS - NO SCORE BONUS
                enhanced_alert['priority_level'] = prime_time_info['priority_level']
                enhanced_alert['is_daily_prime_time'] = True
                enhanced_alert['window_type'] = prime_time_info['window_type']
                
                # Keep original score unchanged
                enhanced_alert['final_score'] = enhanced_alert.get('enhanced_score', enhanced_alert.get('score', 0))
                
                logger.info(f"🚨 DAILY PRIME ALERT: {alert['token']} - {prime_time_info['window_type']} WINDOW")
                
            else:
                # NON-PRIME TIME
                enhanced_alert['priority_level'] = 'MEDIUM'
                enhanced_alert['is_daily_prime_time'] = False
                enhanced_alert['final_score'] = enhanced_alert.get('enhanced_score', enhanced_alert.get('score', 0))
                
                # Log next prime time
                logger.info(f"📊 Non-prime alert: {alert['token']} (next prime: {prime_time_info.get('next_prime_time', 'N/A')} {prime_time_info.get('next_occurrence', '')})")
            
            enhanced_alerts.append(enhanced_alert)
        
        return enhanced_alerts

    def format_daily_prime_time_message(self, alert: Dict) -> str:
        """Format alert message with daily prime time enhancements"""
        
        prime_info = alert.get('prime_time_info', {})
        
        if not prime_info.get('is_prime_time'):
            # Regular formatting for non-prime time
            return self._format_regular_alert(alert)
        
        # PRIME TIME FORMATTING
        data = alert.get('data', {})
        token = alert.get('token', 'UNKNOWN')
        network = alert.get('network', 'ethereum')
        analysis_type = alert.get('alert_type', 'unknown')
        window_type = alert.get('window_type', 'PRIME')
        
        # Priority-specific headers
        if prime_info['priority_level'] == 'CRITICAL':
            header = "🚨🕐🚨 **DAILY PEAK TIME ALERT** 🚨🕐🚨"
            intensity = "PEAK"
        elif window_type == 'PRIME':
            header = "🔥🕐 **DAILY PRIME TIME ALERT** 🕐🔥"
            intensity = "PRIME"
        else:
            header = "⭐🕐 **DAILY PRIME WINDOW** 🕐⭐"
            intensity = "EXTENDED"
        
        # Analysis-specific descriptions
        if analysis_type == 'buy':
            strategy_desc = "🎯 **OPTIMAL ACCUMULATION WINDOW**"
            timing_desc = "💰 Lowest global prices - Peak buying opportunity"
        else:
            strategy_desc = "🎯 **OPTIMAL DISTRIBUTION WINDOW**"  
            timing_desc = "💎 Highest global liquidity - Peak selling opportunity"
        
        # Core metrics
        final_score = alert.get('final_score', 0)
        eth_value = data.get('total_eth_spent', data.get('total_eth_received', 0))
        wallet_count = data.get('wallet_count', 0)
        
        # Time information
        exact_prime = prime_info.get('exact_prime_time', 'N/A')
        minutes_offset = prime_info.get('minutes_from_peak', 0)
        window_start = prime_info.get('window_start', 'N/A')
        window_end = prime_info.get('window_end', 'N/A')
        
        message = f"""{header}

{strategy_desc}
{timing_desc}

🪙 **Token:** `{token}`
🌐 **Network:** {self._get_network_display(network)}
⚡ **SCORE:** {final_score:.1f}
💰 **ETH Volume:** {eth_value:.4f}
👥 **Wallets:** {wallet_count}

🕐 **DAILY PRIME TIME INFO:**
  🎯 **Peak Time:** {exact_prime} NZDT
  📊 **Current Offset:** {minutes_offset:+d} minutes from peak
  ⏰ **Prime Window:** {window_start} - {window_end}
  🔥 **Intensity:** {intensity} WINDOW

💡 **Why This Matters:**
{prime_info.get('description', 'Optimal trading window')}"""

        # Add contract address and priority actions
        contract_address = data.get('contract_address', data.get('ca', ''))
        if contract_address:
            message += f"""

📋 **Contract Address:**
`{contract_address}`

🚨 **PRIORITY ACTIONS:**
{self._get_priority_action_links(token, contract_address, network, analysis_type)}"""
        
        message += f"""

🚨🚨🚨 **DAILY PRIME TIME - ACT NOW** 🚨🚨🚨
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        
        return message
    
    def _format_regular_alert(self, alert: Dict) -> str:
        """Standard formatting for non-prime time alerts"""
        # Your existing alert formatting
        data = alert.get('data', {})
        token = alert.get('token', 'UNKNOWN')
        
        return f"""📊 **Standard Alert**

🪙 **Token:** `{token}`
📊 **Score:** {alert.get('final_score', 0):.1f}
💰 **ETH:** {data.get('total_eth_spent', data.get('total_eth_received', 0)):.4f}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
    
    def _get_network_display(self, network: str) -> str:
        """Enhanced network display for prime time"""
        displays = {
            'ethereum': '🔷 Ethereum (ETH)',
            'base': '🔵 Base (ETH)', 
            'arbitrum': '🔺 Arbitrum (ETH)'
        }
        return displays.get(network.lower(), f"🌐 {network.upper()}")
    
    def _get_priority_action_links(self, token: str, contract_address: str, network: str, analysis_type: str) -> str:
        """Priority action links for prime time"""
        from .notifications import generate_action_links
        
        base_links = generate_action_links(token, contract_address, network)
        
        if analysis_type == 'buy':
            priority_note = "🚨 **BUY NOW** - Optimal entry prices"
        else:
            priority_note = "🚨 **SELL NOW** - Peak liquidity window"
        
        return f"{base_links}\n\n{priority_note}"
    
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
    """Enhanced contract address formatting with validation and short version"""
    if not contract_address or len(contract_address) < 10:
        return "❓ No CA"
    
    # Clean the address
    clean_address = contract_address.strip().lower()
    if not clean_address.startswith('0x'):
        clean_address = '0x' + clean_address
    
    # Validate length
    if len(clean_address) != 42:
        return "❓ Invalid CA"
    
    # Format for display with both full and short versions
    short_ca = f"{clean_address[:6]}...{clean_address[-4:]}"
    return f"`{clean_address}`\n💾 Short: `{short_ca}`"

def generate_action_links(token: str, contract_address: str, network: str) -> str:
    """Enhanced action links with DEXTools and fixed Twitter search"""
    if not contract_address or len(contract_address) < 10:
        return "❌ No contract address available"
    
    network_info = get_network_info(network)
    
    # Clean contract address
    clean_ca = contract_address.strip().lower()
    if not clean_ca.startswith('0x'):
        clean_ca = '0x' + clean_ca
    
    # Validate address
    if len(clean_ca) != 42:
        return "❌ Invalid contract address"
    
    # Generate enhanced links
    links = []
    
    # Primary DEX (Uniswap)
    uniswap_url = f"{network_info['uniswap_base']}{clean_ca}"
    links.append(f"[🦄 Uniswap]({uniswap_url})")
    
    # Chart analysis
    dexscreener_url = f"{network_info['dexscreener_base']}{clean_ca}"
    links.append(f"[📊 Chart]({dexscreener_url})")
    
    # Block explorer
    explorer_url = f"https://{network_info['explorer']}/token/{clean_ca}"
    links.append(f"[🔍 Explorer]({explorer_url})")
    
    # FIXED: Twitter/X search with full contract address only
    twitter_search = f"https://twitter.com/search?q={clean_ca}"
    links.append(f"[🐦 Search X]({twitter_search})")
    
    # DEXTools for advanced analysis
    dextools_url = f"https://www.dextools.io/app/en/{network}/pair-explorer/{clean_ca}"
    links.append(f"[🔧 DEXTools]({dextools_url})")
    
    return " | ".join(links)

def get_network_info(network: str) -> Dict[str, str]:
    """Enhanced network information with emoji and better DEX support"""
    network_configs = {
        'ethereum': {
            'name': 'Ethereum',
            'symbol': 'ETH',
            'explorer': 'etherscan.io',
            'uniswap_base': 'https://app.uniswap.org/#/swap?outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/ethereum/',
            'chain_id': '1',
            'emoji': '🔷'
        },
        'base': {
            'name': 'Base',
            'symbol': 'ETH',
            'explorer': 'basescan.org',
            'uniswap_base': 'https://app.uniswap.org/#/swap?chain=base&outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/base/',
            'chain_id': '8453',
            'emoji': '🔵'
        }
    }
    
    return network_configs.get(network.lower(), network_configs['ethereum'])

def format_enhanced_alert_message(alert: dict) -> str:
    """Format enhanced alert with PROMINENT token age and holder count"""
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

    # ENHANCED: Extract token age and holder count for prominence
    token_age_hours = None
    holder_count = None
    
    # Try multiple sources for token age and holder data
    if ai_data:
        token_age_hours = ai_data.get('token_age_hours')
        holder_count = ai_data.get('holder_count')
    
    if web3_data and (token_age_hours is None or holder_count is None):
        if token_age_hours is None:
            token_age_hours = web3_data.get('token_age_hours')
        if holder_count is None:
            holder_count = web3_data.get('holder_count')

    # ENHANCED: Format age and holder information with risk indicators
    def format_token_age(hours):
        if hours is None:
            return "🕐 Age: ❓ Unknown"
        
        try:
            hours = float(hours)
            if hours < 1:
                return f"🕐 Age: 🆕 {hours*60:.0f}min (BRAND NEW)"
            elif hours < 24:
                return f"🕐 Age: ⚡ {hours:.1f}h (FRESH)"
            elif hours < 168:  # 1 week
                days = hours / 24
                return f"🕐 Age: 📅 {days:.1f}d (RECENT)"
            elif hours < 720:  # 1 month
                days = hours / 24
                return f"🕐 Age: ✅ {days:.1f}d (ESTABLISHED)"
            else:
                days = hours / 24
                return f"🕐 Age: 💎 {days:.0f}d (MATURE)"
        except (ValueError, TypeError):
            return "🕐 Age: ❓ Unknown"

    def format_holder_count(count):
        if count is None:
            return "👥 Holders: ❓ Unknown"
        
        try:
            count = int(count)
            if count < 50:
                return f"👥 Holders: 🚨 {count:,} (RISKY - LOW)"
            elif count < 200:
                return f"👥 Holders: ⚠️ {count:,} (CAUTION - LIMITED)"
            elif count < 1000:
                return f"👥 Holders: ✅ {count:,} (GOOD DISTRIBUTION)"
            elif count < 5000:
                return f"👥 Holders: 💚 {count:,} (STRONG BASE)"
            else:
                return f"👥 Holders: 💎 {count:,} (WIDE ADOPTION)"
        except (ValueError, TypeError):
            return "👥 Holders: ❓ Unknown"

    # Create age and holder display
    age_display = format_token_age(token_age_hours)
    holder_display = format_holder_count(holder_count)
    
    # ENHANCED: Generate combined risk assessment
    def get_combined_risk_indicator(age_hours, holders):
        risk_signals = []
        
        try:
            if age_hours is not None and float(age_hours) < 24:
                risk_signals.append("NEW TOKEN")
            if holders is not None and int(holders) < 50:
                risk_signals.append("FEW HOLDERS")
                
            if len(risk_signals) >= 2:
                return "🚨 HIGH RISK: " + " + ".join(risk_signals)
            elif len(risk_signals) == 1:
                return "⚠️ CAUTION: " + risk_signals[0]
            elif age_hours is not None and holders is not None:
                age_val = float(age_hours)
                holder_val = int(holders)
                if age_val > 720 and holder_val > 1000:
                    return "✅ LOW RISK: Established + Wide distribution"
                elif age_val > 168 and holder_val > 200:
                    return "💡 MODERATE RISK: Growing project"
        except (ValueError, TypeError):
            pass
        
        return None

    risk_indicator = get_combined_risk_indicator(token_age_hours, holder_count)

    # Build Web3 intelligence section (streamlined for space)
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
        liquidity_usd = ai_data.get('liquidity_usd', 0)
        if ai_data.get('has_liquidity'):
            if liquidity_usd > 100000:
                web3_signals.append(f"💧 High Liquidity (${liquidity_usd:,.0f})")
            elif liquidity_usd > 10000:
                web3_signals.append(f"💧 Good Liquidity (${liquidity_usd:,.0f})")
            else:
                web3_signals.append("💧 Has Liquidity")
        else:
            risk_signals.append("🚨 No Liquidity Detected")

    # ENHANCED: Build the complete message with PROMINENT age and holder info
    message_parts = [
        f"{emoji} **{alert_title}**",
        "",
        f"🪙 **Token:** `{token}`",
        f"🌐 **Network:** {network_info['name']} ({network_info['symbol']})",
        f"📊 **Score:** {score:.1f}",
        f"💰 **ETH Volume:** {eth_value:.4f}",
        f"👥 **Wallets:** {wallet_count}",
        f"🔄 **Transactions:** {tx_count}",
        "",
        # PROMINENT PLACEMENT: Age and holder count right after main metrics
        "🔍 **TOKEN FUNDAMENTALS:**",
        f"  {age_display}",
        f"  {holder_display}",
        "",
        f"🎯 **Confidence:** {confidence}"
    ]

    # Add combined risk assessment if available
    if risk_indicator:
        message_parts.extend(["", risk_indicator])

    # Add condensed Web3 intelligence section
    if web3_signals or risk_signals:
        message_parts.extend(["", "🔍 **Web3 Intelligence:**"])
        
        # Show most important signals first (limit to prevent overflow)
        for signal in web3_signals[:2]:  # Limit to 2 to save space
            message_parts.append(f"  {signal}")
        
        for risk in risk_signals[:1]:  # Limit to 1 risk signal
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
        "🚀 Enhanced Web3 Monitoring v4.0"
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
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, min_alpha_score: float = 50.0, store_alerts: bool = True):
        """Send notifications with optional alert storage"""
        try:
            if not result.ranked_tokens:
                message = f"📊 **{result.analysis_type.upper()} Analysis Complete** - No tokens found for {network.upper()}"
                await self.send_message(message)
                return
            
            # Filter tokens by score and convert to alerts
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
                            'ai_data': ai_data,
                            'score': score,
                            'enhanced_score': score
                        }
                        qualifying_tokens.append(alert)
            
            if not qualifying_tokens:
                await self._send_no_alerts_message(result, network, min_alpha_score)
                return
            
            # Add momentum data and store alerts (if enabled)
            for alert in qualifying_tokens:
                try:
                    # Get momentum data
                    momentum_data = {}
                    if self.momentum_tracker:
                        momentum_data = await self.momentum_tracker.get_token_momentum(
                            alert['token'], alert['network'], days_back=5
                        )
                    alert['momentum_data'] = momentum_data
                    
                    # Store alert for future momentum tracking (ONLY if store_alerts=True)
                    if store_alerts and self.momentum_tracker:
                        success = await self.momentum_tracker.store_alert(alert)
                        if success:
                            logger.info(f"✅ Stored alert for momentum tracking: {alert['token']}")
                        else:
                            logger.warning(f"⚠️ Failed to store momentum alert for {alert['token']}")
                    elif not store_alerts:
                        logger.info(f"📊 Skipping alert storage for {alert['token']} (store_alerts=False)")
                            
                except Exception as e:
                    logger.error(f"Error processing momentum for {alert['token']}: {e}")
                    alert['momentum_data'] = {}
            
            # Sort by momentum direction
            enhanced_alerts = self._sort_alerts_by_momentum_direction(qualifying_tokens, result.analysis_type)
            
            # Limit to max tokens
            limited_alerts = enhanced_alerts[:max_tokens]
            
            if limited_alerts:
                # Send individual notifications (same as before)
                for alert in limited_alerts:
                    try:
                        message = self._format_alert_with_momentum(alert)
                        await self.send_message(message)
                        await asyncio.sleep(2)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Failed to send alert: {e}")
                
                # Send trending summary (same as before)
                if self.momentum_tracker:
                    try:
                        await self._send_trending_summary(network, result.analysis_type)
                        logger.info("Trending summary sent successfully")
                    except Exception as e:
                        logger.error(f"Failed to send trending summary: {e}")
                
                # Send analysis summary with storage info
                await self._send_analysis_summary(result, network, len(limited_alerts), min_alpha_score, store_alerts)

                alerts_stored_msg = f"stored {len(limited_alerts)} alerts, " if store_alerts else "no alert storage, "
                logger.info(f"Sent {len(limited_alerts)} notifications for {network} - {alerts_stored_msg}momentum tracking {'enabled' if self.momentum_tracker else 'disabled'}")
                
            else:
                await self._send_no_alerts_message(result, network, min_alpha_score)
                
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
            import traceback
            logger.error(f"Notification error traceback: {traceback.format_exc()}")
            await self.send_message(f"❌ **Notification Error** - Analysis completed but failed to send alerts: {str(e)}")

    def _sort_alerts_by_momentum_direction(self, alerts: List[Dict], analysis_type: str) -> List[Dict]:
        """Sort alerts by momentum score based on analysis type"""
        
        if not alerts:
            return alerts
        
        # Extract momentum scores and sort appropriately
        alerts_with_momentum = []
        
        for alert in alerts:
            # Get momentum score from momentum_data if available
            momentum_data = alert.get('momentum_data', {})
            momentum_score = momentum_data.get('net_momentum_score', 0)
            
            # If no momentum data, use a neutral score
            if momentum_score == 0:
                momentum_score = 0
            
            alert['momentum_score'] = momentum_score
            alerts_with_momentum.append(alert)
        
        # Sort based on analysis type
        if analysis_type == 'buy':
            # BUY: Sort by HIGHEST positive momentum first (descending order)
            sorted_alerts = sorted(alerts_with_momentum, 
                                key=lambda x: x['momentum_score'], 
                                reverse=True)  # Highest first
            logger.info(f"📈 BUY alerts sorted by highest positive momentum")
            
        elif analysis_type == 'sell':
            # SELL: Sort by LOWEST negative momentum first (ascending order) 
            sorted_alerts = sorted(alerts_with_momentum, 
                                key=lambda x: x['momentum_score'], 
                                reverse=False)  # Lowest first (most negative)
            logger.info(f"📉 SELL alerts sorted by lowest negative momentum")
            
        else:
            # Default: no special sorting
            sorted_alerts = alerts_with_momentum
        
        # Log the momentum scores for debugging
        for i, alert in enumerate(sorted_alerts[:3]):  # Show top 3
            score = alert['momentum_score']
            token = alert.get('token', 'Unknown')
            logger.info(f"  {i+1}. {token}: momentum {score:+.1f}")
        
        return sorted_alerts

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

    async def _send_trending_summary(self, network: str, analysis_type: str = None):
        """Send trending summary with momentum sorting based on analysis type"""
        try:
            if not self.momentum_tracker:
                logger.warning("No momentum tracker available for trending summary")
                return
            
            logger.info(f"Getting trending tokens for {network} - {analysis_type} analysis")
            trending = await self.momentum_tracker.get_trending_tokens(
                network=network, hours_back=24, limit=10  # Get more to sort from
            )
            
            if not trending:
                logger.info("No trending tokens found")
                return
            
            logger.info(f"Found {len(trending)} trending tokens")
            
            # SORT BASED ON ANALYSIS TYPE
            if analysis_type == 'buy':
                # BUY: Sort by HIGHEST positive momentum (descending)
                sorted_trending = sorted(trending, 
                                    key=lambda x: x.get('net_momentum_score', 0), 
                                    reverse=True)
                title = "🔥 **24H BULLISH MOMENTUM RANKING**"
                subtitle = "Showing highest positive momentum for buy opportunities"
                
            elif analysis_type == 'sell':
                # SELL: Sort by LOWEST negative momentum (ascending - most negative first)
                sorted_trending = sorted(trending, 
                                    key=lambda x: x.get('net_momentum_score', 0), 
                                    reverse=False)
                title = "🔥 **24H BEARISH MOMENTUM RANKING**"
                subtitle = "Showing strongest negative momentum for sell opportunities"
                
            else:
                # DEFAULT: Sort by absolute momentum strength
                sorted_trending = sorted(trending, 
                                    key=lambda x: abs(x.get('net_momentum_score', 0)), 
                                    reverse=True)
                title = "🔥 **24H MOMENTUM RANKING**"
                subtitle = "Showing strongest momentum in either direction"
            
            trending_lines = [title, subtitle, ""]
            
            # Show top 7 tokens (increased from 3)
            for i, token_data in enumerate(sorted_trending[:7]):
                net_score = token_data.get('net_momentum_score', 0)
                momentum_indicator = token_data.get('momentum_indicator', 'NEUTRAL')
                buy_momentum = token_data.get('buy_momentum', 0)
                sell_momentum = token_data.get('sell_momentum', 0)
                
                # Enhanced momentum display based on analysis type
                if analysis_type == 'buy':
                    if net_score >= 50:
                        rank_emoji = "🚀"
                    elif net_score >= 20:
                        rank_emoji = "📈"
                    elif net_score > 0:
                        rank_emoji = "⬆️"
                    else:
                        rank_emoji = "➡️"
                elif analysis_type == 'sell':
                    if net_score <= -50:
                        rank_emoji = "📉"
                    elif net_score <= -20:
                        rank_emoji = "⬇️"
                    elif net_score < 0:
                        rank_emoji = "📊"
                    else:
                        rank_emoji = "➡️"
                else:
                    rank_emoji = "⚡" if abs(net_score) > 30 else "📊"
                
                trending_lines.append(
                    f"{rank_emoji} {i+1}. **{token_data['token_symbol']}** {momentum_indicator}"
                )
                trending_lines.append(
                    f"   Net: {net_score:+.1f} (💚{buy_momentum:+.1f} 💔-{sell_momentum:.1f})"
                )
                
                volume = token_data.get('total_volume', 0)
                if volume > 1.0:
                    trending_lines.append(f"   Volume: {volume:.2f} ETH")
                
                # Add alert count for context
                alert_count = token_data.get('alert_count', 0)
                if alert_count > 1:
                    trending_lines.append(f"   Alerts: {alert_count} in 24h")
                
                trending_lines.append("")  # Spacing
            
            # Enhanced legend based on analysis type
            if analysis_type == 'buy':
                legend_lines = [
                    "📈 **Buy Strategy Legend:**",
                    "🚀 = Strong bullish (+50+), 📈 = Bullish (+20+), ⬆️ = Positive (>0)",
                    "💚 = Buy momentum, 💔 = Sell pressure",
                    "**Focus on highest positive scores for buy opportunities**"
                ]
            elif analysis_type == 'sell':
                legend_lines = [
                    "📉 **Sell Strategy Legend:**",
                    "📉 = Strong bearish (-50+), ⬇️ = Bearish (-20+), 📊 = Negative (<0)",
                    "💚 = Buy momentum, 💔 = Sell pressure", 
                    "**Focus on lowest negative scores for sell opportunities**"
                ]
            else:
                legend_lines = [
                    "📈 **Legend:**",
                    "💚 = Buy momentum, 💔 = Sell pressure",
                    "Net = Combined score (Buys - Sells)"
                ]
            
            trending_lines.extend(legend_lines)
            trending_lines.extend([
                "",
                f"🌐 **Network:** {network.upper()}",
                f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}"
            ])
            
            message = '\n'.join(trending_lines)
            logger.info(f"Sending {analysis_type} trending summary: {len(message)} characters")
            
            success = await self.send_message(message)
            if success:
                logger.info(f"Trending summary sent successfully for {analysis_type} analysis")
            else:
                logger.error("Failed to send trending summary message")
                
        except Exception as e:
            logger.error(f"Trending summary failed: {e}")
            import traceback
            logger.error(f"Trending summary traceback: {traceback.format_exc()}")
      
    async def _send_analysis_summary(self, result, network: str, alerts_sent: int, min_alpha_score: float, store_alerts: bool = True):
        """Send analysis summary with storage info"""
        
        network_info = get_network_info(network)
        
        storage_info = ""
        if hasattr(result, 'performance_metrics'):
            transfers_stored = result.performance_metrics.get('transfers_stored', 0)
            if transfers_stored > 0:
                storage_info = f"\n🗄️ **Transfer Storage:** {transfers_stored} records"
            else:
                storage_info = f"\n🗄️ **Transfer Storage:** Disabled"
        
        # Add alert storage info
        alert_storage_info = "Enabled" if store_alerts else "Disabled"
        storage_info += f"\n📊 **Alert Storage:** {alert_storage_info}"
        
        summary_message = f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

    ✅ **Alerts Sent:** {alerts_sent}
    📈 **Total Tokens Found:** {result.unique_tokens}
    💰 **Total ETH Volume:** {result.total_eth_value:.4f}
    🔍 **Filter:** min score {min_alpha_score}{storage_info}

    🌐 **Network:** {network_info['name']} ({network_info['symbol']})
    🚀 **Features:** Enhanced intelligence, momentum tracking, Web3 data
    ⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        
        await self.send_message(summary_message.strip())
        
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