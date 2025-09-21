import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# PRIME TIME SYSTEM
# ============================================================================

class PrimeTimeSystem:
    """Unified prime time system for daily and weekly trading optimization"""
    
    def __init__(self):
        # Daily prime time windows (NZDT timezone)
        self.daily_prime_times = {
            'buy': {
                'prime_hour': 20,        # 9:00 AM NZDT - Lowest prices (Asian morning lull)
                'prime_minute': 0,
                'window_minutes': 45,    # ±45 minutes around prime time
                'description': 'DAILY PEAK BUY WINDOW - Lowest global prices'
            },
            'sell': {
                'prime_hour': 5,         # 6:00 PM NZDT - Peak liquidity (Asia close + Europe active)
                'prime_minute': 0,
                'window_minutes': 45,    # ±45 minutes around prime time
                'description': 'DAILY PEAK SELL WINDOW - Highest global liquidity'
            }
        }
        
        # Weekly prime time windows (UTC timezone, Ethereum-focused)
        self.weekly_prime_times = {
            'sell': {
                'day': 1,  # Tuesday UTC (busiest day for Ethereum)
                'hour': 21,  # 9:00 PM UTC = 4PM ET (peak busy hours start)
                'minute': 0,
                'window_hours': 4,  # ±4 hours window
                'description': 'PEAK ETH ACTIVITY WINDOW - Highest trading volume + fees = best selling prices',
                'nz_equivalent': 'Wednesday 10:00 AM NZDT'
            },
            'buy': {
                'day': 6,  # Sunday UTC (least volume day)
                'hour': 10,  # 10:00 AM UTC = 5:00 AM ET (when people sleeping)
                'minute': 0,
                'window_hours': 4,  # ±4 hours window
                'description': 'LOWEST ETH FEES WINDOW - When people sleeping = cheapest prices',
                'nz_equivalent': 'Sunday 11:00 PM NZDT'
            },
            'buy_secondary': {
                'day': 5,  # Saturday UTC = weekend (lower activity)
                'hour': 12,  # 12:00 PM UTC = 7:00 AM ET (early morning weekend)
                'minute': 0,
                'window_hours': 6,  # ±6 hours window (wider weekend window)
                'description': 'WEEKEND SPECIAL - Lower weekend activity = better ETH prices',
                'nz_equivalent': 'Sunday 1:00 AM NZDT'
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
                minutes_until = -minutes_diff - window_minutes
                next_occurrence = "today"
            else:
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
    
    def is_weekly_prime_time(self, analysis_type: str, current_time: datetime = None) -> Dict[str, Any]:
        """Check if current time is within the weekly prime window"""
        if current_time is None:
            current_time = datetime.now()
        
        # Check primary window
        prime_config = self.weekly_prime_times.get(analysis_type)
        if prime_config:
            result = self._check_weekly_time_window(current_time, prime_config, analysis_type)
            if result['is_weekly_prime']:
                return result
        
        # For buy analysis, also check Sunday special window
        if analysis_type == 'buy':
            sunday_config = self.weekly_prime_times.get('buy_secondary')
            if sunday_config:
                sunday_result = self._check_weekly_time_window(current_time, sunday_config, 'buy_sunday')
                if sunday_result['is_weekly_prime']:
                    sunday_result['window_type'] = 'SUNDAY_SPECIAL'
                    return sunday_result
        
        return {'is_weekly_prime': False}
    
    def _check_weekly_time_window(self, current_time: datetime, prime_config: dict, analysis_type: str) -> Dict[str, Any]:
        """Helper method to check if current time is within a specific weekly window"""
        # Get current day of week (0=Monday, 6=Sunday)
        current_weekday = current_time.weekday()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Check if we're on the right day
        if current_weekday != prime_config['day']:
            return {'is_weekly_prime': False}
        
        # Calculate time difference in minutes
        current_total_minutes = current_hour * 60 + current_minute
        prime_total_minutes = prime_config['hour'] * 60 + prime_config['minute']
        window_minutes = prime_config['window_hours'] * 60
        
        minutes_diff = abs(current_total_minutes - prime_total_minutes)
        
        # Check if within the weekly prime window
        if minutes_diff <= window_minutes:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            return {
                'is_weekly_prime': True,
                'window_type': 'WEEKLY_PEAK',
                'description': prime_config['description'],
                'minutes_from_peak': current_total_minutes - prime_total_minutes,
                'day_name': day_names[prime_config['day']],
                'peak_time': f"{prime_config['hour']:02d}:{prime_config['minute']:02d} UTC",
                'nz_time': prime_config.get('nz_equivalent', 'NZ Time'),
                'analysis_type': analysis_type,
                'data_based': True
            }
        
        return {'is_weekly_prime': False}
    
    def _format_time(self, hour: int, minute: int) -> str:
        """Format time handling overflow/underflow"""
        while minute >= 60:
            minute -= 60
            hour += 1
        while minute < 0:
            minute += 60
            hour -= 1
        
        hour = hour % 24
        return f"{hour:02d}:{minute:02d}"
    
    def enhance_alerts_for_prime_time(self, alerts: list, analysis_type: str) -> list:
        """Enhance alerts with both daily and weekly prime time information"""
        daily_prime_info = self.is_daily_prime_time(analysis_type)
        weekly_prime_info = self.is_weekly_prime_time(analysis_type)
        
        # Log prime time status
        if daily_prime_info['is_prime_time']:
            logger.info(f"DAILY PRIME TIME ACTIVE: {daily_prime_info['prime_reason']}")
        
        if weekly_prime_info.get('is_weekly_prime'):
            logger.info(f"WEEKLY PRIME TIME DETECTED: {weekly_prime_info['description']}")
        
        enhanced_alerts = []
        
        # Add special weekly prime time alert if active
        if weekly_prime_info.get('is_weekly_prime'):
            weekly_alert = {
                'token': 'WEEKLY PRIME TIME',
                'alert_type': f'weekly_prime_{analysis_type}',
                'network': 'all',
                'is_weekly_prime_alert': True,
                'weekly_prime_info': weekly_prime_info,
                'priority_level': 'WEEKLY_PEAK',
                'data': {
                    'special_alert': True,
                    'weekly_prime': True,
                    'alpha_score': 999,
                    'total_eth_spent': 0,
                    'wallet_count': 0,
                    'contract_address': '',
                    'ca': ''
                },
                'confidence': 'WEEKLY PEAK TIME',
                'ai_data': {},
                'score': 999
            }
            enhanced_alerts.append(weekly_alert)
        
        # Enhance regular alerts
        for alert in alerts:
            enhanced_alert = alert.copy()
            enhanced_alert['daily_prime_info'] = daily_prime_info
            enhanced_alert['weekly_prime_info'] = weekly_prime_info
            
            # Set priority based on prime time status
            if daily_prime_info['is_prime_time']:
                enhanced_alert['priority_level'] = daily_prime_info['priority_level']
                enhanced_alert['is_daily_prime_time'] = True
                enhanced_alert['window_type'] = daily_prime_info['window_type']
            else:
                enhanced_alert['priority_level'] = 'MEDIUM'
                enhanced_alert['is_daily_prime_time'] = False
            
            enhanced_alert['final_score'] = enhanced_alert.get('enhanced_score', enhanced_alert.get('score', 0))
            enhanced_alerts.append(enhanced_alert)
        
        return enhanced_alerts

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_token_age(ai_data: dict, data: dict) -> float:
    """Extract token age from AI data or data sources with comprehensive search"""
    logger.info(f"Extracting token age...")
    token_age_hours = None
    
    # Try multiple sources and field names
    sources = [ai_data, data.get('web3_data', {}), data.get('web3_analysis', {}), data]
    field_names = ['token_age_hours', 'contract_age_hours', 'age_hours']
    
    for source in sources:
        if not source:
            continue
        for field in field_names:
            if token_age_hours is None:
                token_age_hours = source.get(field)
                if token_age_hours is not None:
                    logger.info(f"Found token age: {token_age_hours} hours from {field}")
                    break
        if token_age_hours is not None:
            break
    
    return token_age_hours

def format_token_age_display(token_age_hours) -> str:
    """Format token age with risk indicators"""
    if token_age_hours is None:
        return "Unknown Age"
    
    try:
        hours = float(token_age_hours)
        
        if hours <= 0:
            return "Just Created (BRAND NEW - EXTREME RISK)"
        elif hours < 0.1:
            return f"{hours*60:.0f} minutes (JUST LAUNCHED - EXTREME RISK)"
        elif hours < 1:
            return f"{hours*60:.0f} minutes (VERY NEW - HIGH RISK)"
        elif hours < 24:
            return f"{hours:.1f} hours (NEW - MONITOR CLOSELY)"
        elif hours < 168:  # 1 week
            return f"{hours/24:.1f} days (RECENT)"
        elif hours < 720:  # 1 month
            return f"{hours/24:.0f} days (ESTABLISHED)"
        else:
            return f"{hours/24:.0f} days (MATURE)"
            
    except (ValueError, TypeError):
        return "Invalid Age Data"

def get_network_info(network: str) -> Dict[str, str]:
    """Get network-specific information for links and explorers"""
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

def format_contract_address(contract_address: str) -> str:
    """Format contract address with validation"""
    if not contract_address or len(contract_address) < 10:
        return "No contract address available"
    
    # Clean the address
    clean_address = contract_address.strip().lower()
    if not clean_address.startswith('0x'):
        clean_address = '0x' + clean_address
    
    # Validate length
    if len(clean_address) != 42:
        return "Invalid contract address"
    
    return f"`{clean_address}`"

def generate_action_links(token: str, contract_address: str, network: str) -> str:
    """Generate action links for trading and analysis"""
    if not contract_address or len(contract_address) < 10:
        return "No contract address available"
    
    network_info = get_network_info(network)
    
    # Clean contract address
    clean_ca = contract_address.strip().lower()
    if not clean_ca.startswith('0x'):
        clean_ca = '0x' + clean_ca
    
    # Validate address
    if len(clean_ca) != 42:
        return "Invalid contract address"
    
    # Generate links
    links = [
        f"[🦄 Uniswap]({network_info['uniswap_base']}{clean_ca})",
        f"[📊 Chart]({network_info['dexscreener_base']}{clean_ca})",
        f"[🔍 Explorer](https://{network_info['explorer']}/token/{clean_ca})",
        f"[🐦 Search X](https://twitter.com/search?q={clean_ca})",
        f"[🔧 DEXTools](https://www.dextools.io/app/en/{network}/pair-explorer/{clean_ca})"
    ]
    
    return " | ".join(links)

def check_notification_config() -> bool:
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    return bool(bot_token and chat_id and len(bot_token) > 40)

# ============================================================================
# TELEGRAM CLIENT
# ============================================================================

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
                "User-Agent": "CryptoMonitor/4.0",
                "Accept": "application/json"
            }
            
            response = await self._client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

# ============================================================================
# ALERT FORMATTING
# ============================================================================

def format_alert_message(alert: dict) -> str:
    """Main alert formatting function with prime time integration"""
    try:
        # Check for special weekly prime time alert
        if alert.get('is_weekly_prime_alert'):
            return format_weekly_prime_alert(alert)
        
        # Check for daily prime time
        daily_prime_info = alert.get('daily_prime_info', {})
        if daily_prime_info.get('is_prime_time'):
            return format_daily_prime_time_message(alert)
        
        # Standard enhanced formatting
        return format_enhanced_alert_message(alert)
        
    except Exception as e:
        logger.error(f"Alert formatting failed: {e}")
        return format_basic_alert_message(alert)

def format_weekly_prime_alert(alert: Dict) -> str:
    """Format the special weekly prime time alert"""
    weekly_info = alert.get('weekly_prime_info', {})
    analysis_type = weekly_info.get('analysis_type', 'unknown')
    
    if analysis_type == 'sell':
        header = "🔥💎🔥 **PEAK ETH TRADING WINDOW** 🔥💎🔥"
        strategy = "🎯 **MAXIMUM LIQUIDITY TIME**"
        action = "💰 **SELL NOW** - Highest ETH/ERC20 activity = best prices!"
        timing = "📈 Peak Ethereum network activity - highest fees = highest prices"
    elif analysis_type == 'buy_sunday':
        header = "🌟⭐🌟 **WEEKEND ETH SPECIAL** 🌟⭐🌟" 
        strategy = "🎯 **WEEKEND LOW ACTIVITY**"
        action = "🛒 **BUY NOW** - Weekend = lower ETH fees and prices!"
        timing = "📉 Weekend activity drops = cheaper transactions"
    else:  # buy
        header = "🔥💚🔥 **SLEEPING HOURS ETH WINDOW** 🔥💚🔥"
        strategy = "🎯 **LOWEST FEES TIME**"
        action = "🛒 **BUY NOW** - When people sleep = cheapest ETH!"
        timing = "😴 1AM-8AM ET = lowest gas fees and prices"
    
    day_name = weekly_info.get('day_name', 'Unknown')
    peak_time = weekly_info.get('peak_time', 'Unknown')
    nz_time = weekly_info.get('nz_time', 'Unknown')
    minutes_offset = weekly_info.get('minutes_from_peak', 0)
    
    message = f"""{header}

{strategy}
{timing}

🗓️ **Prime Day:** {day_name} (UTC)
🕐 **Peak Time:** {peak_time}
🇳🇿 **NZ Time:** {nz_time}
⏰ **Current Offset:** {minutes_offset:+d} minutes from peak

{action}

🌍 **Why This ETH Window Works:**
• Based on Ethereum network activity patterns
• Gas fees directly correlate with trading activity
• ERC20 tokens follow ETH network congestion
• Peak activity = highest prices for sellers
• Low activity = cheapest prices for buyers

🚨🚨🚨 **WEEKLY PEAK ACTIVE** 🚨🚨🚨
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
    
    return message

def format_daily_prime_time_message(alert: Dict) -> str:
    """Format alert message with daily prime time enhancements"""
    prime_info = alert.get('daily_prime_info', {})
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
    
    # Get contract address
    contract_address = data.get('contract_address', data.get('ca', ''))
    
    message = f"""{header}

{strategy_desc}
{timing_desc}

🪙 **Token:** `{token}`
🌐 **Network:** {get_network_info(network)['name']}
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
    if contract_address:
        message += f"""

📋 **Contract Address:**
{format_contract_address(contract_address)}

🚨 **PRIORITY ACTIONS:**
{generate_action_links(token, contract_address, network)}"""
    
    message += f"""

🚨🚨🚨 **DAILY PRIME TIME - ACT NOW** 🚨🚨🚨
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
    
    return message

def format_enhanced_alert_message(alert: dict) -> str:
    """Format enhanced alert with Web3 intelligence and token age"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    token = alert.get('token', 'UNKNOWN')
    network = alert.get('network', 'ethereum')
    confidence = alert.get('confidence', 'Unknown')
    
    # Get contract address
    contract_address = data.get('contract_address', data.get('ca', ''))
    
    # Extract Web3 intelligence from AI analysis
    ai_data = alert.get('ai_data', {})
    
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

    # Get token age
    token_age_hours = extract_token_age(ai_data, data)
    age_display = format_token_age_display(token_age_hours)
    
    # Build Web3 intelligence section
    web3_signals = []
    
    if ai_data:
        # Verification status
        if ai_data.get('is_verified'):
            web3_signals.append("✅ Contract Verified")
        elif ai_data.get('is_verified') is False:
            web3_signals.append("⚠️ Unverified Contract")
        
        # Liquidity signals
        liquidity_usd = ai_data.get('liquidity_usd', 0)
        if ai_data.get('has_liquidity') and liquidity_usd > 10000:
            web3_signals.append(f"💧 Liquidity: ${liquidity_usd:,.0f}")
        
        # Advanced signals
        if ai_data.get('smart_money_buying') or ai_data.get('has_smart_money'):
            web3_signals.append("🧠 Smart Money Active")
        
        if ai_data.get('whale_coordination_detected'):
            web3_signals.append("🐋 Whale Coordination")
        
        # Risk assessment
        honeypot_risk = ai_data.get('honeypot_risk', 0)
        if honeypot_risk > 0.4:
            web3_signals.append(f"⚠️ Risk: {honeypot_risk:.0%}")

    # Build the complete message
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
        f"🕐 **Age:** {age_display}",
        "",
        f"🎯 **Confidence:** {confidence}"
    ]

    # Add Web3 intelligence section if available
    if web3_signals:
        message_parts.extend(["", "🔍 **Web3 Intelligence:**"])
        for signal in web3_signals[:4]:  # Limit to prevent overflow
            message_parts.append(f"  {signal}")

    # Contract address section
    if contract_address:
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

def format_basic_alert_message(alert: dict) -> str:
    """Basic fallback alert formatting"""
    try:
        token = alert.get('token', 'UNKNOWN')
        network = alert.get('network', 'unknown')
        alert_type = alert.get('alert_type', 'unknown')
        
        data = alert.get('data', {})
        score = (data.get('alpha_score') or 
                data.get('sell_pressure_score') or 
                data.get('total_score') or 
                alert.get('score', 0))
        
        eth_value = (data.get('total_eth_spent') or 
                    data.get('total_eth_received') or 
                    data.get('total_eth_value', 0))
        
        return f"""🚨 **{alert_type.upper()} ALERT**

🪙 **Token:** `{token}`
🌐 **Network:** {network.upper()}
📊 **Score:** {score:.1f}
💰 **ETH:** {eth_value:.4f}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
⚠️ Basic formatting (enhanced failed)"""
        
    except Exception as e:
        logger.error(f"Basic formatting failed: {e}")
        return f"🚨 **ALERT** - {alert.get('token', 'ERROR')} - {alert.get('alert_type', 'unknown')}"

# ============================================================================
# TELEGRAM SERVICE (MAIN CLASS)
# ============================================================================

class TelegramService:
    """Main Telegram service with integrated prime time system"""
    
    def __init__(self):
        self.prime_time_system = PrimeTimeSystem()
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
        """Test Telegram connection"""
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
                # Test bot
                bot_response = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
                
                if bot_response.status_code != 200:
                    return {"configured": False, "error": f"Bot API returned {bot_response.status_code}"}
                
                bot_data = bot_response.json()
                if not bot_data.get('ok'):
                    return {"configured": False, "error": f"Bot API error: {bot_data.get('description')}"}
                
                # Test chat
                chat_response = await client.get(
                    f"https://api.telegram.org/bot{bot_token}/getChat",
                    params={"chat_id": chat_id}
                )
                
                chat_accessible = chat_response.status_code == 200
                
                return {
                    "configured": True,
                    "bot_valid": True,
                    "chat_accessible": chat_accessible,
                    "ready_for_notifications": chat_accessible,
                    "prime_time_system": True,
                    "momentum_tracking": bool(self.momentum_tracker)
                }
                
        except Exception as e:
            return {"configured": False, "error": f"Connection test failed: {str(e)}"}
    
    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int, 
                                    days_back: float, use_smart_timing: bool, max_tokens: int, 
                                    min_alpha_score: float, store_data: bool = False):
        """Send enhanced analysis start notification"""
        try:
            timing_info = f"⏰ Smart: {days_back}d" if use_smart_timing else f"⏰ Manual: {days_back}d"
            storage_info = f"🗄️ Storage: {'Enabled' if store_data else 'Disabled'}"
            network_info = get_network_info(network)
            
            # Check current prime time status
            daily_prime = self.prime_time_system.is_daily_prime_time(analysis_type)
            weekly_prime = self.prime_time_system.is_weekly_prime_time(analysis_type)
            
            prime_status = ""
            if daily_prime.get('is_prime_time'):
                prime_status = f"\n🕐 **DAILY PRIME TIME ACTIVE** - {daily_prime['window_type']} WINDOW"
            if weekly_prime.get('is_weekly_prime'):
                prime_status += f"\n🔥 **WEEKLY PRIME TIME ACTIVE** - {weekly_prime['description']}"
            
            start_message = f"""🚀 **ENHANCED ANALYSIS STARTED v4.0**

**Network:** {network_info['name']} ({network_info['symbol']})
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
{timing_info}
{storage_info}
**AI Enhancement:** Enabled
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score{prime_status}

🚀 **Enhanced Features:**
• Integrated Prime Time System (Daily + Weekly)
• Contract addresses with verification status
• Web3 intelligence and risk analysis
• Token age analysis with risk indicators
• Momentum tracking across 5 days
• Direct Uniswap trading links
• DexScreener charts and Twitter search

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
🚀 Enhanced Web3 Monitoring v4.0 with Prime Time"""
            
            await self.send_message(start_message)
            
        except Exception as e:
            logger.error(f"Failed to send start notification: {e}")
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, 
                                        min_alpha_score: float = 50.0, store_alerts: bool = True):
        """Send analysis notifications with prime time enhancement"""
        try:
            if not result.ranked_tokens:
                await self.send_message(f"No tokens found for {network.upper()}")
                return
            
            # Filter and process tokens
            qualifying_tokens = []
            for token_data in result.ranked_tokens[:max_tokens]:
                try:
                    # Handle tuple format safely
                    if isinstance(token_data, (list, tuple)) and len(token_data) >= 3:
                        token = token_data[0]
                        data = token_data[1]
                        score = token_data[2]
                        ai_data = token_data[3] if len(token_data) >= 4 else {}
                        
                        if score >= min_alpha_score:
                            alert = {
                                'token': token,
                                'data': data,
                                'alert_type': result.analysis_type,
                                'network': network,
                                'confidence': ai_data.get('confidence', 'Unknown'),
                                'ai_data': ai_data,
                                'score': score
                            }
                            qualifying_tokens.append(alert)
                            
                except Exception as e:
                    logger.error(f"Error processing token: {e}")
                    continue
            
            if not qualifying_tokens:
                await self.send_message(f"No tokens above threshold {min_alpha_score} for {network.upper()}")
                return
            
            # Enhance alerts with prime time information
            enhanced_alerts = self.prime_time_system.enhance_alerts_for_prime_time(
                qualifying_tokens, result.analysis_type
            )
            
            # Send notifications
            for alert in enhanced_alerts:
                try:
                    # Add momentum data if available
                    if self.momentum_tracker and not alert.get('is_weekly_prime_alert'):
                        momentum_data = await self.momentum_tracker.get_token_momentum(
                            alert['token'], alert['network'], days_back=5
                        )
                        alert['momentum_data'] = momentum_data
                        
                        # Store alert if enabled
                        if store_alerts:
                            await self.momentum_tracker.store_alert(alert)
                    
                    # Format and send
                    message = self._format_alert_with_momentum(alert)
                    await self.send_message(message)
                    await asyncio.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Failed to send alert for {alert.get('token')}: {e}")
                    continue
            
            if self.momentum_tracker:
                    try:
                        await self._send_trending_summary(network, result.analysis_type)
                        logger.info("Trending summary sent successfully")
                    except Exception as e:
                        logger.error(f"Failed to send trending summary: {e}")
                        
            # Send summary
            await self._send_summary(result, network, len([a for a in enhanced_alerts if not a.get('is_weekly_prime_alert')]), store_alerts)
            
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
    
    def _format_alert_with_momentum(self, alert: dict) -> str:
        """Format alert with momentum data"""
        try:
            # Start with prime time aware formatting
            message = format_alert_message(alert)
            
            # Add momentum section if available (not for special alerts)
            if not alert.get('is_weekly_prime_alert'):
                momentum_data = alert.get('momentum_data', {})
                if momentum_data and momentum_data.get('net_momentum_score') is not None:
                    momentum_section = self._build_momentum_section(momentum_data)
                    
                    # Insert before contract address
                    lines = message.split('\n')
                    contract_index = next((i for i, line in enumerate(lines) 
                                        if '**Contract Address:**' in line), len(lines))
                    
                    lines.insert(contract_index, momentum_section)
                    lines.insert(contract_index + 1, "")
                    message = '\n'.join(lines)
                elif not alert.get('is_weekly_prime_alert'):
                    # Add first alert note for regular alerts only
                    lines = message.split('\n')
                    contract_index = next((i for i, line in enumerate(lines) 
                                        if '**Contract Address:**' in line), len(lines))
                    if contract_index < len(lines):
                        lines.insert(contract_index, "📊 **First Alert** - Building momentum data...")
                        lines.insert(contract_index + 1, "")
                        message = '\n'.join(lines)
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting with momentum: {e}")
            return format_basic_alert_message(alert)
    
    def _build_momentum_section(self, momentum_data: dict) -> str:
        """Build momentum analysis section"""
        net_score = momentum_data.get('net_momentum_score', 0)
        alert_count = momentum_data.get('alert_count', 0)
        
        lines = ["📊 **Momentum Analysis:**"]
        
        # Net score display
        if net_score >= 50:
            lines.append(f"  🚀 **Net Score: +{net_score:.1f}** (VERY BULLISH)")
        elif net_score >= 20:
            lines.append(f"  📈 **Net Score: +{net_score:.1f}** (BULLISH)")
        elif net_score >= 5:
            lines.append(f"  ⬆️ **Net Score: +{net_score:.1f}** (SLIGHT BUY)")
        elif net_score <= -50:
            lines.append(f"  📉 **Net Score: {net_score:.1f}** (STRONG SELL)")
        elif net_score <= -20:
            lines.append(f"  ⬇️ **Net Score: {net_score:.1f}** (BEARISH)")
        else:
            lines.append(f"  ➡️ **Net Score: {net_score:.1f}** (NEUTRAL)")
        
        if alert_count > 1:
            lines.append(f"  📊 {alert_count} alerts over 5 days")
        
        return "\n".join(lines)
    
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
            
    async def _send_summary(self, result, network: str, alerts_sent: int, store_alerts: bool):
        """Send analysis summary"""
        network_info = get_network_info(network)
        alert_storage = "Enabled" if store_alerts else "Disabled"
        
        # Check prime time status for summary
        daily_prime = self.prime_time_system.is_daily_prime_time(result.analysis_type)
        weekly_prime = self.prime_time_system.is_weekly_prime_time(result.analysis_type)
        
        prime_summary = ""
        if daily_prime.get('is_prime_time'):
            prime_summary = f"\n🕐 Analysis completed during DAILY PRIME TIME"
        if weekly_prime.get('is_weekly_prime'):
            prime_summary += f"\n🔥 Analysis completed during WEEKLY PRIME TIME"
        
        summary = f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

✅ **Alerts Sent:** {alerts_sent}
📈 **Total Tokens:** {result.unique_tokens}
💰 **Total ETH:** {result.total_eth_value:.4f}
📊 **Alert Storage:** {alert_storage}{prime_summary}

🌐 **Network:** {network_info['name']}
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        
        await self.send_message(summary)

# ============================================================================
# GLOBAL INSTANCES AND CONVENIENCE FUNCTIONS
# ============================================================================

# Global instances
telegram_client = TelegramClient()
telegram_service = TelegramService()

async def send_alert_notifications(alerts: list):
    """Send alert notifications with prime time enhancement"""
    if not alerts:
        return
    
    # Determine analysis type from first alert
    analysis_type = alerts[0].get('alert_type', 'buy') if alerts else 'buy'
    
    # Enhance alerts with prime time information
    enhanced_alerts = telegram_service.prime_time_system.enhance_alerts_for_prime_time(
        alerts, analysis_type
    )
    
    async with telegram_client:
        for alert in enhanced_alerts:
            try:
                message = format_alert_message(alert)
                await telegram_client.send_message(message)
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")

async def send_test_notification():
    """Send test notification with prime time status"""
    # Check current prime time status for both buy and sell
    prime_system = PrimeTimeSystem()
    buy_daily = prime_system.is_daily_prime_time('buy')
    sell_daily = prime_system.is_daily_prime_time('sell')
    buy_weekly = prime_system.is_weekly_prime_time('buy')
    sell_weekly = prime_system.is_weekly_prime_time('sell')
    
    prime_status = []
    if buy_daily.get('is_prime_time'):
        prime_status.append(f"🟢 BUY Daily Prime: {buy_daily['window_type']}")
    if sell_daily.get('is_prime_time'):
        prime_status.append(f"🔴 SELL Daily Prime: {sell_daily['window_type']}")
    if buy_weekly.get('is_weekly_prime'):
        prime_status.append(f"🔥 BUY Weekly Prime Active")
    if sell_weekly.get('is_weekly_prime'):
        prime_status.append(f"🔥 SELL Weekly Prime Active")
    
    if not prime_status:
        prime_status.append("⏰ No prime time windows active")
    
    test_message = f"""🧪 **TEST NOTIFICATION**

✅ Crypto Monitor Active!
🚀 Prime Time System Integrated
{chr(10).join(prime_status)}

🕐 {datetime.now().strftime('%H:%M:%S UTC')}
🚀 All systems operational v4.0"""
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

# Export everything
__all__ = [
    'PrimeTimeSystem', 'TelegramClient', 'TelegramService', 
    'telegram_client', 'telegram_service',
    'send_alert_notifications', 'send_test_notification', 'check_notification_config',
    'format_alert_message', 'format_enhanced_alert_message', 'format_basic_alert_message',
    'format_daily_prime_time_message', 'format_weekly_prime_alert',
    'get_network_info', 'format_contract_address', 'generate_action_links',
    'extract_token_age', 'format_token_age_display'
]