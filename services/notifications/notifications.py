import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# TOKEN AGE AND RISK ASSESSMENT FUNCTIONS
# ============================================================================

def extract_token_age(ai_data: dict, data: dict) -> float:
    """Extract token age from AI data or data sources with comprehensive search"""
    logger.info(f"🔍 DEBUG: Extracting token age...")
    logger.info(f"AI data keys: {list(ai_data.keys()) if ai_data else 'None'}")
    logger.info(f"Data keys: {list(data.keys()) if data else 'None'}")
    
    token_age_hours = None
    
    # Try AI data first - multiple possible locations
    if ai_data:
        # Direct in ai_data
        token_age_hours = ai_data.get('token_age_hours')
        logger.info(f"AI data token_age_hours: {token_age_hours}")
        
        # In web3_analysis within ai_data
        if token_age_hours is None and 'web3_analysis' in ai_data:
            token_age_hours = ai_data['web3_analysis'].get('token_age_hours')
            logger.info(f"AI web3_analysis token_age_hours: {token_age_hours}")
        
        # Sometimes it's stored as contract_age_hours
        if token_age_hours is None:
            token_age_hours = ai_data.get('contract_age_hours')
            logger.info(f"AI contract_age_hours: {token_age_hours}")
    
    # Try web3_data in main data
    if token_age_hours is None and data.get('web3_data'):
        web3_data = data['web3_data']
        token_age_hours = web3_data.get('token_age_hours')
        logger.info(f"Data web3_data token_age_hours: {token_age_hours}")
        
        # Also try contract_age_hours
        if token_age_hours is None:
            token_age_hours = web3_data.get('contract_age_hours')
            logger.info(f"Data web3_data contract_age_hours: {token_age_hours}")
    
    # Try direct from data
    if token_age_hours is None:
        token_age_hours = data.get('token_age_hours')
        logger.info(f"Data direct token_age_hours: {token_age_hours}")
    
    # Check if it's nested in web3_analysis in data
    if token_age_hours is None and data.get('web3_analysis'):
        web3_analysis = data['web3_analysis']
        token_age_hours = web3_analysis.get('token_age_hours')
        logger.info(f"Data web3_analysis token_age_hours: {token_age_hours}")
        
        # Also try contract_age_hours
        if token_age_hours is None:
            token_age_hours = web3_analysis.get('contract_age_hours')
            logger.info(f"Data web3_analysis contract_age_hours: {token_age_hours}")
    
    # Log all available keys that might contain age info for debugging
    debug_age_keys(ai_data, data)
    
    logger.info(f"Final token_age_hours: {token_age_hours}")
    return token_age_hours

def debug_age_keys(ai_data: dict, data: dict):
    """Debug function to find all keys that might contain age information"""
    all_keys = []
    
    def collect_keys_with_values(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if any(term in key.lower() for term in ['age', 'time', 'hour', 'day', 'created', 'deploy']):
                    all_keys.append(f"{full_key}: {value}")
                if isinstance(value, dict):
                    collect_keys_with_values(value, full_key)
    
    # Search both data structures
    if ai_data:
        collect_keys_with_values(ai_data, "ai_data")
    if data:
        collect_keys_with_values(data, "data")
    
    if all_keys:
        logger.info(f"🕐 Age-related keys found: {all_keys}")
    else:
        logger.warning("❌ No age-related keys found in alert data")

def extract_holder_count(ai_data: dict, data: dict) -> int:
    """Extract holder count from AI data or data sources"""
    # Try multiple sources for holder count
    holder_count = None
    
    # Try AI data first
    if ai_data:
        holder_count = ai_data.get('holder_count')
    
    # Try web3_data
    if holder_count is None and data.get('web3_data'):
        holder_count = data['web3_data'].get('holder_count')
    
    # Try direct from data
    if holder_count is None:
        holder_count = data.get('holder_count')
    
    return holder_count

def format_token_age_display(token_age_hours) -> str:
    """Format token age with risk indicators and fallback options"""
    if token_age_hours is None:
        return "❓ **Unknown Age** (Age data not available)"
    
    try:
        hours = float(token_age_hours)
        
        # Handle special cases
        if hours <= 0:
            return "🆕 **Just Created** (BRAND NEW - EXTREME RISK)"
        elif hours < 0.1:  # Less than 6 minutes
            return f"🆕 **{hours*60:.0f} minutes** (JUST LAUNCHED - EXTREME RISK)"
        elif hours < 0.5:  # Less than 30 minutes
            return f"🆕 **{hours*60:.0f} minutes** (BRAND NEW - HIGH RISK)"
        elif hours < 1:
            return f"🆕 **{hours*60:.0f} minutes** (VERY NEW - HIGH RISK)"
        elif hours < 6:
            return f"⚡ **{hours:.1f} hours** (FRESH - CAUTION)"
        elif hours < 24:
            return f"⚡ **{hours:.1f} hours** (NEW - MONITOR CLOSELY)"
        elif hours < 48:
            return f"📅 **{hours/24:.1f} days** (RECENT)"
        elif hours < 168:  # 1 week
            return f"📅 **{hours/24:.1f} days** (ESTABLISHED)"
        elif hours < 720:  # 1 month
            return f"✅ **{hours/24:.0f} days** (MATURE)"
        elif hours < 8760:  # 1 year
            return f"💎 **{hours/24:.0f} days** (WELL ESTABLISHED)"
        else:
            return f"🏛️ **{hours/8760:.1f} years** (ANCIENT)"
            
    except (ValueError, TypeError):
        return "❓ **Invalid Age Data** (Cannot parse age information)"

def format_holder_count_display(holder_count) -> str:
    """Format holder count with risk indicators"""
    if holder_count is None:
        return "❓ Unknown Holders"
    
    try:
        count = int(holder_count)
        
        if count < 10:
            return f"🚨 **{count} holders** (EXTREME RISK)"
        elif count < 50:
            return f"⚠️ **{count} holders** (HIGH RISK)"
        elif count < 100:
            return f"⚠️ **{count} holders** (MEDIUM RISK)"
        elif count < 500:
            return f"✅ **{count:,} holders** (GOOD)"
        elif count < 2000:
            return f"💚 **{count:,} holders** (STRONG)"
        elif count < 10000:
            return f"💎 **{count:,} holders** (EXCELLENT)"
        else:
            return f"🌟 **{count:,} holders** (MASSIVE)"
            
    except (ValueError, TypeError):
        return "❓ Invalid Holder Data"

def get_combined_risk_indicator(token_age_hours, holder_count) -> str:
    """Generate combined risk assessment"""
    if token_age_hours is None and holder_count is None:
        return None
    
    risk_factors = []
    risk_level = "LOW"
    
    # Age-based risk
    if token_age_hours is not None:
        try:
            hours = float(token_age_hours)
            if hours < 1:
                risk_factors.append("BRAND NEW TOKEN")
                risk_level = "EXTREME"
            elif hours < 24:
                risk_factors.append("VERY NEW TOKEN")
                risk_level = "HIGH" if risk_level != "EXTREME" else risk_level
        except (ValueError, TypeError):
            pass
    
    # Holder-based risk
    if holder_count is not None:
        try:
            count = int(holder_count)
            if count < 50:
                risk_factors.append("FEW HOLDERS")
                if risk_level in ["LOW", "MEDIUM"]:
                    risk_level = "HIGH"
            elif count < 10:
                risk_factors.append("VERY FEW HOLDERS")
                risk_level = "EXTREME"
        except (ValueError, TypeError):
            pass
    
    # Generate warning message
    if len(risk_factors) >= 2:
        return f"🚨 **{risk_level} RISK:** {' + '.join(risk_factors)}"
    elif len(risk_factors) == 1:
        return f"⚠️ **{risk_level} RISK:** {risk_factors[0]}"
    elif token_age_hours is not None and holder_count is not None:
        try:
            hours = float(token_age_hours)
            count = int(holder_count)
            if hours > 720 and count > 1000:  # 1 month + 1000 holders
                return "✅ **LOW RISK:** Established token with wide distribution"
            elif hours > 168 and count > 200:  # 1 week + 200 holders
                return "💡 **MODERATE RISK:** Growing established project"
        except (ValueError, TypeError):
            pass
    
    return None

# ============================================================================
# CORE TELEGRAM CLIENT
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
                "User-Agent": "CryptoMonitor/3.2",
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
# UTILITY FUNCTIONS (SINGLE DEFINITIONS)
# ============================================================================

def get_network_info(network: str) -> Dict[str, str]:
    """Get network-specific information - SINGLE DEFINITION"""
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
        },
        'arbitrum': {
            'name': 'Arbitrum',
            'symbol': 'ETH',
            'explorer': 'arbiscan.io',
            'uniswap_base': 'https://app.uniswap.org/#/swap?chain=arbitrum&outputCurrency=',
            'dexscreener_base': 'https://dexscreener.com/arbitrum/',
            'chain_id': '42161',
            'emoji': '🔺'
        }
    }
    return network_configs.get(network.lower(), network_configs['ethereum'])

def format_contract_address(contract_address: str) -> str:
    """Format contract address - SINGLE DEFINITION"""
    if not contract_address or len(contract_address) < 10:
        return "❓ No contract address available"
    
    # Clean the address
    clean_address = contract_address.strip().lower()
    if not clean_address.startswith('0x'):
        clean_address = '0x' + clean_address
    
    # Validate length
    if len(clean_address) != 42:
        return "❓ Invalid contract address"
    
    # Format for display
    short_ca = f"{clean_address[:6]}...{clean_address[-4:]}"
    return f"`{clean_address}`\n💾 Short: `{short_ca}`"

def generate_action_links(token: str, contract_address: str, network: str) -> str:
    """Generate action links - SINGLE DEFINITION"""
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
# ALERT FORMATTING (UNIFIED SYSTEM)
# ============================================================================

def format_alert_message(alert: dict) -> str:
    """Main alert formatting function with fallback handling"""
    try:
        # Try enhanced formatting first
        return format_enhanced_alert_message(alert)
    except Exception as e:
        logger.error(f"Enhanced formatting failed: {e}")
        # Fallback to basic formatting
        return format_basic_alert_message(alert)

def format_enhanced_alert_message(alert: dict) -> str:
    """Format enhanced alert with Web3 intelligence, token age, holder count and Gini coefficient"""
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

    # Extract comprehensive Web3 data for prominence
    token_age_hours = None
    holder_count = None
    gini_coefficient = None
    concentration_risk = None
    top_10_holders_pct = None
    top_holder_pct = None
    
    # Try multiple sources for comprehensive data
    for source in [ai_data, web3_data, data]:
        if token_age_hours is None:
            token_age_hours = source.get('token_age_hours')
        if holder_count is None:
            holder_count = source.get('holder_count')
        if gini_coefficient is None:
            gini_coefficient = source.get('gini_coefficient')
        if concentration_risk is None:
            concentration_risk = source.get('holder_concentration_risk')
        if top_10_holders_pct is None:
            top_10_holders_pct = source.get('top_10_holders_percentage')
        if top_holder_pct is None:
            top_holder_pct = source.get('top_holder_percentage')

    # Enhanced formatting functions
    def format_token_age(hours):
        if hours is None:
            return "🕐 Age: Unknown"
        
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

    def format_holder_count(count):
        if count is None:
            return "👥 Holders: Unknown"
        
        if count < 50:
            return f"👥 Holders: 🚨 {count:,} (VERY RISKY)"
        elif count < 200:
            return f"👥 Holders: ⚠️ {count:,} (RISKY)"
        elif count < 1000:
            return f"👥 Holders: ✅ {count:,} (GOOD)"
        elif count < 5000:
            return f"👥 Holders: 💚 {count:,} (STRONG)"
        else:
            return f"👥 Holders: 💎 {count:,} (EXCELLENT)"

    def format_gini_coefficient(gini, risk, top_10_pct):
        if gini is None:
            return "📊 Distribution: Unknown"
        
        # Format Gini with interpretation
        if gini >= 0.8:
            interpretation = "🚨 VERY CONCENTRATED"
        elif gini >= 0.6:
            interpretation = "⚠️ CONCENTRATED"
        elif gini >= 0.4:
            interpretation = "📊 MODERATE"
        elif gini >= 0.2:
            interpretation = "✅ DISTRIBUTED"
        else:
            interpretation = "💎 WELL DISTRIBUTED"
        
        # Add top 10 info if available
        top_10_info = f" (Top 10: {top_10_pct:.1f}%)" if top_10_pct else ""
        risk_display = f" [{risk}]" if risk and risk != 'UNKNOWN' else ""
        
        return f"📊 Gini: {gini:.3f} {interpretation}{top_10_info}{risk_display}"

    # Create displays
    age_display = format_token_age(token_age_hours)
    holder_display = format_holder_count(holder_count)
    gini_display = format_gini_coefficient(gini_coefficient, concentration_risk, top_10_holders_pct)
    
    # Generate combined risk assessment
    def get_combined_risk_indicator(age_hours, holders, gini, concentration_risk, top_holder_pct):
        risk_signals = []
        
        if age_hours is not None and age_hours < 24:
            risk_signals.append("NEW TOKEN")
        if holders is not None and holders < 50:
            risk_signals.append("FEW HOLDERS")
        if gini is not None and gini > 0.7:
            risk_signals.append("HIGH GINI")
        if concentration_risk in ['VERY_HIGH', 'HIGH']:
            risk_signals.append("CONCENTRATED")
        if top_holder_pct is not None and top_holder_pct > 20:
            risk_signals.append("WHALE DOMINATED")
            
        if len(risk_signals) >= 3:
            return "🚨 EXTREME RISK: " + " + ".join(risk_signals[:2]) + f" + {len(risk_signals)-2} more"
        elif len(risk_signals) == 2:
            return "🚨 HIGH RISK: " + " + ".join(risk_signals)
        elif len(risk_signals) == 1:
            return "⚠️ CAUTION: " + risk_signals[0]
        elif all(x is not None for x in [age_hours, holders, gini]):
            if age_hours > 720 and holders > 1000 and gini < 0.5:
                return "✅ LOW RISK: Mature + Wide distribution + Fair wealth spread"
            elif age_hours > 168 and holders > 200 and gini < 0.6:
                return "💡 MODERATE RISK: Growing project with decent distribution"
        
        return None

    risk_indicator = get_combined_risk_indicator(token_age_hours, holder_count, gini_coefficient, concentration_risk, top_holder_pct)

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
        
        # Advanced signals
        if ai_data.get('smart_money_buying') or ai_data.get('has_smart_money'):
            web3_signals.append("🧠 Smart Money Active")
        
        if ai_data.get('whale_coordination_detected'):
            web3_signals.append("🐋 Whale Coordination")
        
        if ai_data.get('pump_signals_detected'):
            web3_signals.append("🚀 Pump Signals")
        
        # Risk assessment
        honeypot_risk = ai_data.get('honeypot_risk', 0)
        if honeypot_risk > 0.7:
            risk_signals.append(f"🍯 HIGH Honeypot Risk ({honeypot_risk:.0%})")
        elif honeypot_risk > 0.4:
            risk_signals.append(f"⚠️ Medium Risk ({honeypot_risk:.0%})")

    # Build the complete message with PROMINENT distribution analysis
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
        # PROMINENT PLACEMENT: Age, holder count, and distribution analysis
        "🔍 **Distribution Analysis:**",
        f"  {age_display}",
        f"  {holder_display}",
        f"  {gini_display}",
        "",
        f"🎯 **Confidence:** {confidence}"
    ]

    # Add combined risk assessment if available
    if risk_indicator:
        message_parts.extend(["", risk_indicator])

    # Add Web3 intelligence section (condensed)
    if web3_signals or risk_signals:
        message_parts.extend(["", "🔍 **Web3 Intelligence:**"])
        
        # Show most important signals first (limit to prevent overflow)
        for signal in web3_signals[:3]:
            message_parts.append(f"  {signal}")
        
        for risk in risk_signals[:2]:
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
    
    # Enhanced footer with distribution stats
    footer_stats = []
    if holder_count:
        footer_stats.append(f"👥{holder_count:,}")
    if gini_coefficient:
        footer_stats.append(f"📊{gini_coefficient:.3f}")
    if token_age_hours:
        footer_stats.append(f"🕐{token_age_hours/24:.1f}d")
    if concentration_risk and concentration_risk != 'UNKNOWN':
        footer_stats.append(f"⚠️{concentration_risk}")
    
    stats_display = " | ".join(footer_stats) if footer_stats else ""
    
    message_parts.extend([
        "",
        f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}" + (f" | {stats_display}" if stats_display else ""),
        "🚀 Enhanced Web3 Monitoring v4.1 with Distribution Analysis"
    ])
    
    return "\n".join(message_parts)

def debug_alert_data_structure(alert: dict):
    """Debug function to show the complete alert data structure"""
    logger.info("🔍 DEBUGGING ALERT DATA STRUCTURE:")
    logger.info(f"Alert keys: {list(alert.keys())}")
    
    data = alert.get('data', {})
    logger.info(f"Data keys: {list(data.keys())}")
    
    ai_data = alert.get('ai_data', {})
    logger.info(f"AI data keys: {list(ai_data.keys())}")
    
    # Check for web3_data in different places
    if 'web3_data' in data:
        logger.info(f"web3_data in data: {list(data['web3_data'].keys())}")
    
    if 'web3_analysis' in data:
        logger.info(f"web3_analysis in data: {list(data['web3_analysis'].keys())}")
    
    if 'web3_analysis' in ai_data:
        logger.info(f"web3_analysis in ai_data: {list(ai_data['web3_analysis'].keys())}")
    
    return alert

def format_basic_alert_message(alert: dict) -> str:
    """Basic fallback alert formatting with token age"""
    try:
        token = alert.get('token', 'UNKNOWN')
        network = alert.get('network', 'unknown')
        alert_type = alert.get('alert_type', 'unknown')
        
        # Get score safely
        data = alert.get('data', {})
        ai_data = alert.get('ai_data', {})
        
        score = (data.get('alpha_score') or 
                data.get('sell_pressure_score') or 
                data.get('total_score') or 
                alert.get('score', 0))
        
        eth_value = (data.get('total_eth_spent') or 
                    data.get('total_eth_received') or 
                    data.get('total_eth_value', 0))
        
        # Get token age for basic display
        token_age_hours = extract_token_age(ai_data, data)
        age_display = format_token_age_display(token_age_hours)
        
        return f"""🚨 **{alert_type.upper()} ALERT**

🪙 **Token:** `{token}`
🌐 **Network:** {network.upper()}
📊 **Score:** {score:.1f}
💰 **ETH:** {eth_value:.4f}

🕐 **Age:** {age_display}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
⚠️ Basic formatting (enhanced failed)"""
        
    except Exception as e:
        logger.error(f"Even basic formatting failed: {e}")
        return f"🚨 **ALERT** - {alert.get('token', 'ERROR')} - {alert.get('alert_type', 'unknown')}"

def build_web3_intelligence_section(ai_data: dict) -> List[str]:
    """Build Web3 intelligence section"""
    if not ai_data:
        return []
    
    web3_lines = []
    
    # Verification
    if ai_data.get('is_verified'):
        web3_lines.append("  ✅ Contract Verified")
    elif ai_data.get('is_verified') is False:
        web3_lines.append("  ⚠️ Unverified Contract")
    
    # Liquidity
    liquidity_usd = ai_data.get('liquidity_usd', 0)
    if ai_data.get('has_liquidity'):
        if liquidity_usd > 100000:
            web3_lines.append(f"  💧 High Liquidity (${liquidity_usd:,.0f})")
        elif liquidity_usd > 10000:
            web3_lines.append(f"  💧 Good Liquidity (${liquidity_usd:,.0f})")
        else:
            web3_lines.append("  💧 Has Liquidity")
    
    # Advanced signals
    if ai_data.get('smart_money_buying') or ai_data.get('has_smart_money'):
        web3_lines.append("  🧠 Smart Money Active")
    
    if ai_data.get('whale_coordination_detected'):
        web3_lines.append("  🐋 Whale Coordination")
    
    if ai_data.get('pump_signals_detected'):
        web3_lines.append("  🚀 Pump Signals")
    
    # Risk assessment
    honeypot_risk = ai_data.get('honeypot_risk', 0)
    if honeypot_risk > 0.7:
        web3_lines.append(f"  🍯 HIGH Honeypot Risk ({honeypot_risk:.0%})")
    elif honeypot_risk > 0.4:
        web3_lines.append(f"  ⚠️ Medium Risk ({honeypot_risk:.0%})")
    
    return web3_lines

# ============================================================================
# TELEGRAM SERVICE (MAIN CLASS)
# ============================================================================

class TelegramService:
    """Main Telegram service with momentum tracking"""
    
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
                    "momentum_tracking": bool(self.momentum_tracker)
                }
                
        except Exception as e:
            return {"configured": False, "error": f"Connection test failed: {str(e)}"}
    
    async def send_start_notification(self, network: str, analysis_type: str, num_wallets: int, 
                                    days_back: float, use_smart_timing: bool, max_tokens: int, 
                                    min_alpha_score: float, store_data: bool = False):
        """Send enhanced analysis start notification - MISSING METHOD ADDED"""
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to send start notification: {e}")
    
    async def send_analysis_notifications(self, result, network: str, max_tokens: int = 7, 
                                        min_alpha_score: float = 50.0, store_alerts: bool = True):
        """Send analysis notifications with momentum tracking"""
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
            
            # Send notifications
            for alert in qualifying_tokens:
                try:
                    # Add momentum data if available
                    if self.momentum_tracker:
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
            
            # Send summary
            await self._send_summary(result, network, len(qualifying_tokens), store_alerts)
            
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
    
    def _format_alert_with_momentum(self, alert: dict) -> str:
        """Format alert with momentum data"""
        try:
            # Start with standard formatting
            message = format_alert_message(alert)
            
            # Add momentum section if available
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
            else:
                # Add first alert note
                lines = message.split('\n')
                contract_index = next((i for i, line in enumerate(lines) 
                                    if '**Contract Address:**' in line), len(lines))
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
    
    async def _send_summary(self, result, network: str, alerts_sent: int, store_alerts: bool):
        """Send analysis summary"""
        network_info = get_network_info(network)
        alert_storage = "Enabled" if store_alerts else "Disabled"
        
        summary = f"""📊 **{result.analysis_type.upper()} ANALYSIS COMPLETE**

✅ **Alerts Sent:** {alerts_sent}
📈 **Total Tokens:** {result.unique_tokens}
💰 **Total ETH:** {result.total_eth_value:.4f}
📊 **Alert Storage:** {alert_storage}

🌐 **Network:** {network_info['name']}
⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
        
        await self.send_message(summary)

# ============================================================================
# GLOBAL INSTANCES AND EXPORTS
# ============================================================================

# Global instances
telegram_client = TelegramClient()
telegram_service = TelegramService()

# Simple notification functions
async def send_alert_notifications(alerts: list):
    """Send alert notifications"""
    if not alerts:
        return
    
    async with telegram_client:
        for alert in alerts:
            try:
                message = format_alert_message(alert)
                await telegram_client.send_message(message)
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")

async def send_test_notification():
    """Send test notification"""
    test_message = f"""🧪 **TEST NOTIFICATION**

✅ Crypto Monitor Active!
🕐 {datetime.now().strftime('%H:%M:%S UTC')}
🚀 All systems operational"""
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

# Export everything
__all__ = [
    'TelegramClient', 'TelegramService', 'telegram_client', 'telegram_service',
    'send_alert_notifications', 'send_test_notification', 'check_notification_config',
    'format_alert_message', 'format_enhanced_alert_message', 'format_basic_alert_message',
    'get_network_info', 'format_contract_address', 'generate_action_links'
]