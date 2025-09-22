# utils/web3_utils.py - Consolidated Web3 utilities
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.constants import (
    EXCLUDED_TOKENS, EXCLUDED_CONTRACTS, SPENDING_CURRENCIES, 
    RECEIVING_CURRENCIES, TOKEN_CLASSIFICATIONS, NETWORK_DISPLAY,
    API_ENDPOINTS, DEFAULTS
)

logger = logging.getLogger(__name__)

def extract_contract_address(transfer: Dict) -> str:
    """Single implementation for contract address extraction - consolidates 3+ duplicate functions"""
    contract_address = ""
    
    # Method 1: rawContract.address
    raw_contract = transfer.get("rawContract", {})
    if isinstance(raw_contract, dict) and raw_contract.get("address"):
        contract_address = raw_contract["address"]
    
    # Method 2: contractAddress field
    elif transfer.get("contractAddress"):
        contract_address = transfer["contractAddress"]
    
    # Method 3: 'to' address for ERC20
    elif transfer.get("to"):
        to_address = transfer["to"]
        if to_address != "0x0000000000000000000000000000000000000000":
            contract_address = to_address
    
    # Clean and validate
    if contract_address:
        contract_address = contract_address.strip().lower()
        if not contract_address.startswith('0x'):
            contract_address = '0x' + contract_address
        
        # Validate Ethereum address format
        if len(contract_address) == 42:
            return contract_address
    
    return ""

def is_excluded_token(asset: str, contract_address: str = None) -> bool:
    """Consolidated token exclusion logic - removes duplicates across files"""
    if not asset:
        return True
        
    asset_upper = asset.upper()
    
    # Check excluded tokens
    if asset_upper in EXCLUDED_TOKENS:
        return True
    
    # Check excluded contracts
    if contract_address and contract_address.lower() in EXCLUDED_CONTRACTS:
        return True
    
    # Check stablecoin patterns
    if len(asset) <= 6 and any(stable in asset_upper for stable in ['USD', 'DAI']):
        return True
    
    return False

def calculate_eth_value(transfers: List[Dict], target_tx: str, target_block: str, 
                       currency_map: Dict[str, float]) -> float:
    """Unified ETH calculation for both spending and receiving - consolidates duplicate logic"""
    if not transfers:
        return 0.0
    
    total_eth = 0.0
    
    # Exact transaction match
    for transfer in transfers:
        if transfer.get("hash") == target_tx:
            asset = transfer.get("asset", "")
            if asset in currency_map:
                try:
                    amount = float(transfer.get("value", "0"))
                    total_eth += amount * currency_map[asset]
                except (ValueError, TypeError):
                    continue
    
    if total_eth > 0:
        return total_eth
    
    # Block-based matching for spending
    if currency_map == SPENDING_CURRENCIES:
        return _calculate_block_based_spending(transfers, target_block, currency_map)
    
    # Block proximity matching for receiving  
    else:
        return _calculate_block_based_receiving(transfers, target_block, currency_map)

def _calculate_block_based_spending(transfers: List[Dict], target_block: str, currency_map: Dict[str, float]) -> float:
    """Block-based spending calculation"""
    total_eth = 0.0
    
    for transfer in transfers:
        if transfer.get("blockNum") == target_block:
            asset = transfer.get("asset", "")
            if asset in currency_map:
                try:
                    amount = float(transfer.get("value", "0"))
                    eth_equivalent = amount * currency_map[asset]
                    if 0.0001 <= eth_equivalent <= 50.0:  # Reasonable range
                        total_eth += eth_equivalent
                except (ValueError, TypeError):
                    continue
    
    return total_eth

def _calculate_block_based_receiving(transfers: List[Dict], target_block: str, currency_map: Dict[str, float]) -> float:
    """Block proximity receiving calculation"""
    try:
        target_block_num = int(target_block, 16) if target_block.startswith('0x') else int(target_block)
    except (ValueError, TypeError):
        return 0.0
    
    proximity_values = []
    for transfer in transfers:
        transfer_block = transfer.get("blockNum", "0x0")
        try:
            transfer_block_num = int(transfer_block, 16) if transfer_block.startswith('0x') else int(transfer_block)
            if abs(transfer_block_num - target_block_num) <= 10:  # Within 10 blocks
                asset = transfer.get("asset", "")
                if asset in currency_map:
                    amount = float(transfer.get("value", "0"))
                    eth_equivalent = amount * currency_map[asset]
                    if 0.00001 <= eth_equivalent <= 100.0:  # Reasonable range
                        proximity_values.append(eth_equivalent)
        except (ValueError, TypeError):
            continue
    
    return sum(proximity_values)

def calculate_eth_spent(outgoing_transfers: List[Dict], target_tx: str, target_block: str) -> float:
    """Calculate ETH spent using unified logic"""
    return calculate_eth_value(outgoing_transfers, target_tx, target_block, SPENDING_CURRENCIES)

def calculate_eth_received(incoming_transfers: List[Dict], target_tx: str, target_block: str) -> float:
    """Calculate ETH received using unified logic"""
    return calculate_eth_value(incoming_transfers, target_tx, target_block, RECEIVING_CURRENCIES)

def parse_timestamp(transfer: Dict, block_number: int = None) -> datetime:
    """Unified timestamp parsing - consolidates duplicate logic"""
    if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata']:
        try:
            timestamp_str = transfer['metadata']['blockTimestamp']
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            pass
    
    return datetime.utcnow()

def format_contract_address(contract_address: str) -> str:
    """Unified contract address formatting - consolidates notification logic"""
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
    """Unified action link generation - consolidates notification logic"""
    if not contract_address or len(contract_address) < 10:
        return "❌ No contract address available"
    
    # Get network config
    network_config = NETWORK_DISPLAY.get(network.lower(), NETWORK_DISPLAY['ethereum'])
    
    # Clean contract address
    clean_ca = contract_address.strip().lower()
    if not clean_ca.startswith('0x'):
        clean_ca = '0x' + clean_ca
    
    # Validate address
    if len(clean_ca) != 42:
        return "❌ Invalid contract address"
    
    # Generate links
    links = []
    
    # Uniswap
    if network.lower() == 'base':
        uniswap_url = f"https://app.uniswap.org/#/swap?chain=base&outputCurrency={clean_ca}"
    else:
        uniswap_url = f"https://app.uniswap.org/#/swap?outputCurrency={clean_ca}"
    links.append(f"[🦄 Uniswap]({uniswap_url})")
    
    # DexScreener
    dexscreener_url = f"https://dexscreener.com/{network.lower()}/{clean_ca}"
    links.append(f"[📊 Chart]({dexscreener_url})")
    
    # Block Explorer
    explorer_url = f"https://{network_config['explorer']}/token/{clean_ca}"
    links.append(f"[🔍 Explorer]({explorer_url})")
    
    # Twitter/X search
    twitter_search = f"{API_ENDPOINTS['twitter_search']}{clean_ca}"
    links.append(f"[🐦 Search X]({twitter_search})")
    
    # DEXTools
    dextools_url = f"{API_ENDPOINTS['dextools_base']}{network}/pair-explorer/{clean_ca}"
    links.append(f"[🔧 DEXTools]({dextools_url})")
    
    return " | ".join(links)

def get_network_info(network: str) -> Dict[str, str]:
    """Get network information - consolidates scattered network configs"""
    return NETWORK_DISPLAY.get(network.lower(), NETWORK_DISPLAY['ethereum'])

def apply_token_heuristics(token_symbol: str, network: str = 'ethereum') -> Dict[str, Any]:
    """Apply heuristic analysis for well-known tokens - consolidates classification logic"""
    symbol_upper = token_symbol.upper()
    
    # Check each classification
    for classification, config in TOKEN_CLASSIFICATIONS.items():
        tokens = config.get('tokens', set())
        networks = config.get('networks', set())
        
        # Token match
        if symbol_upper in tokens:
            # Network-specific check if specified
            if networks and network.lower() not in networks:
                continue
            
            result = config['properties'].copy()
            result['heuristic_classification'] = classification
            result['token_symbol'] = token_symbol
            result['network'] = network
            return result
    
    # Default for unknown tokens
    return {
        'is_verified': False,
        'has_liquidity': False,
        'honeypot_risk': DEFAULTS['honeypot_risk'],
        'heuristic_classification': 'unknown',
        'token_symbol': token_symbol,
        'network': network
    }

def create_default_web3_intelligence(token_symbol: str, network: str) -> Dict[str, Any]:
    """Create default Web3 intelligence structure - consolidates default creation"""
    return {
        'contract_address': '',
        'ca': '',
        'token_symbol': token_symbol,
        'network': network,
        'is_verified': False,
        'has_liquidity': False,
        'liquidity_usd': DEFAULTS['liquidity_usd'],
        'honeypot_risk': DEFAULTS['honeypot_risk'],
        'smart_money_buying': False,
        'whale_accumulation': False,
        'data_sources': [],
        'error': 'no_data_available'
    }

def validate_ethereum_address(address: str) -> bool:
    """Validate Ethereum address format - consolidates validation logic"""
    if not address:
        return False
    
    # Clean address
    clean_address = address.strip().lower()
    if not clean_address.startswith('0x'):
        clean_address = '0x' + clean_address
    
    # Check length and hex format
    if len(clean_address) != 42:
        return False
    
    try:
        # Verify it's valid hex
        int(clean_address[2:], 16)
        return True
    except ValueError:
        return False

def format_token_age(hours: Optional[float]) -> str:
    """Format token age with risk indicators - consolidates age formatting"""
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

def format_holder_count(count: Optional[int]) -> str:
    """Format holder count with risk indicators - consolidates holder formatting"""
    if count is None:
        return "👥 Holders: Unknown"
    
    if count < 50:
        return f"👥 Holders: 🚨 {count} (RISKY)"
    elif count < 200:
        return f"👥 Holders: ⚠️ {count} (LOW)"
    elif count < 1000:
        return f"👥 Holders: ✅ {count:,} (GOOD)"
    elif count < 5000:
        return f"👥 Holders: 💚 {count:,} (STRONG)"
    else:
        return f"👥 Holders: 💎 {count:,} (WIDE)"

def get_combined_risk_indicator(age_hours: Optional[float], holders: Optional[int]) -> Optional[str]:
    """Get combined risk assessment - consolidates risk logic"""
    risk_signals = []
    
    if age_hours is not None and age_hours < 24:
        risk_signals.append("NEW")
    if holders is not None and holders < 50:
        risk_signals.append("FEW HOLDERS")
        
    if len(risk_signals) >= 2:
        return "🚨 HIGH RISK: " + " + ".join(risk_signals)
    elif len(risk_signals) == 1:
        return "⚠️ CAUTION: " + risk_signals[0]
    elif age_hours is not None and holders is not None:
        if age_hours > 720 and holders > 1000:
            return "✅ LOW RISK: Established + Wide distribution"
        elif age_hours > 168 and holders > 200:
            return "💡 MODERATE RISK: Growing project"
    
    return None

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float - consolidates type conversion"""
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int - consolidates type conversion"""
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_bool_conversion(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool - consolidates type conversion"""
    if value is None:
        return default
    
    # Handle numpy bool types and other variations
    try:
        return bool(value)
    except (ValueError, TypeError):
        return default

def extract_web3_data_safely(web3_analysis: Optional[Dict], field: str, default: Any = None) -> Any:
    """Safely extract field from web3_analysis dict - consolidates data extraction"""
    if not web3_analysis or not isinstance(web3_analysis, dict):
        return default
    
    return web3_analysis.get(field, default)

def build_token_metadata(token_symbol: str, contract_address: str, network: str, web3_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Build standardized token metadata - consolidates metadata creation"""
    return {
        'token_symbol': token_symbol,
        'contract_address': contract_address,
        'ca': contract_address,  # Alternative field name
        'network': network,
        'network_info': get_network_info(network),
        'is_valid_address': validate_ethereum_address(contract_address),
        'web3_data': web3_data or {},
        'metadata_created_at': datetime.utcnow().isoformat()
    }

def log_token_processing(token: str, operation: str, details: str = ""):
    """Standardized token processing logging - consolidates logging patterns"""
    ca_display = details[:10] + '...' if len(details) > 10 else details
    logger.info(f"🪙 {token}: {operation} {ca_display}")

# Batch processing utilities - consolidates batch logic
def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks for batch processing"""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def extract_unique_tokens_info(purchases: List) -> Dict[str, Dict[str, Any]]:
    """Extract unique token information for batch processing - consolidates token extraction"""
    unique_tokens = {}
    
    for purchase in purchases:
        token = purchase.token_bought
        if token not in unique_tokens:
            # Extract basic info
            contract_address = ""
            network = "ethereum"  # Default
            
            if hasattr(purchase, 'web3_analysis') and purchase.web3_analysis:
                contract_address = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                network = purchase.web3_analysis.get('network', 'ethereum')
            
            unique_tokens[token] = {
                'contract_address': contract_address,
                'network': network,
                'token_symbol': token,
                'purchase_count': 0,
                'total_eth_value': 0,
                'purchases': []
            }
        
        # Aggregate stats
        unique_tokens[token]['purchase_count'] += 1
        unique_tokens[token]['purchases'].append(purchase)
        
        # Add ETH value
        if hasattr(purchase, 'eth_spent'):
            unique_tokens[token]['total_eth_value'] += purchase.eth_spent
        elif hasattr(purchase, 'amount_received'):
            unique_tokens[token]['total_eth_value'] += purchase.amount_received
    
    return unique_tokens