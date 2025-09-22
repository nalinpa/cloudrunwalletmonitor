from typing import Dict, Set, Any

# Token Exclusions (consolidate from multiple files)
EXCLUDED_TOKENS: Set[str] = frozenset({
    'ETH', 'WETH', 'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'LUSD', 'USDC.E'
})

EXCLUDED_CONTRACTS: Set[str] = frozenset({
    '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
    '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH
    '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC on Base
})

# Currency Conversion Rates (consolidate from data_processor)
SPENDING_CURRENCIES: Dict[str, float] = {
    'ETH': 1.0,
    'WETH': 1.0,
    'USDT': 1/4500,
    'USDC': 1/4500,
    'AERO': 1/4800,
}

RECEIVING_CURRENCIES: Dict[str, float] = {
    'ETH': 1.0,
    'WETH': 1.0,
    'USDT': 1/4500,
    'USDC': 1/4500,
    'USDC.E': 1/4500,
    'DAI': 1/4500,
    'AERO': 1/4800,
    'FRAX': 1/4500,
    'LUSD': 1/4500,
}

# AI Scoring Weights (consolidate from ai_system.py)
AI_SCORING_WEIGHTS: Dict[str, float] = {
    'volume_score': 0.25,
    'wallet_quality': 0.20,
    'momentum': 0.15,
    'liquidity': 0.15,
    'holder_distribution': 0.10,
    'age_factor': 0.08,
    'risk_factor': 0.07
}

# AI Thresholds (consolidate from ai_system.py)
AI_THRESHOLDS: Dict[str, float] = {
    'whale_threshold': 0.7,
    'pump_threshold': 0.65,
    'anomaly_threshold': 0.3,
    'smart_money_threshold': 200,
    'min_holder_count': 50,
    'min_liquidity_eth': 5.0
}

# Notification Settings (consolidate from analysis_handler.py)
NOTIFICATION_SETTINGS: Dict[str, Any] = {
    'default_min_alpha_score': 12.0,
    'max_min_alpha_score': 30.0,
    'default_max_tokens': 12,
    'min_max_tokens': 5,
    'auto_adjust_thresholds': True,
    'send_summary_always': True
}

# Network Display Mappings (consolidate from notifications.py)
NETWORK_DISPLAY: Dict[str, Dict[str, str]] = {
    'ethereum': {
        'name': 'Ethereum',
        'symbol': 'ETH',
        'emoji': '🔷',
        'explorer': 'etherscan.io'
    },
    'base': {
        'name': 'Base',
        'symbol': 'ETH', 
        'emoji': '🔵',
        'explorer': 'basescan.org'
    }
}

# API Endpoints (consolidate scattered URLs)
API_ENDPOINTS: Dict[str, str] = {
    'dexscreener_tokens': 'https://api.dexscreener.com/latest/dex/tokens/',
    'coingecko_contract': 'https://api.coingecko.com/api/v3/coins/ethereum/contract/',
    'etherscan_v2': 'https://api.etherscan.io/v2/api',
    'telegram_api': 'https://api.telegram.org/bot',
    'dextools_base': 'https://www.dextools.io/app/en/',
    'twitter_search': 'https://twitter.com/search?q='
}

# Analysis Limits (consolidate from various configs)
ANALYSIS_LIMITS: Dict[str, Any] = {
    'max_wallets': 200,
    'max_days_back': 7.0,
    'timeout_seconds': 300,
    'batch_size': 10,
    'rate_limit_delay': 0.1,
    'max_retries': 3,
    'min_eth_value': 0.00001
}

# Message Templates (consolidate from notifications.py)
MESSAGE_TEMPLATES: Dict[str, str] = {
    'health_check': "🧪 **TEST NOTIFICATION**\n\n✅ Crypto Monitor is working!\n🕐 {timestamp}",
    'analysis_start': "🚀 **ANALYSIS STARTED**\n\n**Network:** {network}\n**Type:** {analysis_type}\n⏰ {timestamp}",
    'analysis_complete': "📊 **ANALYSIS COMPLETE**\n\n✅ **Alerts:** {alerts_sent}\n📈 **Tokens:** {unique_tokens}\n💰 **ETH:** {total_eth}\n⏰ {timestamp}"
}

# Risk Classifications (consolidate risk assessment logic)
RISK_LEVELS: Dict[str, Dict[str, Any]] = {
    'honeypot_risk': {
        'high': {'threshold': 0.7, 'emoji': '🍯', 'label': 'HIGH Honeypot Risk'},
        'medium': {'threshold': 0.4, 'emoji': '⚠️', 'label': 'Medium Risk'},
        'low': {'threshold': 0.0, 'emoji': '✅', 'label': 'Low Risk'}
    },
    'age_risk': {
        'brand_new': {'hours': 1, 'emoji': '🆕', 'label': 'BRAND NEW'},
        'fresh': {'hours': 24, 'emoji': '⚡', 'label': 'FRESH'},
        'recent': {'hours': 168, 'emoji': '📅', 'label': 'RECENT'},
        'established': {'hours': 720, 'emoji': '✅', 'label': 'ESTABLISHED'},
        'mature': {'hours': float('inf'), 'emoji': '💎', 'label': 'MATURE'}
    },
    'holder_risk': {
        'risky': {'count': 50, 'emoji': '🚨', 'label': 'RISKY'},
        'low': {'count': 200, 'emoji': '⚠️', 'label': 'LOW'},
        'good': {'count': 1000, 'emoji': '✅', 'label': 'GOOD'},
        'strong': {'count': 5000, 'emoji': '💚', 'label': 'STRONG'},
        'wide': {'count': float('inf'), 'emoji': '💎', 'label': 'WIDE'}
    }
}

# Well-known Token Classifications (consolidate heuristics)
TOKEN_CLASSIFICATIONS: Dict[str, Dict[str, Any]] = {
    'major': {
        'tokens': {'WETH', 'USDC', 'USDT', 'DAI', 'ETH'},
        'properties': {'is_verified': True, 'has_liquidity': True, 'honeypot_risk': 0.0}
    },
    'defi': {
        'tokens': {'UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'SUSHI', 'CRV'},
        'properties': {'is_verified': True, 'has_liquidity': True, 'honeypot_risk': 0.1}
    },
    'meme': {
        'tokens': {'PEPE', 'SHIB', 'DOGE', 'FLOKI', 'WIF', 'BONK'},
        'properties': {'is_verified': True, 'has_liquidity': True, 'honeypot_risk': 0.2}
    },
    'l2': {
        'tokens': {'AERO', 'ZORA'},
        'networks': {'base'},
        'properties': {'is_verified': True, 'has_liquidity': True, 'honeypot_risk': 0.1}
    }
}

# Default Values (consolidate all defaults)
DEFAULTS: Dict[str, Any] = {
    'confidence': 0.7,
    'honeypot_risk': 0.3,
    'token_age_hours': 999999,  # Very old
    'holder_count': 0,
    'liquidity_usd': 0,
    'wallet_score': 0,
    'eth_price_usd': 4500,  # Approximate
    'quality_score': 1.0,
    'ai_enhanced': False
}

# Time Windows (consolidate time-based calculations)
TIME_WINDOWS: Dict[str, int] = {
    'momentum_hours': 12,
    'recent_activity_hours': 6,
    'token_age_fresh_hours': 24,
    'token_age_recent_hours': 168,  # 1 week
    'token_age_established_hours': 720,  # 1 month
    'cleanup_interval_seconds': 30,
    'duplicate_window_seconds': 60,
    'retention_days': 5
}

# Feature Flags (consolidate feature toggles)
FEATURES: Dict[str, bool] = {
    'ai_enhancement': True,
    'web3_intelligence': True,
    'momentum_tracking': True,
    'duplicate_prevention': True,
    'smart_timing': True,
    'contract_verification': True,
    'liquidity_checking': True,
    'risk_assessment': True
}