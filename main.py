import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json
import traceback

# Cloud Functions imports
import functions_framework
from fastapi import Request

# Setup logging first
def setup_logging():
    """Setup logging for Cloud Functions"""
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"}',
        force=True
    )
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Try imports with error handling
try:
    from utils.config import Config
    logger.info("Successfully imported Config")
except Exception as e:
    logger.error(f"Failed to import Config: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

# Global variables
_analyzers = {}
_initialized = False

def initialize_services():
    """Initialize services once - BigQuery only"""
    global _initialized
    if not _initialized:
        logger.info("Initializing Cloud Function services (BigQuery only)...")
        
        try:
            # Validate configuration
            config = Config()
            logger.info(f"Config loaded - BigQuery project: {config.bigquery_project_id}")
            
            errors = config.validate()
            if errors:
                logger.warning(f"Configuration warnings: {', '.join(errors)}")
                # Don't fail on warnings - continue
            
            # Log Telegram configuration status (if configured)
            telegram_configured = check_telegram_config()
            if telegram_configured:
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                logger.info(f"Telegram configured - Bot: {bot_token[:10]}...{bot_token[-5:] if len(bot_token) > 15 else '****'}, Chat: {chat_id}")
            else:
                logger.warning("Telegram notifications not configured - alerts will be disabled")
                logger.warning(f"Bot token present: {bool(os.getenv('TELEGRAM_BOT_TOKEN'))}")
                logger.warning(f"Chat ID present: {bool(os.getenv('TELEGRAM_CHAT_ID'))}")
            
            logger.info(f"BigQuery configured: {bool(config.bigquery_project_id)}")
            logger.info(f"Alchemy configured: {bool(config.alchemy_api_key)}")
            
            logger.info("Services initialized successfully")
            _initialized = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - let the function start but log the error
            _initialized = False

async def get_analyzer(network: str, analysis_type: str):
    """Get or create analyzer instance with error handling"""
    key = f"{network}_{analysis_type}"
    
    if key not in _analyzers:
        try:
            # Import here to avoid circular imports
            logger.info(f"Creating analyzer: {key}")
            
            if analysis_type == 'buy':
                from core.analysis.buy_analyzer import CloudBuyAnalyzer
                analyzer = CloudBuyAnalyzer(network)
            else:
                from core.analysis.sell_analyzer import CloudSellAnalyzer
                analyzer = CloudSellAnalyzer(network)
            
            await analyzer.initialize()
            _analyzers[key] = analyzer
            logger.info(f"Successfully created analyzer: {key}")
            
        except Exception as e:
            logger.error(f"Failed to create analyzer {key}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    return _analyzers[key]

def check_telegram_config() -> bool:
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        return False
    
    # More thorough validation
    if len(bot_token) < 40 or ':' not in bot_token:
        logger.error(f"Invalid bot token format: {bot_token[:10]}...")
        return False
    
    try:
        int(chat_id)  # Chat ID should be numeric
    except ValueError:
        logger.error(f"Invalid chat ID format: {chat_id}")
        return False
    
    return True

async def send_telegram_notification(message: str, parse_mode: str = "Markdown") -> bool:
    """Send a Telegram notification with improved error handling"""
    if not check_telegram_config():
        logger.warning("Telegram not configured - skipping notification")
        return False
    
    try:
        import httpx
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message[:4000],  # Telegram limit
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        logger.info(f"Sending Telegram notification to chat {chat_id}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url, 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {data.get('description', 'Unknown error')}")
                    return False
            else:
                logger.error(f"Telegram HTTP error: {response.status_code} - {response.text}")
                return False
            
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def send_individual_token_notifications(result, network: str, 
                                             max_tokens: int = 7, 
                                             min_alpha_score: float = 50.0) -> bool:
    """Send individual Telegram notifications with DexScreener, Uniswap & X links"""
    if not check_telegram_config():
        logger.warning("Telegram not configured - skipping individual notifications")
        return False
    
    if not result.ranked_tokens:
        logger.info("No tokens to send individual notifications for")
        return True
    
    def create_token_links(contract_address: str, token_symbol: str, network: str) -> Dict[str, str]:
        """Create useful links for each token"""
        links = {}
        
        if contract_address and contract_address != 'Unknown' and len(contract_address) > 10:
            # Clean contract address
            clean_contract = contract_address.lower().strip()
            
            # DexScreener link
            if network.lower() == 'ethereum':
                links['dexscreener'] = f"https://dexscreener.com/ethereum/{clean_contract}"
            elif network.lower() == 'base':
                links['dexscreener'] = f"https://dexscreener.com/base/{clean_contract}"
            else:
                links['dexscreener'] = f"https://dexscreener.com/{network.lower()}/{clean_contract}"
            
            # Uniswap trade link
            if network.lower() == 'ethereum':
                links['uniswap'] = f"https://app.uniswap.org/swap?outputCurrency={clean_contract}&chain=mainnet"
            elif network.lower() == 'base':
                links['uniswap'] = f"https://app.uniswap.org/swap?outputCurrency={clean_contract}&chain=base"
            else:
                links['uniswap'] = f"https://app.uniswap.org/swap?outputCurrency={clean_contract}"
            
            # X (Twitter) search link
            links['twitter'] = f"https://x.com/search?q={clean_contract}&src=typed_query&f=live"
            
            # Etherscan/Basescan link
            if network.lower() == 'ethereum':
                links['explorer'] = f"https://etherscan.io/token/{clean_contract}"
            elif network.lower() == 'base':
                links['explorer'] = f"https://basescan.org/token/{clean_contract}"
            else:
                links['explorer'] = f"#{clean_contract[:10]}..."
        
        return links
    
    try:
        import httpx
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        analysis_type = result.analysis_type.upper()
        emoji = "🟢" if result.analysis_type == "buy" else "🔴"
        action = "BUYING" if result.analysis_type == "buy" else "SELLING"
        
        # FILTER TOKENS by alpha score first
        filtered_tokens = []
        for token, token_data, score in result.ranked_tokens:
            if score >= min_alpha_score:
                filtered_tokens.append((token, token_data, score))
        
        # LIMIT to max_tokens count
        limited_tokens = filtered_tokens[:max_tokens]
        
        logger.info(f"Token filtering: {len(result.ranked_tokens)} total → {len(filtered_tokens)} above {min_alpha_score} score → {len(limited_tokens)} final (max {max_tokens})")
        
        if not limited_tokens:
            logger.info(f"No tokens meet criteria: min_score={min_alpha_score}")
            
            # Send "no alerts" message
            no_alerts_message = f"""
⚪ **NO {analysis_type} ALERTS**

🌐 **Network:** {network.upper()}
📊 **Tokens Found:** {len(result.ranked_tokens)}
🚫 **Above {min_alpha_score} Score:** 0
📈 **Highest Score:** {max([score for _, _, score in result.ranked_tokens[:3]]):.1f}

💡 **Tip:** Lower min_alpha_score to see more alerts

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await send_telegram_notification(no_alerts_message.strip())
            return True
        
        sent_count = 0
        failed_count = 0
        skipped_by_score = len(result.ranked_tokens) - len(filtered_tokens)
        skipped_by_limit = len(filtered_tokens) - len(limited_tokens)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            
            # Send individual message for each qualifying token
            for i, (token, token_data, score) in enumerate(limited_tokens):
                try:
                    # Determine quality emoji based on score
                    if score >= 80:
                        quality_emoji = "🔥🔥🔥"
                    elif score >= 70:
                        quality_emoji = "🔥🔥"
                    elif score >= 60:
                        quality_emoji = "🔥"
                    else:
                        quality_emoji = "⭐"
                    
                    # Get contract address and create links
                    contract_address = token_data.get('contract_address', 'Unknown')
                    links = create_token_links(contract_address, token, network)
                    
                    # Format individual token message with links
                    if result.analysis_type == "buy":
                        message = f"""
{emoji} **{action} ALERT** {quality_emoji}

🪙 **Token:** `{token}`
🌐 **Network:** {network.upper()}
📊 **Alpha Score:** {score:.1f}
💰 **ETH Spent:** {token_data.get('total_eth_spent', 0):.4f}
👥 **Wallets:** {token_data.get('wallet_count', 0)}
🔄 **Purchases:** {token_data.get('total_purchases', 0)}
⭐ **Avg Wallet Score:** {token_data.get('avg_wallet_score', 0):.1f}

📍 **Contract:** `{contract_address[:10]}...`
🏆 **Rank:** #{i+1} of {len(limited_tokens)} alerts

🔗 **Quick Links:**
"""
                    else:  # sell
                        message = f"""
{emoji} **{action} ALERT** {quality_emoji}

🪙 **Token:** `{token}`
🌐 **Network:** {network.upper()}
📊 **Sell Pressure:** {score:.1f}
💰 **ETH Received:** {token_data.get('total_eth_received', 0):.4f}
👥 **Wallets Selling:** {token_data.get('wallet_count', 0)}
🔄 **Sells:** {token_data.get('total_sells', 0)}
⭐ **Avg Wallet Score:** {token_data.get('avg_wallet_score', 0):.1f}

📍 **Contract:** `{contract_address[:10]}...`
🏆 **Rank:** #{i+1} of {len(limited_tokens)} alerts

🔗 **Quick Links:**
"""
                    
                    # Add links if available
                    if links.get('dexscreener'):
                        message += f"📈 [DexScreener]({links['dexscreener']})\n"
                    
                    if links.get('uniswap'):
                        message += f"🦄 [Trade on Uniswap]({links['uniswap']})\n"
                    
                    if links.get('twitter'):
                        message += f"🐦 [Search on X]({links['twitter']})\n"
                    
                    if links.get('explorer'):
                        if network.lower() == 'ethereum':
                            message += f"🔍 [Etherscan]({links['explorer']})\n"
                        elif network.lower() == 'base':
                            message += f"🔍 [Basescan]({links['explorer']})\n"
                        else:
                            message += f"🔍 [Explorer]({links['explorer']})\n"
                    
                    message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                    
                    payload = {
                        "chat_id": chat_id,
                        "text": message.strip(),
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    }
                    
                    response = await client.post(
                        url, 
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('ok'):
                            sent_count += 1
                            logger.info(f"Sent alert for {token} (score: {score:.1f}, rank: #{i+1}) with links")
                        else:
                            logger.error(f"Telegram API error for {token}: {data.get('description')}")
                            failed_count += 1
                    else:
                        logger.error(f"HTTP error for {token}: {response.status_code}")
                        failed_count += 1
                    
                    # Rate limiting - wait between messages
                    await asyncio.sleep(1.5)  # Slightly longer for messages with links
                    
                except Exception as e:
                    logger.error(f"Failed to send notification for {token}: {e}")
                    failed_count += 1
                    continue
        
        # Send summary message with filtering info
        summary_message = f"""
📊 **{analysis_type} ALERTS SUMMARY**

✅ **Alerts Sent:** {sent_count}
❌ **Failed:** {failed_count}
📈 **Total Tokens Found:** {len(result.ranked_tokens)}

🔽 **Filtering Applied:**
• Min Score: {min_alpha_score} (filtered {skipped_by_score})
• Max Count: {max_tokens} (limited {skipped_by_limit})

🔗 **Each alert includes:**
📈 DexScreener charts
🦄 Uniswap trading links  
🐦 X (Twitter) search
🔍 Blockchain explorer

🌐 **Network:** {network.upper()}
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        
        # Send summary
        try:
            summary_payload = {
                "chat_id": chat_id,
                "text": summary_message.strip(),
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(url, json=summary_payload)
                
        except Exception as e:
            logger.error(f"Failed to send summary message: {e}")
        
        logger.info(f"Enhanced notifications complete: {sent_count} sent with links, {failed_count} failed")
        return sent_count > 0
        
    except Exception as e:
        logger.error(f"Failed to send individual token notifications: {e}")
        return False

async def send_bulk_summary_notification(result, network: str) -> bool:
    """Send bulk summary notification"""
    if not check_telegram_config():
        return False
    
    try:
        analysis_type = result.analysis_type.upper()
        emoji = "🟢" if result.analysis_type == "buy" else "🔴"
        
        # Create bulk summary
        message = f"""
{emoji} **{analysis_type} ANALYSIS COMPLETE**

🌐 **Network:** {network.upper()}
📊 **Transactions:** {result.total_transactions}
🪙 **Unique Tokens:** {result.unique_tokens}
💰 **Total ETH:** {result.total_eth_value:.4f}

🏆 **Top 5 Tokens:**
"""
        
        # Add top 5 tokens
        for i, (token, token_data, score) in enumerate(result.ranked_tokens[:5]):
            if result.analysis_type == "buy":
                eth_value = token_data.get('total_eth_spent', 0)
                message += f"{i+1}. `{token}` - {score:.1f} - {eth_value:.3f}Ξ\n"
            else:
                eth_value = token_data.get('total_eth_received', 0)
                message += f"{i+1}. `{token}` - {score:.1f} - {eth_value:.3f}Ξ\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return await send_telegram_notification(message)
        
    except Exception as e:
        logger.error(f"Failed to send bulk summary: {e}")
        return False

async def test_telegram_connection() -> Dict[str, Any]:
    """Test Telegram connection and return detailed status"""
    if not check_telegram_config():
        return {
            "configured": False,
            "error": "Bot token or chat ID missing",
            "bot_token_present": bool(os.getenv('TELEGRAM_BOT_TOKEN')),
            "chat_id_present": bool(os.getenv('TELEGRAM_CHAT_ID'))
        }
    
    try:
        import httpx
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Test bot info
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
                "ready_for_notifications": chat_accessible
            }
            
    except Exception as e:
        return {
            "configured": False,
            "error": f"Connection test failed: {str(e)}",
            "exception": True
        }

async def _run_analysis_with_notifications(request_data: Dict) -> Dict[str, Any]:
    """Enhanced analysis with notification filtering options"""
    try:
        # Extract and validate parameters
        config = Config()
        network = request_data.get('network', 'ethereum')
        analysis_type = request_data.get('analysis_type', 'buy')
        num_wallets = min(int(request_data.get('num_wallets', 50)), config.max_wallets)
        days_back = float(request_data.get('days_back', 1.0))
        debug_mode = request_data.get('debug', False)
        
        # Notification settings
        send_notifications = request_data.get('notifications', True)
        notification_type = request_data.get('notification_type', 'individual')
        
        # NEW: Filtering options with defaults
        max_tokens = int(request_data.get('max_tokens', 7))  # Default: 7 tokens max
        min_alpha_score = float(request_data.get('min_alpha_score', 50.0))  # Default: 50 min score
        
        logger.info(f"Analysis params: {network} {analysis_type}, {num_wallets} wallets, {days_back} days")
        logger.info(f"Notification filters: max_tokens={max_tokens}, min_alpha_score={min_alpha_score}")
        
        # Validate network
        if network not in config.supported_networks:
            error_msg = f'Invalid network: {network}'
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Validate analysis type
        if analysis_type not in config.supported_analysis_types:
            error_msg = f'Invalid analysis_type: {analysis_type}'
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Test Telegram connection if notifications are enabled
        telegram_status = None
        if send_notifications:
            telegram_status = await test_telegram_connection()
            logger.info(f"Telegram status: {telegram_status}")
        
        debug_info = {
            'config_validation': 'passed',
            'requested_params': {
                'network': network,
                'analysis_type': analysis_type,
                'num_wallets': num_wallets,
                'days_back': days_back,
                'max_tokens': max_tokens,
                'min_alpha_score': min_alpha_score
            },
            'telegram_status': telegram_status,
            'notifications_enabled': send_notifications,
            'bigquery_configured': bool(config.bigquery_project_id),
            'alchemy_configured': bool(config.alchemy_api_key)
        }
        
        # If debug mode, return early with config info
        if debug_mode:
            logger.info("Running in debug mode - returning config info")
            return {
                'debug_mode': True,
                'debug_info': debug_info,
                'timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
        
        # Send start notification if enabled
        if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
            start_message = f"""🚀 **ANALYSIS STARTED**

**Network:** {network.upper()}
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
**Time Range:** {days_back} days
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score

⏰ {datetime.now().strftime('%H:%M:%S')}"""
            
            await send_telegram_notification(start_message)
        
        # Run analysis
        try:
            logger.info(f"Starting {analysis_type} analysis for {network}")
            analyzer = await get_analyzer(network, analysis_type)
            result = await analyzer.analyze(num_wallets, days_back)
            
            # Send notifications with filtering
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                
                if notification_type == 'individual':
                    await send_individual_token_notifications(
                        result, network, max_tokens=max_tokens, min_alpha_score=min_alpha_score
                    )
                    
                elif notification_type == 'bulk':
                    await send_bulk_summary_notification(result, network)
                    
                elif notification_type == 'both':
                    # Send filtered individual notifications first
                    await send_individual_token_notifications(
                        result, network, max_tokens=max_tokens, min_alpha_score=min_alpha_score
                    )
                    # Wait then send summary
                    await asyncio.sleep(3)
                    await send_bulk_summary_notification(result, network)
            
            # Add filtering info to result
            result_dict = {
                'network': result.network,
                'analysis_type': result.analysis_type,
                'total_transactions': result.total_transactions,
                'unique_tokens': result.unique_tokens,
                'total_eth_value': result.total_eth_value,
                'top_tokens': result.ranked_tokens[:10],
                'performance_metrics': result.performance_metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'success': True,
                'notification_filters': {
                    'max_tokens': max_tokens,
                    'min_alpha_score': min_alpha_score,
                    'tokens_above_threshold': len([t for t in result.ranked_tokens if t[2] >= min_alpha_score]),
                    'notifications_sent': min(max_tokens, len([t for t in result.ranked_tokens if t[2] >= min_alpha_score]))
                },
                'debug_info': debug_info
            }
            
            logger.info(f"Analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens")
            return result_dict
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Analysis traceback: {traceback.format_exc()}")
            
            # Send error notification if configured
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                try:
                    error_notification = f"❌ **ERROR ({network.upper()})**\n\n"
                    error_notification += f"**Type:** Analysis Error\n"
                    error_notification += f"**Details:** {str(e)[:200]}\n"
                    error_notification += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                    await send_telegram_notification(error_notification)
                except Exception as notify_error:
                    logger.error(f"Failed to send error notification: {notify_error}")
            
            return {
                'error': error_msg,
                'success': False,
                'timestamp': datetime.utcnow().isoformat(),
                'debug_info': debug_info,
                'traceback': traceback.format_exc()
            }
    
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        logger.error(f"Request processing traceback: {traceback.format_exc()}")
        return {
            'error': f"Request processing failed: {str(e)}",
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'traceback': traceback.format_exc()
        }

@functions_framework.http
def crypto_analysis_function(request: Request):
    """Cloud Functions HTTP entry point with filtered notifications"""
    
    logger.info("Function invoked with filtered notification support")
    
    # Initialize services on first request
    try:
        initialize_services()
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        error_response = {
            "error": f"Service initialization failed: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        return (json.dumps(error_response), 500, {'Content-Type': 'application/json'})
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for main response
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        # Handle GET request (health check)
        if request.method == 'GET':
            debug_param = request.args.get('debug', '').lower() == 'true'
            
            basic_response = {
                "message": "Crypto Analysis Function with Filtered Notifications",
                "status": "healthy",
                "version": "5.0.0-filtered-notifications",
                "service": "crypto-analysis-cloud-function",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": _initialized,
                "telegram_configured": check_telegram_config(),
                "database_type": "BigQuery",
                "features": [
                    "Individual token notifications",
                    "Notification filtering by score and count",
                    "Quality emojis",
                    "Rate limiting"
                ]
            }
            
            if debug_param and _initialized:
                try:
                    config = Config()
                    telegram_status = asyncio.run(test_telegram_connection())
                    
                    basic_response['debug_info'] = {
                        'bigquery_configured': bool(config.bigquery_project_id),
                        'alchemy_configured': bool(config.alchemy_api_key),
                        'supported_networks': config.supported_networks,
                        'telegram_detailed': telegram_status,
                        'notification_defaults': {
                            'max_tokens': 7,
                            'min_alpha_score': 50.0,
                            'notification_type': 'individual'
                        }
                    }
                except Exception as debug_error:
                    basic_response['debug_error'] = str(debug_error)
            
            return (json.dumps(basic_response), 200, headers)
        
        # Handle POST request for analysis
        if request.method == 'POST':
            # Get JSON data
            request_json = request.get_json(silent=True)
            if not request_json:
                return (
                    json.dumps({"error": "No JSON data provided"}), 
                    400, 
                    headers
                )
            
            logger.info(f"Running analysis request: {request_json}")
            
            # Run analysis with filtering
            try:
                result = asyncio.run(_run_analysis_with_notifications(request_json))
                status_code = 200 if result.get('success', False) else 500
                return (json.dumps(result), status_code, headers)
                    
            except Exception as e:
                logger.error(f"Analysis request failed: {e}")
                logger.error(f"Analysis request traceback: {traceback.format_exc()}")
                
                error_response = {
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "traceback": traceback.format_exc()
                }
                return (json.dumps(error_response), 500, headers)
        
        # Method not allowed
        return (
            json.dumps({"error": "Method not allowed"}), 
            405, 
            headers
        )
        
    except Exception as e:
        logger.error(f"Function failed: {e}")
        logger.error(f"Function traceback: {traceback.format_exc()}")
        
        error_response = {
            "error": f"Function error: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        return (json.dumps(error_response), 500, headers)

# For local testing
if __name__ == "__main__":
    logger.info("Starting local test with filtered notifications")
    initialize_services()