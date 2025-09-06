import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json
import traceback

# Cloud Functions imports
import functions_framework
from flask import Request 

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
            
            # Safely log config attributes if they exist
            logger.info(f"BigQuery project: {getattr(config, 'bigquery_project_id', 'Not configured')}")
            
            # Check for dataset attribute safely
            dataset = getattr(config, 'bigquery_dataset', getattr(config, 'dataset', 'crypto_analysis'))
            logger.info(f"BigQuery dataset: {dataset}")
            
            # Check for location attribute safely  
            location = getattr(config, 'bigquery_location', getattr(config, 'location', 'asia-southeast1'))
            logger.info(f"BigQuery location: {location}")
            
            # Check for cloud function project safely
            cf_project = getattr(config, 'cloud_function_project', 'None')
            logger.info(f"Cloud Function project: {cf_project}")
            
            logger.info(f"Config loaded - BigQuery project: {getattr(config, 'bigquery_project_id', 'Not configured')}")
            
            # Log Telegram configuration status (if configured)
            telegram_configured = check_telegram_config()
            if telegram_configured:
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                logger.info(f"Telegram configured - Bot: {bot_token[:10]}...{bot_token[-5:] if len(bot_token) > 15 else '****'}, Chat: {chat_id}")
            
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

async def send_analysis_notifications(result, network: str, max_tokens: int = 7, min_alpha_score: float = 50.0):
    """Send notifications for analysis results"""
    try:
        if not result.ranked_tokens:
            await send_telegram_notification(f"📊 **Analysis Complete** - No tokens found for {network.upper()}")
            return
        
        # Filter tokens by score
        qualifying_tokens = []
        for token_data in result.ranked_tokens:
            if len(token_data) >= 3 and token_data[2] >= min_alpha_score:
                qualifying_tokens.append(token_data)
        
        limited_tokens = qualifying_tokens[:max_tokens]
        
        if limited_tokens:
            # Send individual notifications
            for i, token_tuple in enumerate(limited_tokens):
                token = token_tuple[0]
                data = token_tuple[1] 
                score = token_tuple[2]
                
                if result.analysis_type == "buy":
                    message = f"""🟢 **BUY ALERT**

🪙 **Token:** `{token}`
📊 **Score:** {score:.1f}
🌐 **Network:** {network.upper()}

💰 **ETH Spent:** {data.get('total_eth_spent', 0):.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🔄 **Purchases:** {data.get('total_purchases', 0)}
⭐ **Avg Wallet Score:** {data.get('avg_wallet_score', 0):.1f}

🏆 **Rank:** #{i+1} of {len(limited_tokens)}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                else:  # sell
                    message = f"""🔴 **SELL ALERT**

🪙 **Token:** `{token}`
📊 **Sell Pressure:** {score:.1f}
🌐 **Network:** {network.upper()}

💰 **ETH Received:** {data.get('total_eth_received', 0):.4f}
👥 **Wallets Selling:** {data.get('wallet_count', 0)}
🔄 **Sells:** {data.get('total_sells', 0)}
⭐ **Avg Wallet Score:** {data.get('avg_wallet_score', 0):.1f}

🏆 **Rank:** #{i+1} of {len(limited_tokens)}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
                
                await send_telegram_notification(message.strip())
                await asyncio.sleep(1.5)  # Rate limiting
            
            # Send summary
            summary_message = f"""📊 **{result.analysis_type.upper()} ANALYSIS SUMMARY**

✅ **Alerts Sent:** {len(limited_tokens)}
📈 **Total Tokens Found:** {result.unique_tokens}
💰 **Total ETH Volume:** {result.total_eth_value:.4f}
🔍 **Filtering:** min score {min_alpha_score}, max {max_tokens} tokens

🌐 **Network:** {network.upper()}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
            
            await send_telegram_notification(summary_message.strip())
            logger.info(f"Sent {len(limited_tokens)} notifications for {network}")
            
        else:
            # No qualifying tokens
            max_score = max([t[2] for t in result.ranked_tokens[:3]]) if result.ranked_tokens else 0
            message = f"""📊 **{result.analysis_type.upper()} ANALYSIS - NO ALERTS**

🌐 **Network:** {network.upper()}
📊 **Tokens Found:** {result.unique_tokens}
🚫 **Above {min_alpha_score} Score:** 0
📈 **Highest Score:** {max_score:.1f}

💡 **Tip:** Lower min_alpha_score to see more alerts
⏰ {datetime.now().strftime('%H:%M:%S')}"""
            
            await send_telegram_notification(message.strip())
            logger.info(f"No qualifying tokens found for {network} (max score: {max_score:.1f})")
            
    except Exception as e:
        logger.error(f"Failed to send analysis notifications: {e}")
        await send_telegram_notification(f"❌ **Notification Error** - Analysis completed but failed to send alerts: {str(e)}")

async def _run_analysis_with_notifications(request_data: Dict) -> Dict[str, Any]:
    """Enhanced analysis with AI notification support"""
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
        
        # Filtering options with defaults
        max_tokens = int(request_data.get('max_tokens', 7))
        min_alpha_score = float(request_data.get('min_alpha_score', 50.0))
        
        logger.info(f"Enhanced analysis params: {network} {analysis_type}, {num_wallets} wallets, {days_back} days")
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
            'alchemy_configured': bool(config.alchemy_api_key),
            'ai_enhancement': 'pandas-ta_available'
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
            start_message = f"""🚀 **ENHANCED ANALYSIS STARTED**

**Network:** {network.upper()}
**Type:** {analysis_type.capitalize()}
**Wallets:** {num_wallets}
**Time Range:** {days_back} days
**AI Enhancement:** pandas-ta Enabled
**Filters:** max {max_tokens} tokens, ≥{min_alpha_score} score

⏰ {datetime.now().strftime('%H:%M:%S')}"""
            
            await send_telegram_notification(start_message)
        
        # Run enhanced analysis
        try:
            logger.info(f"Starting enhanced {analysis_type} analysis for {network}")
            analyzer = await get_analyzer(network, analysis_type)
            result = await analyzer.analyze(num_wallets, days_back)
            
            # Enhanced result
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
                'ai_enhancement': 'pandas-ta',
                'debug_info': debug_info
            }
            
            logger.info(f"Enhanced analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens")
            
            # Send notifications after analysis completes
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                try:
                    await send_analysis_notifications(result, network, max_tokens, min_alpha_score)
                except Exception as e:
                    logger.error(f"Failed to send notifications: {e}")
            
            return result_dict
            
        except Exception as e:
            error_msg = f"Enhanced analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Analysis traceback: {traceback.format_exc()}")
            
            return {
                'error': error_msg,
                'success': False,
                'timestamp': datetime.utcnow().isoformat(),
                'debug_info': debug_info,
                'traceback': traceback.format_exc()
            }
    
    except Exception as e:
        logger.error(f"Enhanced request processing failed: {e}")
        logger.error(f"Request processing traceback: {traceback.format_exc()}")
        return {
            'error': f"Enhanced request processing failed: {str(e)}",
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'traceback': traceback.format_exc()
        }

@functions_framework.http
def main(request: Request):
    """Cloud Functions HTTP entry point"""
    
    logger.info("🚀 Cloud Function started with pandas-ta support")
    
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
            basic_response = {
                "message": "Crypto Analysis Function with pandas-ta",
                "status": "healthy",
                "version": "6.0.0-pandas-ta",
                "service": "crypto-analysis-cloud-function",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": _initialized,
                "telegram_configured": check_telegram_config(),
                "database_type": "BigQuery",
                "ai_library": "pandas-ta",
                "features": [
                    "pandas-ta Technical Analysis (15+ indicators)",
                    "ML Anomaly Detection",
                    "Sentiment Analysis",
                    "Whale Coordination Detection",
                    "Smart Money Flow Analysis",
                    "Telegram notifications"
                ]
            }
            
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
            
            # Run analysis
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
    logger.info("Starting local test with pandas-ta support")
    initialize_services()
    print("Function ready for local testing. Send requests to test endpoints.")