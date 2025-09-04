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
    """Initialize services once with better error handling"""
    global _initialized
    if not _initialized:
        logger.info("Initializing Cloud Function services...")
        
        try:
            # Validate configuration
            config = Config()
            logger.info(f"Config loaded - BigQuery project: {getattr(config, 'bigquery_project_id', 'Not set')}")
            
            errors = config.validate()
            if errors:
                logger.warning(f"Configuration warnings: {', '.join(errors)}")
                # Don't fail on warnings - continue
            
            # Log Telegram configuration status
            telegram_configured = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
            if telegram_configured:
                logger.info("Telegram notifications configured")
            else:
                logger.warning("Telegram notifications not configured - alerts will be disabled")
            
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
    return bool(bot_token and chat_id and len(bot_token) > 40)

async def send_telegram_notification(message: str) -> bool:
    """Send a basic Telegram notification"""
    if not check_telegram_config():
        return False
    
    try:
        import httpx
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message[:4000],
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            success = response.status_code == 200
            
            if success:
                logger.info("Telegram notification sent successfully")
            else:
                logger.error(f"Telegram API error: {response.status_code}")
            
            return success
            
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False

async def _run_analysis_with_notifications(request_data: Dict) -> Dict[str, Any]:
    """Run analysis and send notifications with better error handling"""
    try:
        # Extract and validate parameters
        config = Config()
        network = request_data.get('network', 'ethereum')
        analysis_type = request_data.get('analysis_type', 'buy')
        num_wallets = min(int(request_data.get('num_wallets', 50)), config.max_wallets)
        days_back = float(request_data.get('days_back', 1.0))
        debug_mode = request_data.get('debug', False)
        send_notifications = request_data.get('notifications', True)
        
        logger.info(f"Running analysis: {network} {analysis_type} with {num_wallets} wallets")
        
        # Validate network
        if network not in config.supported_networks:
            error_msg = f'Invalid network: {network}'
            logger.error(error_msg)
            if send_notifications and check_telegram_config():
                try:
                    await send_telegram_notification(f"❌ **ERROR**\n\n**Type:** Configuration Error\n**Details:** {error_msg}")
                except:
                    pass
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
        
        debug_info = {
            'config_validation': 'passed',
            'requested_params': {
                'network': network,
                'analysis_type': analysis_type,
                'num_wallets': num_wallets,
                'days_back': days_back
            },
            'notifications_configured': check_telegram_config(),
            'notifications_enabled': send_notifications
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
        
        # Run normal analysis
        try:
            logger.info(f"Starting {analysis_type} analysis for {network}")
            analyzer = await get_analyzer(network, analysis_type)
            result = await analyzer.analyze(num_wallets, days_back)
            
            # Convert to dict for JSON serialization
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
                'debug_info': debug_info
            }
            
            logger.info(f"Analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens")
            return result_dict
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Analysis traceback: {traceback.format_exc()}")
            
            # Send error notification if configured
            if send_notifications and check_telegram_config():
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
    """Cloud Functions HTTP entry point with comprehensive error handling"""
    
    logger.info("Function invoked")
    
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
                "message": "Crypto Analysis Function with BigQuery Storage",
                "status": "healthy",
                "version": "3.0.0-bigquery-debug",
                "service": "crypto-analysis-cloud-function",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": _initialized,
                "telegram_configured": check_telegram_config()
            }
            
            if debug_param and _initialized:
                # Add detailed config info for debugging
                try:
                    config = Config()
                    basic_response['debug_info'] = {
                        'mongo_configured': bool(config.mongo_uri),
                        'alchemy_configured': bool(config.alchemy_api_key),
                        'bigquery_configured': bool(getattr(config, 'bigquery_project_id', None)),
                        'bigquery_project': getattr(config, 'bigquery_project_id', 'Not configured'),
                        'bigquery_location': getattr(config, 'bigquery_location', 'Not configured'),
                        'supported_networks': config.supported_networks,
                        'db_name': config.db_name,
                        'wallets_collection': config.wallets_collection
                    }
                except Exception as debug_error:
                    basic_response['debug_error'] = str(debug_error)
            
            return (json.dumps(basic_response), 200, headers)
        
        # Handle POST request (analysis)
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
    logger.info("Starting local test")
    initialize_services()