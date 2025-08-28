import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Cloud Functions imports
import functions_framework
from flask import Request

# Your existing imports
from utils.logger import setup_logging
from utils.config import Config

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Global variables
_analyzers = {}
_initialized = False

def initialize_services():
    """Initialize services once"""
    global _initialized
    if not _initialized:
        logger.info("Initializing Cloud Function services...")
        
        # Validate configuration
        config = Config()
        errors = config.validate()
        if errors:
            logger.error(f"Configuration errors: {', '.join(errors)}")
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Log Telegram configuration status
        telegram_configured = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
        if telegram_configured:
            logger.info("✅ Telegram notifications configured")
        else:
            logger.warning("⚠️ Telegram notifications not configured - alerts will be disabled")
        
        logger.info("Services initialized successfully")
        _initialized = True

async def get_analyzer(network: str, analysis_type: str):
    """Get or create analyzer instance"""
    key = f"{network}_{analysis_type}"
    
    if key not in _analyzers:
        try:
            # Import here to avoid circular imports
            from core.analysis.buy_analyzer import CloudBuyAnalyzer
            from core.analysis.sell_analyzer import CloudSellAnalyzer
            
            if analysis_type == 'buy':
                analyzer = CloudBuyAnalyzer(network)
            else:
                analyzer = CloudSellAnalyzer(network)
            
            await analyzer.initialize()
            _analyzers[key] = analyzer
            logger.info(f"Created analyzer: {key}")
            
        except Exception as e:
            logger.error(f"Failed to create analyzer {key}: {e}")
            raise
    
    return _analyzers[key]

def check_telegram_config() -> bool:
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    return bool(bot_token and chat_id and len(bot_token) > 40)

def get_telegram_status() -> Dict[str, Any]:
    """Get Telegram configuration status"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return {
        'configured': check_telegram_config(),
        'bot_token_length': len(bot_token) if bot_token else 0,
        'chat_id_set': bool(chat_id)
    }

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

async def send_analysis_notification(result: Dict[str, Any]) -> int:
    """Send analysis notifications via Telegram"""
    if not check_telegram_config():
        logger.info("Telegram not configured - skipping notifications")
        return 0
    
    notifications_sent = 0
    
    try:
        # Create analysis summary message
        network = result.get('network', 'unknown').upper()
        analysis_type = result.get('analysis_type', 'analysis')
        total_transactions = result.get('total_transactions', 0)
        unique_tokens = result.get('unique_tokens', 0)
        total_eth_value = result.get('total_eth_value', 0)
        top_tokens = result.get('top_tokens', [])
        
        emoji = "🔍" if analysis_type == "buy" else "📉"
        type_text = analysis_type.replace('_', ' ').upper()
        
        # Summary message
        summary_msg = f"{emoji} **{type_text} ANALYSIS**\n\n"
        summary_msg += f"🌐 **{network}** | {total_transactions:,} txs | {unique_tokens} tokens\n"
        summary_msg += f"💰 **Total:** {total_eth_value:.4f} ETH\n"
        
        if top_tokens and len(top_tokens) > 0:
            summary_msg += f"\n**Top Tokens:**\n"
            for i, (token, data, score) in enumerate(top_tokens[:3], 1):
                eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
                wallet_count = data.get('wallet_count', 0)
                summary_msg += f"{i}. `{token}` - {score:.1f} | {eth_value:.3f} ETH | {wallet_count}W\n"
        
        summary_msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        # Send summary
        if await send_telegram_notification(summary_msg):
            notifications_sent += 1
        
        # Send individual token alerts for high-scoring tokens
        if top_tokens:
            for token, data, score in top_tokens[:10]: 
                # Check if token meets alert criteria
                eth_value = data.get('total_eth_spent', data.get('total_eth_value', 0))
                wallet_count = data.get('wallet_count', 0)
                
                # Basic thresholds
                if score >= 25.0 and eth_value >= 0.05 and wallet_count >= 1:
                    # Create token alert
                    contract = data.get('contract_address', '')
                    
                    if 'alpha_score' in data:
                        emoji = "🆕"
                        alert_type = "NEW TOKEN"
                    else:
                        emoji = "📉" 
                        alert_type = "SELL PRESSURE"
                    
                    alert_msg = f"{emoji} **{alert_type}**\n\n"
                    alert_msg += f"🪙 **Token:** `{token}`\n"
                    alert_msg += f"🌐 **Network:** {network}\n"
                    alert_msg += f"📊 **Score:** {score:.1f}\n"
                    alert_msg += f"💰 **ETH:** {eth_value:.4f}\n"
                    alert_msg += f"👥 **Wallets:** {wallet_count}\n"
                    
                    # Add trading links if contract available
                    if contract and len(contract) > 10:
                        alert_msg += f"📝 **Contract:** `{contract[:6]}...{contract[-4:]}`\n"
                        
                        # Add links based on network
                        if network.lower() == 'ethereum':
                            dex_link = f"https://dexscreener.com/ethereum/{contract}"
                            uni_link = f"https://app.uniswap.org/swap?outputCurrency={contract}"
                            explorer_link = f"https://etherscan.io/token/{contract}"
                        elif network.lower() == 'base':
                            dex_link = f"https://dexscreener.com/base/{contract}"
                            uni_link = f"https://app.uniswap.org/swap?outputCurrency={contract}&chain=base"
                            explorer_link = f"https://basescan.org/token/{contract}"
                        else:
                            dex_link = uni_link = explorer_link = None
                        
                        if dex_link:
                            alert_msg += f"\n📊 [DEXScreener]({dex_link}) | 🦄 [Uniswap]({uni_link})\n"
                            alert_msg += f"🔍 [Explorer]({explorer_link})"
                    
                    alert_msg += f"\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                    
                    # Send token alert
                    if await send_telegram_notification(alert_msg):
                        notifications_sent += 1
                        logger.info(f"Alert sent for token {token}")
                    
                    # Rate limiting
                    await asyncio.sleep(2)
        
        logger.info(f"Sent {notifications_sent} Telegram notifications")
        return notifications_sent
        
    except Exception as e:
        logger.error(f"Failed to send analysis notifications: {e}")
        return notifications_sent

async def _debug_database_connection(config: Config, network: str) -> Dict:
    """Debug database connection and data"""
    debug_info = {}
    
    try:
        from services.database.database_client import DatabaseService
        db_service = DatabaseService(config)
        await db_service.initialize()
        
        if db_service._connected and db_service.db is not None:
            debug_info['database_connected'] = True
            
            collections = await db_service.db.list_collection_names()
            debug_info['collections'] = collections
            debug_info['wallets_collection_exists'] = config.wallets_collection in collections
            
            if config.wallets_collection in collections:
                collection = db_service.db[config.wallets_collection]
                
                # Count total documents
                total_count = await collection.count_documents({})
                debug_info['total_wallets'] = total_count
                
                # Check field existence
                field_counts = {}
                for field in ['address', 'wallet_address', 'wallet', 'score', 'rating']:
                    try:
                        field_counts[field] = await collection.count_documents({field: {"$exists": True}})
                    except Exception as e:
                        field_counts[field] = f"Error: {str(e)}"
                
                debug_info['field_existence'] = field_counts
                
                # Try getting wallets using the service
                try:
                    wallets = await db_service.get_top_wallets(network, 10)
                    debug_info['fetched_wallets_count'] = len(wallets)
                except Exception as e:
                    debug_info['wallet_fetch_error'] = str(e)
            
        else:
            debug_info['database_connected'] = False
            debug_info['error'] = 'Database connection failed'
            
        await db_service.cleanup()
        
    except Exception as e:
        debug_info['database_error'] = str(e)
        debug_info['database_connected'] = False
        
        import traceback
        debug_info['database_traceback'] = traceback.format_exc()
    
    return debug_info

async def _debug_alchemy_connection(config: Config, network: str) -> Dict:
    """Debug Alchemy API connection"""
    debug_info = {}
    
    try:
        from services.blockchain.alchemy_client import AlchemyService
        alchemy_service = AlchemyService(config)
        
        # Check configuration first
        base_url = config.alchemy_endpoints.get(network)
        debug_info['alchemy_url_configured'] = bool(base_url)
        debug_info['api_key_configured'] = bool(config.alchemy_api_key)
        debug_info['api_key_length'] = len(config.alchemy_api_key) if config.alchemy_api_key else 0
        
        if not base_url or not config.alchemy_api_key:
            debug_info['alchemy_api_responsive'] = False
            debug_info['alchemy_error'] = "Missing API URL or key"
            return debug_info
        
        # Test block range retrieval
        start_block, end_block = await alchemy_service.get_block_range(network, 1.0)
        debug_info['block_range_success'] = start_block > 0 and end_block > 0
        debug_info['start_block'] = start_block
        debug_info['end_block'] = end_block
        debug_info['blocks_in_range'] = end_block - start_block if end_block > start_block else 0
        
        if start_block > 0:
            debug_info['alchemy_api_responsive'] = True
        else:
            debug_info['alchemy_api_responsive'] = False
            debug_info['alchemy_error'] = "Block range retrieval failed"
        
    except Exception as e:
        debug_info['alchemy_error'] = str(e)
        debug_info['alchemy_api_responsive'] = False
        
        import traceback
        debug_info['alchemy_traceback'] = traceback.format_exc()
    
    return debug_info

async def _run_analysis_with_notifications(request_data: Dict) -> Dict[str, Any]:
    """Run analysis and send notifications"""
    # Extract and validate parameters
    config = Config()
    network = request_data.get('network', 'ethereum')
    analysis_type = request_data.get('analysis_type', 'buy')
    num_wallets = min(int(request_data.get('num_wallets', 50)), config.max_wallets)
    days_back = float(request_data.get('days_back', 1.0))
    debug_mode = request_data.get('debug', False)
    send_notifications = request_data.get('notifications', True)
    
    # Validate
    if network not in config.supported_networks:
        error_msg = f'Invalid network: {network}'
        if send_notifications and check_telegram_config():
            await send_telegram_notification(f"❌ **ERROR**\n\n**Type:** Configuration Error\n**Details:** {error_msg}")
        raise ValueError(error_msg)
    
    if analysis_type not in config.supported_analysis_types:
        error_msg = f'Invalid analysis_type: {analysis_type}'
        if send_notifications and check_telegram_config():
            await send_telegram_notification(f"❌ **ERROR**\n\n**Type:** Configuration Error\n**Details:** {error_msg}")
        raise ValueError(error_msg)
    
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
    
    # If debug mode, run diagnostics
    if debug_mode:
        logger.info("Running in debug mode")
        debug_info['database_debug'] = await _debug_database_connection(config, network)
        debug_info['alchemy_debug'] = await _debug_alchemy_connection(config, network)
        
        # Send debug status notification if configured
        if send_notifications and check_telegram_config():
            db_ok = debug_info['database_debug'].get('database_connected', False)
            api_ok = debug_info['alchemy_debug'].get('alchemy_api_responsive', False)
            
            status_msg = f"✅ **SYSTEM: {'HEALTHY' if (db_ok and api_ok) else 'ISSUES'}**\n\n"
            status_msg += f"Database: {'✅' if db_ok else '❌'}\n"
            status_msg += f"Alchemy: {'✅' if api_ok else '❌'}\n"
            
            if debug_info['database_debug'].get('total_wallets'):
                status_msg += f"Wallets: {debug_info['database_debug']['total_wallets']:,}\n"
            
            status_msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
            
            await send_telegram_notification(status_msg)
        
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
        
        # Convert to dict for JSON serialization and notifications
        result_dict = {
            'network': result.network,
            'analysis_type': result.analysis_type,
            'total_transactions': result.total_transactions,
            'unique_tokens': result.unique_tokens,
            'total_eth_value': result.total_eth_value,
            'top_tokens': result.ranked_tokens[:10],  # Limit for notifications
            'performance_metrics': result.performance_metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'debug_info': debug_info
        }
        
        # Send notifications if enabled and configured
        notifications_sent = 0
        if send_notifications and check_telegram_config():
            try:
                notifications_sent = await send_analysis_notification(result_dict)
                logger.info(f"Sent {notifications_sent} notifications")
                
            except Exception as e:
                logger.error(f"Failed to send notifications: {e}")
                # Don't fail the analysis due to notification errors
        
        result_dict['notifications_sent'] = notifications_sent
        return result_dict
        
    except Exception as e:
        debug_info['analysis_error'] = str(e)
        logger.error(f"Analysis failed: {e}")
        
        # Send error notification if configured
        if send_notifications and check_telegram_config():
            try:
                error_msg = f"❌ **ERROR ({network.upper()})**\n\n"
                error_msg += f"**Type:** Analysis Error\n"
                error_msg += f"**Details:** {str(e)[:200]}\n"
                error_msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                await send_telegram_notification(error_msg)
            except Exception as notify_error:
                logger.error(f"Failed to send error notification: {notify_error}")
        
        # Add full traceback for debugging
        import traceback
        debug_info['analysis_traceback'] = traceback.format_exc()
        
        return {
            'error': str(e),
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'debug_info': debug_info,
            'notifications_sent': 0
        }

@functions_framework.http
def crypto_analysis_function(request: Request):
    """Cloud Functions HTTP entry point with Telegram notifications"""
    
    # Initialize services on first request
    try:
        initialize_services()
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        error_response = {
            "error": f"Service initialization failed: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
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
        # Handle GET request (health check with Telegram status)
        if request.method == 'GET':
            debug_param = request.args.get('debug', '').lower() == 'true'
            
            basic_response = {
                "message": "Crypto Analysis Function with Telegram Notifications",
                "status": "healthy",
                "version": "2.3.0-telegram-fixed",
                "service": "crypto-analysis-cloud-function",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": _initialized,
                "telegram_configured": check_telegram_config()
            }
            
            if debug_param:
                # Add detailed config info for debugging
                config = Config()
                telegram_status = get_telegram_status()
                
                basic_response['debug_info'] = {
                    'mongo_configured': bool(config.mongo_uri),
                    'alchemy_configured': bool(config.alchemy_api_key),
                    'supported_networks': config.supported_networks,
                    'db_name': config.db_name,
                    'wallets_collection': config.wallets_collection,
                    'mongo_uri_length': len(config.mongo_uri) if config.mongo_uri else 0,
                    'alchemy_key_length': len(config.alchemy_api_key) if config.alchemy_api_key else 0,
                    'telegram_status': telegram_status
                }
            
            return (json.dumps(basic_response), 200, headers)
        
        # Handle POST request (analysis with notifications)
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
            
            # Run analysis with notifications
            try:
                result = asyncio.run(_run_analysis_with_notifications(request_json))
                return (json.dumps(result), 200, headers)
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                
                # Try to send error notification
                if check_telegram_config():
                    try:
                        error_msg = f"❌ **CRITICAL ERROR**\n\n**Details:** {str(e)[:150]}\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                        asyncio.run(send_telegram_notification(error_msg))
                    except:
                        pass  # Don't fail on notification errors
                
                import traceback
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
        
        # Try to send critical error notification
        if check_telegram_config():
            try:
                error_msg = f"❌ **FUNCTION ERROR**\n\n**Details:** {str(e)[:150]}\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                asyncio.run(send_telegram_notification(error_msg))
            except:
                pass
        
        import traceback
        error_response = {
            "error": f"Function error: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        return (json.dumps(error_response), 500, headers)

# For local testing
if __name__ == "__main__":
    pass