import os
import asyncio
import logging
from datetime import datetime, timedelta
import traceback
import hashlib
import numpy as np  # Added for numpy type conversion

try:
    import orjson as json
    def json_dumps(data):
        # Convert numpy types to Python types for orjson
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            return obj
        
        # Convert numpy types before serializing
        converted_data = convert_numpy(data)
        return json.dumps(converted_data).decode('utf-8')
        
    def json_loads(data):
        return json.loads(data)
    logger = logging.getLogger(__name__)
    logger.info("🚀 Using orjson for 3x faster JSON processing with numpy support")
except ImportError:
    import json
    def json_dumps(data):
        # Convert numpy types to Python types for standard json
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            return obj
        
        # Convert numpy types before serializing
        converted_data = convert_numpy(data)
        return json.dumps(converted_data)
        
    def json_loads(data):
        return json.loads(data)
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ orjson not available, using standard json with numpy support")

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

# Import our services
try:
    from utils.config import Config
    from handlers.analysis_handler import analysis_handler
    from services.notifications.notifications import telegram_service
    logger.info("Successfully imported all services")
except Exception as e:
    logger.error(f"Failed to import services: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

# Global initialization flag
_initialized = False

# DUPLICATE PREVENTION SYSTEM
_recent_requests = {}
_cleanup_last = datetime.now()

def generate_request_hash(request_data: dict) -> str:
    """Generate a hash for the request to detect duplicates"""
    # Create a consistent string from request parameters
    key_params = {
        'network': request_data.get('network', ''),
        'analysis_type': request_data.get('analysis_type', ''),
        'num_wallets': request_data.get('num_wallets', 0),
        'days_back': request_data.get('days_back', 0)
    }
    
    # Create hash from parameters
    request_string = f"{key_params['network']}-{key_params['analysis_type']}-{key_params['num_wallets']}-{key_params['days_back']}"
    return hashlib.md5(request_string.encode()).hexdigest()

def cleanup_old_requests():
    """Clean up old request tracking (older than 2 minutes)"""
    global _recent_requests, _cleanup_last
    
    now = datetime.now()
    
    # Only cleanup every 30 seconds
    if (now - _cleanup_last).seconds < 30:
        return
    
    cutoff = now - timedelta(minutes=2)
    keys_to_remove = []
    
    for key, timestamp in _recent_requests.items():
        if timestamp < cutoff:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del _recent_requests[key]
    
    _cleanup_last = now
    if len(keys_to_remove) > 0:
        logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old request hashes")

def is_duplicate_request(request_data: dict) -> bool:
    """Check if this is a duplicate request within the last 60 seconds"""
    global _recent_requests
    
    # Clean up old requests first
    cleanup_old_requests()
    
    # Generate hash for this request
    request_hash = generate_request_hash(request_data)
    now = datetime.now()
    
    # Check if we've seen this request recently
    if request_hash in _recent_requests:
        last_seen = _recent_requests[request_hash]
        time_diff = (now - last_seen).total_seconds()
        
        if time_diff < 60:  # Within 60 seconds
            logger.warning(f"🚫 DUPLICATE REQUEST BLOCKED: {request_hash} (last seen {time_diff:.1f}s ago)")
            return True
    
    # Record this request
    _recent_requests[request_hash] = now
    logger.info(f"✅ New unique request: {request_hash}")
    return False

async def initialize_services():
    """Initialize all services once"""
    global _initialized
    if not _initialized:
        logger.info("Initializing Cloud Function services with orjson and duplicate prevention...")
        
        try:
            # Validate configuration
            config = Config()
            logger.info(f"BigQuery project: {getattr(config, 'bigquery_project_id', 'Not configured')}")
            logger.info(f"BigQuery configured: {bool(config.bigquery_project_id)}")
            logger.info(f"Alchemy configured: {bool(config.alchemy_api_key)}")
            
            # Initialize analysis handler (includes last run tracker)
            await analysis_handler.initialize()
            
            # Log Telegram configuration status
            if telegram_service.is_configured():
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                logger.info(f"Telegram configured - Bot: {bot_token[:10]}...{bot_token[-5:] if len(bot_token) > 15 else '****'}, Chat: {chat_id}")
            else:
                logger.warning("Telegram not configured")
            
            logger.info("✅ Services initialized with orjson performance + duplicate prevention")
            _initialized = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - let the function start but log the error
            _initialized = False

@functions_framework.http
def main(request: Request):
    """Cloud Functions HTTP entry point with orjson performance and duplicate prevention"""
    
    logger.info("🚀 Cloud Function started with orjson + duplicate prevention + numpy support")
    
    # Initialize services on first request
    try:
        if not _initialized:
            asyncio.run(initialize_services())
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        error_response = {
            "error": f"Service initialization failed: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        return (json_dumps(error_response), 500, {'Content-Type': 'application/json'})
    
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
        # Handle GET request (health check + status)
        if request.method == 'GET':
            path = request.path or request.url.path
        
            if path == '/debug-etherscan':  
                return asyncio.run(handle_etherscan_debug(headers))
            else:       
                return asyncio.run(handle_health_check(headers))
        
        # Handle POST request (analysis) with duplicate prevention
        if request.method == 'POST':
            return asyncio.run(handle_analysis_request_with_deduplication(request, headers))
        
        # Method not allowed
        return (
            json_dumps({"error": "Method not allowed"}), 
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
        return (json_dumps(error_response), 500, headers)

async def handle_etherscan_debug(headers):
    """Debug Etherscan API calls step by step"""
    try:
        from utils.config import Config
        import aiohttp
        
        config = Config()
        
        debug_results = {
            "config_check": {
                "api_key_exists": bool(config.etherscan_api_key),
                "api_key_length": len(config.etherscan_api_key) if config.etherscan_api_key else 0,
                "api_key_preview": config.etherscan_api_key[:10] + "..." if config.etherscan_api_key else "None",
                "endpoint": config.etherscan_endpoint,
                "rate_limit": config.etherscan_api_rate_limit
            },
            "api_tests": {}
        }
        
        if not config.etherscan_api_key:
            debug_results["error"] = "No API key configured"
            return (json_dumps(debug_results), 500, headers)
        
        # Test known verified contracts
        test_contracts = {
            "ethereum_usdc": {
                "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                "network": "ethereum",
                "expected": "USDC - should be verified"
            },
            "base_usdc": {
                "address": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
                "network": "base", 
                "expected": "USDC on Base - should be verified"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            for test_name, test_data in test_contracts.items():
                address = test_data["address"]
                network = test_data["network"]

                chain_id = 1

                if network == 'ethereum':
                    chain_id = 1
                elif network == 'base':
                    chain_id = 8453
                
                # Get endpoint
                endpoint = config.etherscan_endpoint
                if not endpoint:
                    debug_results["api_tests"][test_name] = {
                        "error": f"No endpoint configured for {network}"
                    }
                    continue
                
                # Build URL
                url = f"{endpoint}?chainid={chain_id}&module=contract&action=getsourcecode&address={address}&apikey={config.etherscan_api_key}"

                debug_results["api_tests"][test_name] = {
                    "network": network,
                    "address": address,
                    "endpoint": endpoint,
                    "expected": test_data["expected"]
                }
                
                try:
                    # Make API call
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        debug_results["api_tests"][test_name]["http_status"] = response.status
                        
                        if response.status == 200:
                            data = await response.json()
                            debug_results["api_tests"][test_name]["api_response"] = {
                                "status": data.get("status"),
                                "message": data.get("message", ""),
                                "result_exists": bool(data.get("result"))
                            }
                            
                            if data.get("status") == "1" and data.get("result"):
                                result = data["result"][0] if isinstance(data["result"], list) else data["result"]
                                
                                source_code = result.get("SourceCode", "")
                                contract_name = result.get("ContractName", "")
                                
                                debug_results["api_tests"][test_name]["contract_details"] = {
                                    "contract_name": contract_name,
                                    "has_source_code": bool(source_code and source_code.strip()),
                                    "source_code_length": len(source_code) if source_code else 0,
                                    "compiler_version": result.get("CompilerVersion", ""),
                                    "abi_available": bool(result.get("ABI", "").strip())
                                }
                                
                                # Determine if verified
                                is_verified = bool(source_code and source_code.strip() and source_code != "{{}}")
                                debug_results["api_tests"][test_name]["is_verified"] = is_verified
                                debug_results["api_tests"][test_name]["verification_status"] = "✅ VERIFIED" if is_verified else "❌ NOT VERIFIED"
                                
                            elif data.get("status") == "0":
                                error_msg = data.get("message", "Unknown error")
                                debug_results["api_tests"][test_name]["api_error"] = error_msg
                                
                                if "rate limit" in error_msg.lower():
                                    debug_results["api_tests"][test_name]["issue"] = "RATE LIMITED"
                                elif "invalid" in error_msg.lower():
                                    debug_results["api_tests"][test_name]["issue"] = "INVALID API KEY"
                                else:
                                    debug_results["api_tests"][test_name]["issue"] = "API ERROR"
                        
                        elif response.status == 403:
                            debug_results["api_tests"][test_name]["issue"] = "FORBIDDEN - Check API key permissions"
                        elif response.status == 429:
                            debug_results["api_tests"][test_name]["issue"] = "RATE LIMITED"
                        else:
                            debug_results["api_tests"][test_name]["issue"] = f"HTTP {response.status}"
                
                except Exception as e:
                    debug_results["api_tests"][test_name]["exception"] = str(e)
                
                # Add delay between tests
                await asyncio.sleep(0.5)
        
        # Summary
        successful_tests = sum(1 for test in debug_results["api_tests"].values() 
                             if test.get("is_verified") is True)
        total_tests = len(debug_results["api_tests"])
        
        debug_results["summary"] = {
            "successful_verifications": f"{successful_tests}/{total_tests}",
            "api_working": successful_tests > 0,
            "all_tests_passed": successful_tests == total_tests
        }
        
        status_code = 200 if successful_tests > 0 else 500
        return (json_dumps(debug_results), status_code, headers)
        
    except Exception as e:
        error_response = {
            "error": f"Debug test failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }
        return (json_dumps(error_response), 500, headers)

async def handle_health_check(headers):
    """Handle GET request for health check"""
    try:
        # Get run history if available
        run_history = await analysis_handler.get_run_history(5)
        
        response = {
            "message": "Crypto Analysis Function with orjson Performance + Duplicate Prevention + Numpy Support",
            "status": "healthy",
            "version": "8.3.0-orjson-dedup-numpy",
            "service": "crypto-analysis-cloud-function",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": _initialized,
            "telegram_configured": telegram_service.is_configured(),
            "database_type": "BigQuery",
            "ai_library": "Enhanced AI with Web3 Intelligence",
            "architecture": "modular",
            "performance_boost": "orjson (3x faster JSON) + numpy support",
            "duplicate_prevention": "enabled (60s window)",
            "last_run_tracking": analysis_handler.last_run_tracker and analysis_handler.last_run_tracker.is_available(),
            "recent_runs": run_history,
            "request_tracking": {
                "active_requests": len(_recent_requests),
                "cleanup_interval": "30s",
                "duplicate_window": "60s"
            },
            "services": {
                "analysis_handler": "✓ Initialized" if _initialized else "✗ Not initialized",
                "telegram_service": "✓ Configured" if telegram_service.is_configured() else "✗ Not configured",
                "last_run_tracker": "✓ Available" if analysis_handler.last_run_tracker and analysis_handler.last_run_tracker.is_available() else "✗ Not available"
            },
            "features": [
                "orjson Performance Boost (3x faster)",
                "Numpy Type Support (no serialization errors)",
                "Duplicate Request Prevention (60s window)",
                "Enhanced AI Analysis with Web3 Intelligence", 
                "Token Age & Holder Count Display",
                "Contract Verification Status",
                "ML Anomaly Detection", 
                "Sentiment Analysis",
                "Whale Coordination Detection",
                "Smart Money Flow Analysis",
                "Telegram notifications with enhanced formatting",
                "Smart Timing (automatic days_back calculation)",
                "Last Run Tracking (BigQuery)",
                "Contract Address Extraction",
                "Lower Notification Thresholds",
                "Modular Architecture"
            ]
        }
        
        return (json_dumps(response), 200, headers)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_response = {
            "error": f"Health check failed: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        return (json_dumps(error_response), 500, headers)

async def handle_analysis_request_with_deduplication(request: Request, headers):
    """Handle POST request for analysis with test flag support"""
    try:
        # Get JSON data using orjson for 3x faster parsing
        request_json = request.get_json(silent=True)
        if not request_json:
            return (
                json_dumps({"error": "No JSON data provided"}), 
                400, 
                headers
            )
        
        # DUPLICATE PREVENTION CHECK (skip for test mode)
        test_mode = request_json.get('test_mode', False)
        
        if not test_mode and is_duplicate_request(request_json):
            logger.warning("🚫 Rejecting duplicate request")
            duplicate_response = {
                "error": "Duplicate request detected",
                "message": "This exact analysis was requested within the last 60 seconds. Please wait before retrying.",
                "success": False,
                "timestamp": datetime.utcnow().isoformat(),
                "duplicate_prevention": True,
                "retry_after_seconds": 60
            }
            return (json_dumps(duplicate_response), 429, headers)
        
        # Log the request type
        if test_mode:
            logger.info(f"🧪 TEST MODE: Processing test request without database writes")
        else:
            # Log the unique request
            request_hash = generate_request_hash(request_json)
            logger.info(f"🚀 Processing unique request: {request_hash}")
        
        logger.info(f"Request: {request_json.get('network')}-{request_json.get('analysis_type')} ({request_json.get('num_wallets', 0)} wallets)")
        
        # Pass test_mode flag to analysis handler
        result = await analysis_handler.handle_analysis_request(request_json)
        status_code = 200 if result.get('success', False) else 500
        
        # Add test mode info to response
        if test_mode:
            result['test_mode'] = True
            result['test_info'] = {
                'database_writes_disabled': True,
                'alert_storage_disabled': True,
                'notification_disabled': True,
                'intelligence_api_testing': True
            }
            logger.info(f"✅ Test request completed successfully")
        else:
            # Add deduplication info to response for production
            result['duplicate_prevention'] = {
                'request_hash': request_hash,
                'processed_at': datetime.utcnow().isoformat(),
                'duplicate_check_passed': True,
                'active_requests_tracked': len(_recent_requests)
            }
            logger.info(f"✅ Request completed: {request_hash} (status: {status_code})")
        
        # Use orjson for 3x faster response serialization with numpy support
        return (json_dumps(result), status_code, headers)
        
    except Exception as e:
        logger.error(f"Analysis request failed: {e}")
        logger.error(f"Analysis request traceback: {traceback.format_exc()}")
        
        error_response = {
            "error": str(e),
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        }
        return (json_dumps(error_response), 500, headers)
    
# For local testing
if __name__ == "__main__":
    logger.info("Starting local test with orjson performance boost + duplicate prevention + numpy support")
    asyncio.run(initialize_services())
    print("Function ready for local testing with 3x faster JSON processing, duplicate prevention, and numpy support.")