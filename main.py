import os
import asyncio
import logging
from datetime import datetime, timedelta
import traceback
import hashlib

# Fast JSON handling
try:
    import orjson as json
    def json_dumps(data): return json.dumps(data).decode('utf-8')
    def json_loads(data): return json.loads(data)
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    def json_dumps(data): return json.dumps(data)
    def json_loads(data): return json.loads(data)
    ORJSON_AVAILABLE = False

import functions_framework
from flask import Request 

from utils.config import Config
from handlers.analysis_handler import analysis_handler

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"}',
        force=True
    )

setup_logging()
logger = logging.getLogger(__name__)

# Global state
_initialized = False
_recent_requests = {}
_cleanup_last = datetime.now()

# Utility functions
def generate_request_hash(request_data: dict) -> str:
    """Generate request hash for duplicate detection"""
    key_params = {
        'network': request_data.get('network', ''),
        'analysis_type': request_data.get('analysis_type', ''),
        'num_wallets': request_data.get('num_wallets', 0),
        'days_back': request_data.get('days_back', 0)
    }
    request_string = f"{key_params['network']}-{key_params['analysis_type']}-{key_params['num_wallets']}-{key_params['days_back']}"
    return hashlib.md5(request_string.encode()).hexdigest()

def cleanup_old_requests():
    """Clean up old request tracking"""
    global _recent_requests, _cleanup_last
    
    now = datetime.now()
    if (now - _cleanup_last).seconds < 30:
        return
    
    cutoff = now - timedelta(minutes=2)
    keys_to_remove = [k for k, timestamp in _recent_requests.items() if timestamp < cutoff]
    
    for key in keys_to_remove:
        del _recent_requests[key]
    
    _cleanup_last = now
    if keys_to_remove:
        logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old request hashes")

def is_duplicate_request(request_data: dict) -> bool:
    """Check for duplicate requests"""
    global _recent_requests
    
    cleanup_old_requests()
    request_hash = generate_request_hash(request_data)
    now = datetime.now()
    
    if request_hash in _recent_requests:
        last_seen = _recent_requests[request_hash]
        time_diff = (now - last_seen).total_seconds()
        
        if time_diff < 60:
            logger.warning(f"🚫 DUPLICATE REQUEST BLOCKED: {request_hash} (last seen {time_diff:.1f}s ago)")
            return True
    
    _recent_requests[request_hash] = now
    logger.info(f"✅ New unique request: {request_hash}")
    return False

async def initialize_services():
    """Initialize services once"""
    global _initialized
    if not _initialized:
        logger.info("Initializing services...")
        
        try:
            config = Config()
            errors = config.validate()
            if errors:
                logger.warning(f"Config warnings: {errors}")
            
            await analysis_handler.initialize()
            
            logger.info("✅ Services initialized")
            _initialized = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            _initialized = False

@functions_framework.http
def main(request: Request):
    """Main HTTP entry point"""
    logger.info("🚀 Function started")
    
    # Initialize on first request
    try:
        if not _initialized:
            asyncio.run(initialize_services())
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return (json_dumps({"error": f"Initialization failed: {str(e)}", "success": False}), 500)
    
    # CORS handling
    if request.method == 'OPTIONS':
        return ('', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
    
    headers = {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'}
    
    try:
        # Route requests
        if request.method == 'GET':
            return asyncio.run(handle_get_request(headers))
        elif request.method == 'POST':
            return asyncio.run(handle_post_request(request, headers))
        else:
            return (json_dumps({"error": "Method not allowed"}), 405, headers)
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return (json_dumps({
            "error": f"Request error: {str(e)}",
            "success": False,
            "traceback": traceback.format_exc()
        }), 500, headers)

async def handle_get_request(headers):
    """Handle GET requests (health check)"""
    try:
        run_history = await analysis_handler.get_run_history(5)
        
        response = {
            "message": "Crypto Analysis Function",
            "status": "healthy",
            "version": "9.0.0-unified",
            "service": "crypto-analysis-cloud-function",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": _initialized,
            "performance_boost": f"orjson {'enabled' if ORJSON_AVAILABLE else 'disabled'}",
            "duplicate_prevention": "enabled",
            "recent_runs": run_history,
            "request_tracking": {
                "active_requests": len(_recent_requests),
                "duplicate_window": "60s"
            },
            "features": [
                "Unified Analysis Engine",
                "Consolidated Configuration", 
                "Enhanced AI with Web3",
                "Telegram Notifications",
                "Momentum Tracking",
                "Smart Timing"
            ]
        }
        
        return (json_dumps(response), 200, headers)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (json_dumps({"error": f"Health check failed: {str(e)}", "success": False}), 500, headers)

async def handle_post_request(request: Request, headers):
    """Handle POST requests (analysis)"""
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return (json_dumps({"error": "No JSON data provided"}), 400, headers)
        
        # Duplicate prevention (skip for test mode)
        test_mode = request_json.get('test_mode', False)
        
        if not test_mode and is_duplicate_request(request_json):
            return (json_dumps({
                "error": "Duplicate request detected",
                "message": "This analysis was requested within the last 60 seconds",
                "success": False,
                "retry_after_seconds": 60
            }), 429, headers)
        
        # Process request
        if test_mode:
            logger.info("🧪 TEST MODE: Processing without database writes")
        
        result = await analysis_handler.handle_analysis_request(request_json)
        status_code = 200 if result.get('success', False) else 500
        
        # Add metadata
        if test_mode:
            result['test_mode'] = True
        else:
            request_hash = generate_request_hash(request_json)
            result['duplicate_prevention'] = {
                'request_hash': request_hash,
                'processed_at': datetime.utcnow().isoformat()
            }
        
        return (json_dumps(result), status_code, headers)
        
    except Exception as e:
        logger.error(f"Analysis request failed: {e}")
        return (json_dumps({
            "error": str(e),
            "success": False,
            "traceback": traceback.format_exc()
        }), 500, headers)

# Local testing
if __name__ == "__main__":
    logger.info("Starting local test")
    asyncio.run(initialize_services())
    print("Function ready for testing")