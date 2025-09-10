import os
import asyncio
import logging
from datetime import datetime
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

async def initialize_services():
    """Initialize all services once"""
    global _initialized
    if not _initialized:
        logger.info("Initializing Cloud Function services...")
        
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
            
            logger.info("Services initialized successfully")
            _initialized = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - let the function start but log the error
            _initialized = False

@functions_framework.http
def main(request: Request):
    """Cloud Functions HTTP entry point"""
    
    logger.info("🚀 Cloud Function started with modular architecture")
    
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
        # Handle GET request (health check + status)
        if request.method == 'GET':
            return asyncio.run(handle_health_check(headers))
        
        # Handle POST request (analysis)
        if request.method == 'POST':
            return asyncio.run(handle_analysis_request(request, headers))
        
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

async def handle_health_check(headers):
    """Handle GET request for health check"""
    try:
        # Get run history if available
        run_history = await analysis_handler.get_run_history(5)
        
        response = {
            "message": "Crypto Analysis Function with Modular Architecture",
            "status": "healthy",
            "version": "8.0.0-modular",
            "service": "crypto-analysis-cloud-function",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": _initialized,
            "telegram_configured": telegram_service.is_configured(),
            "database_type": "BigQuery",
            "ai_library": "pandas-ta",
            "architecture": "modular",
            "last_run_tracking": analysis_handler.last_run_tracker and analysis_handler.last_run_tracker.is_available(),
            "recent_runs": run_history,
            "services": {
                "analysis_handler": "✓ Initialized" if _initialized else "✗ Not initialized",
                "telegram_service": "✓ Configured" if telegram_service.is_configured() else "✗ Not configured",
                "last_run_tracker": "✓ Available" if analysis_handler.last_run_tracker and analysis_handler.last_run_tracker.is_available() else "✗ Not available"
            },
            "features": [
                "pandas-ta Technical Analysis (15+ indicators)",
                "ML Anomaly Detection", 
                "Sentiment Analysis",
                "Whale Coordination Detection",
                "Smart Money Flow Analysis",
                "Telegram notifications",
                "Smart Timing (automatic days_back calculation)",
                "Last Run Tracking (BigQuery)",
                "Modular Architecture"
            ]
        }
        
        return (json.dumps(response), 200, headers)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_response = {
            "error": f"Health check failed: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        return (json.dumps(error_response), 500, headers)

async def handle_analysis_request(request: Request, headers):
    """Handle POST request for analysis"""
    try:
        # Get JSON data
        request_json = request.get_json(silent=True)
        if not request_json:
            return (
                json.dumps({"error": "No JSON data provided"}), 
                400, 
                headers
            )
        
        logger.info(f"Running analysis request: {request_json}")
        
        # Delegate to analysis handler
        result = await analysis_handler.handle_analysis_request(request_json)
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

# For local testing
if __name__ == "__main__":
    logger.info("Starting local test with modular architecture")
    asyncio.run(initialize_services())
    print("Function ready for local testing. Send requests to test endpoints.")