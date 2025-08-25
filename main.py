import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Cloud Functions imports (NOT FastAPI/uvicorn)
import functions_framework
from flask import Request

# Your existing imports
from utils.logger import setup_logging
from utils.config import Config
from core.analysis.buy_analyzer import CloudBuyAnalyzer
from core.analysis.sell_analyzer import CloudSellAnalyzer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global analyzer instances (reused across requests)
_analyzers = {}

async def get_analyzer(network: str, analysis_type: str):
    """Get or create analyzer instance"""
    key = f"{network}_{analysis_type}"
    
    if key not in _analyzers:
        if analysis_type == 'buy':
            analyzer = CloudBuyAnalyzer(network)
        else:
            analyzer = CloudSellAnalyzer(network)
        
        await analyzer.initialize()
        _analyzers[key] = analyzer
    
    return _analyzers[key]

async def _run_analysis(request_data: Dict) -> Dict[str, Any]:
    """Run the analysis with validation - your existing logic"""
    # Extract and validate parameters
    config = Config()
    network = request_data.get('network', 'ethereum')
    analysis_type = request_data.get('analysis_type', 'buy')
    num_wallets = min(int(request_data.get('num_wallets', 50)), config.max_wallets)
    days_back = float(request_data.get('days_back', 1.0))
    
    # Validate
    if network not in config.supported_networks:
        raise ValueError(f'Invalid network: {network}')
    
    if analysis_type not in config.supported_analysis_types:
        raise ValueError(f'Invalid analysis_type: {analysis_type}')
    
    # Get analyzer and run analysis
    analyzer = await get_analyzer(network, analysis_type)
    result = await analyzer.analyze(num_wallets, days_back)
    
    return {
        'network': result.network,
        'analysis_type': result.analysis_type,
        'total_transactions': result.total_transactions,
        'unique_tokens': result.unique_tokens,
        'total_eth_value': result.total_eth_value,
        'top_tokens': result.ranked_tokens[:10],
        'performance_metrics': result.performance_metrics,
        'timestamp': datetime.utcnow().isoformat(),
        'success': True
    }

@functions_framework.http
def crypto_analysis_function(request: Request):
    """Cloud Functions HTTP entry point"""
    
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
            return (
                json.dumps({
                    "message": "Crypto Analysis Function is running!",
                    "status": "healthy",
                    "version": "2.0.0",
                    "service": "crypto-analysis-cloud-function",
                    "timestamp": datetime.utcnow().isoformat()
                }), 
                200, 
                headers
            )
        
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
            
            # Run analysis (convert async to sync for Cloud Functions)
            try:
                # Create new event loop for the analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(_run_analysis(request_json))
                    return (json.dumps(result), 200, headers)
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                error_response = {
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat()
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
        error_response = {
            "error": f"Function error: {str(e)}",
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        return (json.dumps(error_response), 500, headers)

# For local testing (not used in Cloud Functions)
if __name__ == "__main__":
    import functions_framework
    # This won't be called in Cloud Functions, but useful for local testing
    pass