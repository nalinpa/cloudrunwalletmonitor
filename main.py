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

async def _debug_database_connection(config: Config, network: str) -> Dict:
    """Debug database connection and data - Fixed for Motor driver"""
    debug_info = {}
    
    try:
        from services.database.database_client import DatabaseService
        db_service = DatabaseService(config)
        await db_service.initialize()
        
        # Fix: Use db_service._connected instead of db_service.db
        if db_service._connected and db_service.db is not None:
            debug_info['database_connected'] = True
            
            collections = await db_service.db.list_collection_names()
            debug_info['collections'] = collections
            debug_info['wallets_collection_exists'] = config.wallets_collection in collections
            
            if config.wallets_collection in collections:
                collection = db_service.db[config.wallets_collection]
                
                # Count total documents (all wallets, not network-specific)
                total_count = await collection.count_documents({})
                debug_info['total_wallets'] = total_count
                
                # Check field existence for different possible field names
                field_counts = {}
                for field in ['address', 'wallet_address', 'wallet', 'score', 'rating']:
                    try:
                        field_counts[field] = await collection.count_documents({field: {"$exists": True}})
                    except Exception as e:
                        field_counts[field] = f"Error: {str(e)}"
                
                debug_info['field_existence'] = field_counts
                
                # Sample documents
                try:
                    sample_docs = await collection.find({}).limit(3).to_list(length=3)
                    safe_samples = []
                    for doc in sample_docs:
                        safe_doc = {
                            'fields': list(doc.keys()),
                            'has_address': 'address' in doc,
                            'has_wallet_address': 'wallet_address' in doc, 
                            'has_wallet': 'wallet' in doc,
                            'has_score': 'score' in doc,
                            'has_rating': 'rating' in doc,
                            'address_length': len(doc.get('address', '')) if doc.get('address') else 0,
                            'score_value': doc.get('score'),
                            'rating_value': doc.get('rating')
                        }
                        safe_samples.append(safe_doc)
                    
                    debug_info['sample_documents'] = safe_samples
                except Exception as e:
                    debug_info['sample_error'] = str(e)
                
                # Try getting wallets using the service
                try:
                    wallets = await db_service.get_top_wallets(network, 10)
                    debug_info['fetched_wallets_count'] = len(wallets)
                    debug_info['sample_wallet_data'] = [
                        {
                            'address_length': len(w.address) if w.address else 0,
                            'score': w.score,
                            'network': w.network
                        } for w in wallets[:3]
                    ]
                except Exception as e:
                    debug_info['wallet_fetch_error'] = str(e)
            
        else:
            debug_info['database_connected'] = False
            debug_info['error'] = 'Database connection failed'
            
        await db_service.cleanup()
        
    except Exception as e:
        debug_info['database_error'] = str(e)
        debug_info['database_connected'] = False
        
        # Add more specific error information
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
        debug_info['alchemy_url'] = base_url[:50] + "..." if base_url and len(base_url) > 50 else base_url
        
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
            # Test with a dummy wallet address
            test_address = "0x742b15dbd95c4c8de2c0c8e44e2e8c6a8e9b9a78"
            transfers = await alchemy_service.get_transfers_batch(
                network, [test_address], start_block, end_block
            )
            debug_info['alchemy_api_responsive'] = True
            debug_info['test_transfer_structure'] = {
                'has_test_address': test_address in transfers,
                'transfer_keys': list(transfers.get(test_address, {}).keys()) if test_address in transfers else []
            }
        else:
            debug_info['alchemy_api_responsive'] = False
            debug_info['alchemy_error'] = "Block range retrieval failed"
        
    except Exception as e:
        debug_info['alchemy_error'] = str(e)
        debug_info['alchemy_api_responsive'] = False
        
        # Add more specific error information
        import traceback
        debug_info['alchemy_traceback'] = traceback.format_exc()
    
    return debug_info

async def _run_analysis_with_debug(request_data: Dict) -> Dict[str, Any]:
    """Run the analysis with detailed debugging"""
    # Extract and validate parameters
    config = Config()
    network = request_data.get('network', 'ethereum')
    analysis_type = request_data.get('analysis_type', 'buy')
    num_wallets = min(int(request_data.get('num_wallets', 50)), config.max_wallets)
    days_back = float(request_data.get('days_back', 1.0))
    debug_mode = request_data.get('debug', False)
    
    # Validate
    if network not in config.supported_networks:
        raise ValueError(f'Invalid network: {network}')
    
    if analysis_type not in config.supported_analysis_types:
        raise ValueError(f'Invalid analysis_type: {analysis_type}')
    
    debug_info = {
        'config_validation': 'passed',
        'requested_params': {
            'network': network,
            'analysis_type': analysis_type,
            'num_wallets': num_wallets,
            'days_back': days_back
        }
    }
    
    # If debug mode, run diagnostics
    if debug_mode:
        logger.info("Running in debug mode")
        debug_info['database_debug'] = await _debug_database_connection(config, network)
        debug_info['alchemy_debug'] = await _debug_alchemy_connection(config, network)
        
        return {
            'debug_mode': True,
            'debug_info': debug_info,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        }
    
    # Run normal analysis
    try:
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
            'success': True,
            'debug_info': debug_info  # Include basic debug info
        }
        
    except Exception as e:
        debug_info['analysis_error'] = str(e)
        logger.error(f"Analysis failed: {e}")
        
        # Add full traceback for debugging
        import traceback
        debug_info['analysis_traceback'] = traceback.format_exc()
        
        return {
            'error': str(e),
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'debug_info': debug_info
        }

@functions_framework.http
def crypto_analysis_function(request: Request):
    """Cloud Functions HTTP entry point - Fixed event loop management"""
    
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
        # Handle GET request (health check with basic debug info)
        if request.method == 'GET':
            debug_param = request.args.get('debug', '').lower() == 'true'
            
            basic_response = {
                "message": "Crypto Analysis Function is running!",
                "status": "healthy",
                "version": "2.2.0-debug",
                "service": "crypto-analysis-cloud-function",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": _initialized
            }
            
            if debug_param:
                # Add basic config info for debugging
                config = Config()
                basic_response['debug_info'] = {
                    'mongo_configured': bool(config.mongo_uri),
                    'alchemy_configured': bool(config.alchemy_api_key),
                    'supported_networks': config.supported_networks,
                    'db_name': config.db_name,
                    'wallets_collection': config.wallets_collection,
                    'mongo_uri_length': len(config.mongo_uri) if config.mongo_uri else 0,
                    'alchemy_key_length': len(config.alchemy_api_key) if config.alchemy_api_key else 0
                }
            
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
            
            # Run analysis with proper event loop handling
            try:
                # FIXED: Use asyncio.run instead of managing event loops manually
                result = asyncio.run(_run_analysis_with_debug(request_json))
                return (json.dumps(result), 200, headers)
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                
                # Include traceback in error response
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
        
        # Include traceback in error response
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
    import functions_framework
    # This allows local testing with functions-framework
    # Run with: functions-framework --target=crypto_analysis_function --debug
    pass