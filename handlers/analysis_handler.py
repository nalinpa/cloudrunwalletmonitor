import logging
import traceback
from datetime import datetime
from typing import Dict, Any

from utils.config import Config
from services.tracker.last_run_tracker import LastRunTracker
from services.notifications.notifications import telegram_service

logger = logging.getLogger(__name__)

class AnalysisHandler:
    """Handle analysis requests with smart timing and notifications"""
    
    def __init__(self):
        self.config = Config()
        self.last_run_tracker: LastRunTracker = None
        self._analyzers = {}
    
    async def initialize(self):
        """Initialize the analysis handler"""
        # Initialize last run tracker
        self.last_run_tracker = LastRunTracker(self.config)
        await self.last_run_tracker.initialize()
        logger.info("Analysis handler initialized")
    
    async def get_analyzer(self, network: str, analysis_type: str):
        """Get or create analyzer instance with error handling"""
        key = f"{network}_{analysis_type}"
        
        if key not in self._analyzers:
            try:
                logger.info(f"Creating analyzer: {key}")
                
                if analysis_type == 'buy':
                    from core.analysis.buy_analyzer import CloudBuyAnalyzer
                    analyzer = CloudBuyAnalyzer(network)
                else:
                    from core.analysis.sell_analyzer import CloudSellAnalyzer
                    analyzer = CloudSellAnalyzer(network)
                
                await analyzer.initialize()
                self._analyzers[key] = analyzer
                logger.info(f"Successfully created analyzer: {key}")
                
            except Exception as e:
                logger.error(f"Failed to create analyzer {key}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        return self._analyzers[key]
    
    async def handle_analysis_request(self, request_data: Dict) -> Dict[str, Any]:
        """Handle analysis request with smart timing and notifications"""
        try:
            # Extract and validate parameters
            network = request_data.get('network', 'ethereum')
            analysis_type = request_data.get('analysis_type', 'buy')
            num_wallets = min(int(request_data.get('num_wallets', 50)), self.config.max_wallets)
            
            # Smart timing parameters
            requested_days_back = float(request_data.get('days_back', 1.0))
            use_smart_timing = request_data.get('smart_timing', True)
            
            # NEW: Store data flag - defaults to FALSE (no storage)
            store_data = request_data.get('store_data', False)
            
            debug_mode = request_data.get('debug', False)
            
            # Notification settings
            send_notifications = request_data.get('notifications', True)
            max_tokens = int(request_data.get('max_tokens', 7))
            min_alpha_score = float(request_data.get('min_alpha_score', 50.0))
            
            # Log the storage decision
            storage_status = "ENABLED" if store_data else "DISABLED"
            logger.info(f"Transfer data storage: {storage_status}")
            
            # Calculate smart days_back
            if use_smart_timing and self.last_run_tracker and self.last_run_tracker.is_available():
                days_back = await self.last_run_tracker.get_days_since_last_run(
                    network, analysis_type, requested_days_back
                )
                logger.info(f"Smart timing: Using {days_back} days back (requested: {requested_days_back})")
            else:
                days_back = requested_days_back
                logger.info(f"Manual timing: Using {days_back} days back")
            
            logger.info(f"Enhanced analysis params: {network} {analysis_type}, {num_wallets} wallets, {days_back} days, store_data={store_data}")
            
            # Validate parameters
            validation_error = self._validate_parameters(network, analysis_type)
            if validation_error:
                return validation_error
            
            # Test Telegram if notifications enabled
            telegram_status = None
            if send_notifications:
                telegram_status = await telegram_service.test_connection()
                logger.info(f"Telegram status: {telegram_status}")
            
            # Build debug info
            debug_info = self._build_debug_info(
                network, analysis_type, num_wallets, requested_days_back, days_back,
                use_smart_timing, max_tokens, min_alpha_score, telegram_status, store_data
            )
            
            # Return early if debug mode
            if debug_mode:
                return self._handle_debug_mode(debug_info)
            
            # Send start notification
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                await telegram_service.send_start_notification(
                    network, analysis_type, num_wallets, days_back, 
                    use_smart_timing, max_tokens, min_alpha_score, store_data
                )
            
            # Run analysis with store_data flag
            try:
                logger.info(f"Starting enhanced {analysis_type} analysis for {network}")
                analyzer = await self.get_analyzer(network, analysis_type)
                
                # Pass the store_data flag to the analyzer
                result = await analyzer.analyze(num_wallets, days_back, store_data=store_data)
                
                # Record successful run
                if self.last_run_tracker and self.last_run_tracker.is_available():
                    await self.last_run_tracker.record_run(network, analysis_type, days_back, "success")
                
                # Build result
                result_dict = self._build_success_result(result, days_back, use_smart_timing, debug_info, store_data)
                
                storage_msg = f", stored {result.performance_metrics.get('transfers_stored', 0)} transfers" if store_data else ", no data stored"
                logger.info(f"Enhanced analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens{storage_msg}")
                
                # Send notifications
                if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                    try:
                        await telegram_service.send_analysis_notifications(result, network, max_tokens, min_alpha_score)
                    except Exception as e:
                        logger.error(f"Failed to send notifications: {e}")
                
                return result_dict
                
            except Exception as e:
                error_msg = f"Enhanced analysis failed: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Analysis traceback: {traceback.format_exc()}")
                
                # Record failed run
                if self.last_run_tracker and self.last_run_tracker.is_available():
                    await self.last_run_tracker.record_run(network, analysis_type, days_back, "failed")
                
                return self._build_error_result(error_msg, days_back, debug_info, traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Enhanced request processing failed: {e}")
            logger.error(f"Request processing traceback: {traceback.format_exc()}")
            return {
                'error': f"Enhanced request processing failed: {str(e)}",
                'success': False,
                'timestamp': datetime.utcnow().isoformat(),
                'traceback': traceback.format_exc()
            }
    
    def _validate_parameters(self, network: str, analysis_type: str) -> Dict[str, Any]:
        """Validate request parameters"""
        if network not in self.config.supported_networks:
            error_msg = f'Invalid network: {network}'
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if analysis_type not in self.config.supported_analysis_types:
            error_msg = f'Invalid analysis_type: {analysis_type}'
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    def _build_debug_info(self, network: str, analysis_type: str, num_wallets: int,
                         requested_days_back: float, actual_days_back: float,
                         use_smart_timing: bool, max_tokens: int, min_alpha_score: float,
                         telegram_status: Dict, store_data: bool) -> Dict:
        """Build debug information"""
        return {
            'config_validation': 'passed',
            'requested_params': {
                'network': network,
                'analysis_type': analysis_type,
                'num_wallets': num_wallets,
                'requested_days_back': requested_days_back,
                'actual_days_back': actual_days_back,
                'smart_timing_enabled': use_smart_timing,
                'max_tokens': max_tokens,
                'min_alpha_score': min_alpha_score,
                'store_data': store_data  # NEW: Include storage flag in debug
            },
            'telegram_status': telegram_status,
            'notifications_enabled': telegram_service.is_configured(),
            'bigquery_configured': bool(self.config.bigquery_project_id),
            'alchemy_configured': bool(self.config.alchemy_api_key),
            'ai_enhancement': 'pandas-ta_available',
            'last_run_tracking': self.last_run_tracker and self.last_run_tracker.is_available(),
            'storage_enabled': store_data  # NEW: Storage status in debug
        }
    
    async def _handle_debug_mode(self, debug_info: Dict) -> Dict[str, Any]:
        """Handle debug mode request"""
        logger.info("Running in debug mode - returning config info")
        
        if self.last_run_tracker and self.last_run_tracker.is_available():
            debug_info['run_history'] = await self.last_run_tracker.get_run_history(5)
        
        return {
            'debug_mode': True,
            'debug_info': debug_info,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        }
    
    def _build_success_result(self, result, days_back: float, use_smart_timing: bool, 
                             debug_info: Dict, store_data: bool) -> Dict[str, Any]:
        """Build successful analysis result"""
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
            'ai_enhancement': 'pandas-ta',
            'days_back_used': days_back,
            'smart_timing_used': use_smart_timing,
            'data_stored': store_data,  # NEW: Include storage status in response
            'transfers_stored': result.performance_metrics.get('transfers_stored', 0) if store_data else 0,
            'debug_info': debug_info
        }
    
    def _build_error_result(self, error_msg: str, days_back: float, debug_info: Dict, tb: str) -> Dict[str, Any]:
        """Build error result"""
        return {
            'error': error_msg,
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'days_back_used': days_back,
            'debug_info': debug_info,
            'traceback': tb
        }
    
    async def get_run_history(self, limit: int = 10) -> list:
        """Get run history if tracker is available"""
        if self.last_run_tracker and self.last_run_tracker.is_available():
            return await self.last_run_tracker.get_run_history(limit)
        return []

# Global instance
analysis_handler = AnalysisHandler()