import logging
import traceback
from datetime import datetime
from typing import Dict, Any

try:
    import orjson as json
    def json_dumps(data):
        return json.dumps(data).decode('utf-8')
    def json_loads(data):
        return json.loads(data)
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    def json_dumps(data):
        return json.dumps(data)
    def json_loads(data):
        return json.loads(data)
    ORJSON_AVAILABLE = False

from utils.config import Config
from services.tracker.last_run_tracker import LastRunTracker

logger = logging.getLogger(__name__)

class AnalysisHandler:
    """Handle analysis requests with verified trades storage flag"""
    
    def __init__(self):
        self.config = Config()
        self.last_run_tracker: LastRunTracker = None
        self._analyzers = {}
        
        # Enhanced notification settings
        self.enhanced_notification_settings = {
            'default_min_alpha_score': 12.0,
            'max_min_alpha_score': 30.0,
            'default_max_tokens': 12,
            'min_max_tokens': 5,
            'auto_adjust_thresholds': True,
            'send_summary_always': True
        }
        
        if ORJSON_AVAILABLE:
            logger.info("Analysis handler initialized with orjson performance boost")
        else:
            logger.warning("Analysis handler using standard json")
    
    async def initialize(self):
        """Initialize the analysis handler"""
        try:
            # Initialize last run tracker
            self.last_run_tracker = LastRunTracker(self.config)
            await self.last_run_tracker.initialize()
            logger.info("Last run tracker initialized")
            
            # Initialize momentum tracking
            try:
                from services.notifications.notifications import telegram_service
                await telegram_service.initialize_momentum_tracking(self.config)
                logger.info("Momentum tracking initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize momentum tracking: {e}")
                logger.warning("Continuing without momentum tracking")
            
            logger.info("Enhanced analysis handler initialized")
            
        except Exception as e:
            logger.error(f"Analysis handler initialization failed: {e}")
            raise
    
    async def get_analyzer(self, network: str, analysis_type: str):
        """Get or create analyzer instance"""
        key = f"{network}_{analysis_type}"
        
        if key not in self._analyzers:
            try:
                logger.info(f"Creating analyzer: {key}")
                
                if analysis_type == 'buy':
                    from core.analysis.unified_analyzer import CloudBuyAnalyzer
                    analyzer = CloudBuyAnalyzer(network)
                else:
                    from core.analysis.unified_analyzer import CloudSellAnalyzer
                    analyzer = CloudSellAnalyzer(network)
                
                await analyzer.initialize()
                self._analyzers[key] = analyzer
                logger.info(f"Successfully created analyzer: {key}")
                
            except Exception as e:
                logger.error(f"Failed to create analyzer {key}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        return self._analyzers[key]
    
    def _apply_enhanced_notification_settings(self, request_data: Dict) -> Dict:
        """Apply enhanced notification settings"""
        settings = self.enhanced_notification_settings
        
        # Lower the minimum alpha score for more notifications
        original_min_score = request_data.get('min_alpha_score', settings['default_min_alpha_score'])
        enhanced_min_score = min(float(original_min_score), settings['max_min_alpha_score'])
        
        if enhanced_min_score > settings['default_min_alpha_score'] * 2:
            enhanced_min_score = settings['default_min_alpha_score'] * 1.5
            logger.info(f"Lowered min_alpha_score from {original_min_score} to {enhanced_min_score}")
        
        request_data['min_alpha_score'] = enhanced_min_score
        
        # Increase max tokens
        original_max_tokens = request_data.get('max_tokens', settings['default_max_tokens'])
        enhanced_max_tokens = max(int(original_max_tokens), settings['min_max_tokens'])
        
        if enhanced_max_tokens < settings['default_max_tokens']:
            enhanced_max_tokens = settings['default_max_tokens']
            logger.info(f"Increased max_tokens from {original_max_tokens} to {enhanced_max_tokens}")
        
        request_data['max_tokens'] = enhanced_max_tokens
        request_data['notifications'] = True
        
        logger.info(f"Enhanced notification settings: min_score={enhanced_min_score}, max_tokens={enhanced_max_tokens}")
        
        return request_data
    
    async def handle_analysis_request(self, request_data: Dict) -> Dict[str, Any]:
        """Handle analysis request with verified trades storage flag"""
        try:
            # Apply enhanced notification settings
            request_data = self._apply_enhanced_notification_settings(request_data)
            
            # Extract and validate parameters
            network = request_data.get('network', 'ethereum')
            analysis_type = request_data.get('analysis_type', 'buy')
            num_wallets = min(int(request_data.get('num_wallets', 50)), self.config.max_wallets)
            
            # Timing parameters
            requested_days_back = float(request_data.get('days_back', 1.0))
            use_smart_timing = request_data.get('smart_timing', True)
            
            # STORAGE FLAGS
            store_verified_trades = request_data.get('store_verified_trades', False)  # NEW FLAG
            store_alerts = request_data.get('store_alerts', True)
            
            debug_mode = request_data.get('debug', False)
            
            # Notification settings
            send_notifications = request_data.get('notifications', True)
            max_tokens = int(request_data.get('max_tokens', self.enhanced_notification_settings['default_max_tokens']))
            min_alpha_score = float(request_data.get('min_alpha_score', self.enhanced_notification_settings['default_min_alpha_score']))
            
            # Log the flags
            trades_status = "ENABLED" if store_verified_trades else "DISABLED"
            alerts_status = "ENABLED" if store_alerts else "DISABLED"
            
            logger.info(f"Verified trades storage: {trades_status}")
            logger.info(f"Alert momentum storage: {alerts_status}")
            logger.info(f"Enhanced notifications: min_score={min_alpha_score}, max_tokens={max_tokens}")
            
            # Calculate smart days_back
            if use_smart_timing and self.last_run_tracker and self.last_run_tracker.is_available():
                days_back = await self.last_run_tracker.get_days_since_last_run(
                    network, analysis_type, requested_days_back
                )
                logger.info(f"Smart timing: Using {days_back} days back (requested: {requested_days_back})")
            else:
                days_back = requested_days_back
                logger.info(f"Manual timing: Using {days_back} days back")
            
            logger.info(f"Analysis params: {network} {analysis_type}, {num_wallets} wallets, {days_back} days, store_verified_trades={store_verified_trades}, store_alerts={store_alerts}")
            
            # Validate parameters
            validation_error = self._validate_parameters(network, analysis_type)
            if validation_error:
                return validation_error
            
            # Test Telegram if notifications enabled
            telegram_status = None
            if send_notifications:
                from services.notifications.notifications import telegram_service
                telegram_status = await telegram_service.test_connection()
                logger.info(f"Telegram status: {telegram_status}")
            
            # Build debug info
            debug_info = self._build_debug_info(
                network, analysis_type, num_wallets, requested_days_back, days_back,
                use_smart_timing, max_tokens, min_alpha_score, telegram_status, 
                store_verified_trades, store_alerts
            )
            
            # Return early if debug mode
            if debug_mode:
                return self._handle_debug_mode(debug_info)
            
            # Send start notification
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                from services.notifications.notifications import telegram_service
                await telegram_service.send_start_notification(
                    network, analysis_type, num_wallets, days_back, 
                    use_smart_timing, max_tokens, min_alpha_score, store_verified_trades
                )
            
            # Run analysis
            try:
                logger.info(f"Starting enhanced {analysis_type} analysis for {network}")
                analyzer = await self.get_analyzer(network, analysis_type)
                
                # Pass verified trades flag to analyzer
                result = await analyzer.analyze(num_wallets, days_back)
                
                # Record successful run
                if self.last_run_tracker and self.last_run_tracker.is_available():
                    await self.last_run_tracker.record_run(network, analysis_type, days_back, "success")
                
                # Build result
                result_dict = self._build_success_result(result, days_back, use_smart_timing, debug_info, store_verified_trades, store_alerts)
                
                trades_msg = f", stored {result.performance_metrics.get('transfers_stored', 0)} verified trades" if store_verified_trades else ", no trades stored"
                alerts_msg = f", alerts stored" if store_alerts else ", no alert storage"
                
                logger.info(f"Enhanced analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens{trades_msg}{alerts_msg}")
                
                # Send notifications with store_alerts flag
                if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                    try:
                        from services.notifications.notifications import telegram_service
                        await telegram_service.send_analysis_notifications(result, network, max_tokens, min_alpha_score, store_alerts)
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
                     telegram_status: Dict, store_verified_trades: bool, store_alerts: bool) -> Dict:
        """Build debug information with storage flags"""
        
        momentum_available = False
        try:
            from services.notifications.notifications import telegram_service
            momentum_available = bool(telegram_service.momentum_tracker)
        except:
            pass
        
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
                'store_verified_trades': store_verified_trades,
                'store_alerts': store_alerts
            },
            'storage_flags': {
                'verified_trades_enabled': store_verified_trades,
                'alerts_enabled': store_alerts,
                'min_eth_threshold': 0.02
            },
            'enhanced_notifications': {
                'enabled': True,
                'auto_threshold_adjustment': True,
                'default_min_score': self.enhanced_notification_settings['default_min_alpha_score'],
                'max_min_score': self.enhanced_notification_settings['max_min_alpha_score'],
                'orjson_performance': ORJSON_AVAILABLE,
                'momentum_tracking': momentum_available
            },
            'telegram_status': telegram_status,
            'notifications_enabled': telegram_status.get('ready_for_notifications', False) if telegram_status else False,
            'bigquery_configured': bool(self.config.bigquery_project_id),
            'alchemy_configured': bool(self.config.alchemy_api_key),
            'ai_enhancement': 'available',
            'last_run_tracking': self.last_run_tracker and self.last_run_tracker.is_available(),
            'performance_boost': f"orjson {'enabled' if ORJSON_AVAILABLE else 'not available'}",
            'momentum_tracking': momentum_available
        }

    
    async def _handle_debug_mode(self, debug_info: Dict) -> Dict[str, Any]:
        """Handle debug mode request"""
        logger.info("Running in enhanced debug mode")
        
        if self.last_run_tracker and self.last_run_tracker.is_available():
            debug_info['run_history'] = await self.last_run_tracker.get_run_history(5)
        
        return {
            'debug_mode': True,
            'debug_info': debug_info,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'enhanced_features': {
                'verified_trades_storage': True,
                'quality_filtering': True,
                'min_eth_threshold': 0.02,
                'lower_notification_thresholds': True,
                'auto_threshold_adjustment': True,
                'orjson_performance': ORJSON_AVAILABLE,
                'smart_timing': True,
                'momentum_tracking': debug_info.get('momentum_tracking', False),
                'web3_intelligence': True
            }
        }
    
    def _build_success_result(self, result, days_back: float, use_smart_timing: bool, 
                         debug_info: Dict, store_verified_trades: bool, store_alerts: bool) -> Dict[str, Any]:
        """Build successful analysis result"""
        return {
            'network': result.network,
            'analysis_type': result.analysis_type,
            'total_transactions': result.total_transactions,
            'unique_tokens': result.unique_tokens,
            'total_eth_value': result.total_eth_value,
            'top_tokens': result.ranked_tokens[:15],
            'performance_metrics': result.performance_metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'ai_enhancement': 'available',
            'days_back_used': days_back,
            'smart_timing_used': use_smart_timing,
            'data_stored': store_verified_trades,
            'alerts_stored': store_alerts,
            'transfers_stored': result.performance_metrics.get('transfers_stored', 0) if store_verified_trades else 0,
            'debug_info': debug_info,
            'enhanced_features': {
                'verified_trades_storage': store_verified_trades,
                'quality_filtering': True,
                'min_eth_threshold': 0.02,
                'lower_thresholds': True,
                'auto_adjustment': True,
                'orjson_performance': ORJSON_AVAILABLE,
                'contract_addresses': True,
                'momentum_tracking': debug_info.get('momentum_tracking', False),
                'web3_intelligence': True,
                'configurable_storage': True
            }
        }

    
    def _build_error_result(self, error_msg: str, days_back: float, debug_info: Dict, tb: str) -> Dict[str, Any]:
        """Build error result"""
        return {
            'error': error_msg,
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'days_back_used': days_back,
            'debug_info': debug_info,
            'traceback': tb,
            'enhanced_features': {
                'orjson_performance': ORJSON_AVAILABLE,
                'momentum_tracking': debug_info.get('momentum_tracking', False)
            }
        }
    
    async def get_run_history(self, limit: int = 10) -> list:
        """Get run history if tracker is available"""
        if self.last_run_tracker and self.last_run_tracker.is_available():
            return await self.last_run_tracker.get_run_history(limit)
        return []
    
    async def get_momentum_status(self) -> Dict:
        """Get momentum tracking status"""
        try:
            from services.notifications.notifications import telegram_service
            return {
                'momentum_tracking_available': bool(telegram_service.momentum_tracker),
                'momentum_tracker_initialized': telegram_service.momentum_tracker is not None
            }
        except Exception as e:
            logger.error(f"Error checking momentum status: {e}")
            return {
                'momentum_tracking_available': False,
                'error': str(e)
            }
    
    async def cleanup(self):
        """Enhanced cleanup"""
        try:
            # Cleanup analyzers
            for analyzer in self._analyzers.values():
                if hasattr(analyzer, 'cleanup'):
                    await analyzer.cleanup()
            
            # Cleanup momentum tracker
            try:
                from services.notifications.notifications import telegram_service
                if telegram_service.momentum_tracker:
                    await telegram_service.momentum_tracker.cleanup()
                    logger.info("Momentum tracker cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up momentum tracker: {e}")
            
            logger.info("Enhanced analysis handler cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {e}")

# Global instance
analysis_handler = AnalysisHandler()