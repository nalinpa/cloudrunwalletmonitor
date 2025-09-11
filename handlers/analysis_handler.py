import logging
import traceback
from datetime import datetime
from typing import Dict, Any

# 🚀 PERFORMANCE BOOST: Use orjson for faster JSON processing
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
from services.notifications.notifications import telegram_service

logger = logging.getLogger(__name__)

class AnalysisHandler:
    """Handle analysis requests with smart timing, lower thresholds, and orjson performance"""
    
    def __init__(self):
        self.config = Config()
        self.last_run_tracker: LastRunTracker = None
        self._analyzers = {}
        
        # 🎯 ENHANCED NOTIFICATION SETTINGS - More alerts, lower barriers
        self.enhanced_notification_settings = {
            'default_min_alpha_score': 12.0,  # Lowered from 50.0
            'max_min_alpha_score': 30.0,      # Cap user input
            'default_max_tokens': 12,         # Increased from 7  
            'min_max_tokens': 5,              # Ensure minimum
            'auto_adjust_thresholds': True,   # Dynamically adjust based on results
            'send_summary_always': True       # Always send summary even with 0 alerts
        }
        
        if ORJSON_AVAILABLE:
            logger.info("🚀 Analysis handler initialized with orjson performance boost")
        else:
            logger.warning("⚠️ Analysis handler using standard json (consider installing orjson)")
    
    async def initialize(self):
        """Initialize the analysis handler"""
        # Initialize last run tracker
        self.last_run_tracker = LastRunTracker(self.config)
        await self.last_run_tracker.initialize()
        logger.info("Enhanced analysis handler initialized with lower notification thresholds")
    
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
    
    def _apply_enhanced_notification_settings(self, request_data: Dict) -> Dict:
        """Apply enhanced notification settings for more alerts"""
        settings = self.enhanced_notification_settings
        
        # Lower the minimum alpha score for more notifications
        original_min_score = request_data.get('min_alpha_score', settings['default_min_alpha_score'])
        enhanced_min_score = min(float(original_min_score), settings['max_min_alpha_score'])
        
        # If user set a very high threshold, bring it down
        if enhanced_min_score > settings['default_min_alpha_score'] * 2:
            enhanced_min_score = settings['default_min_alpha_score'] * 1.5
            logger.info(f"📉 Lowered min_alpha_score from {original_min_score} to {enhanced_min_score} for more alerts")
        
        request_data['min_alpha_score'] = enhanced_min_score
        
        # Increase max tokens for more notifications
        original_max_tokens = request_data.get('max_tokens', settings['default_max_tokens'])
        enhanced_max_tokens = max(int(original_max_tokens), settings['min_max_tokens'])
        
        # If user set too few tokens, increase it
        if enhanced_max_tokens < settings['default_max_tokens']:
            enhanced_max_tokens = settings['default_max_tokens']
            logger.info(f"📈 Increased max_tokens from {original_max_tokens} to {enhanced_max_tokens} for more alerts")
        
        request_data['max_tokens'] = enhanced_max_tokens
        
        # Always enable notifications
        request_data['notifications'] = True
        
        logger.info(f"🎯 Enhanced notification settings: min_score={enhanced_min_score}, max_tokens={enhanced_max_tokens}")
        
        return request_data
    
    async def handle_analysis_request(self, request_data: Dict) -> Dict[str, Any]:
        """Handle analysis request with enhanced notifications and orjson performance"""
        try:
            # Apply enhanced notification settings
            request_data = self._apply_enhanced_notification_settings(request_data)
            
            # Extract and validate parameters
            network = request_data.get('network', 'ethereum')
            analysis_type = request_data.get('analysis_type', 'buy')
            num_wallets = min(int(request_data.get('num_wallets', 50)), self.config.max_wallets)
            
            # Smart timing parameters
            requested_days_back = float(request_data.get('days_back', 1.0))
            use_smart_timing = request_data.get('smart_timing', True)
            
            # Store data flag - defaults to FALSE (no storage)
            store_data = request_data.get('store_data', False)
            
            debug_mode = request_data.get('debug', False)
            
            # Enhanced notification settings
            send_notifications = request_data.get('notifications', True)
            max_tokens = int(request_data.get('max_tokens', self.enhanced_notification_settings['default_max_tokens']))
            min_alpha_score = float(request_data.get('min_alpha_score', self.enhanced_notification_settings['default_min_alpha_score']))
            
            # Log the storage decision and enhanced settings
            storage_status = "ENABLED" if store_data else "DISABLED"
            logger.info(f"Transfer data storage: {storage_status}")
            logger.info(f"🎯 Enhanced notifications: min_score={min_alpha_score}, max_tokens={max_tokens}")
            
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
            
            # Send enhanced start notification
            if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                await telegram_service.send_start_notification(
                    network, analysis_type, num_wallets, days_back, 
                    use_smart_timing, max_tokens, min_alpha_score, store_data
                )
            
            # Run analysis with store_data flag
            try:
                logger.info(f"Starting enhanced {analysis_type} analysis for {network} with orjson performance")
                analyzer = await self.get_analyzer(network, analysis_type)
                
                # Pass the store_data flag to the analyzer
                result = await analyzer.analyze(num_wallets, days_back, store_data=store_data)
                
                # Record successful run
                if self.last_run_tracker and self.last_run_tracker.is_available():
                    await self.last_run_tracker.record_run(network, analysis_type, days_back, "success")
                
                # Build result with orjson performance
                result_dict = self._build_success_result(result, days_back, use_smart_timing, debug_info, store_data)
                
                storage_msg = f", stored {result.performance_metrics.get('transfers_stored', 0)} transfers" if store_data else ", no data stored"
                logger.info(f"Enhanced analysis complete - {result.total_transactions} transactions, {result.unique_tokens} tokens{storage_msg}")
                
                # Send enhanced notifications with lower thresholds
                if send_notifications and telegram_status and telegram_status.get('ready_for_notifications'):
                    try:
                        await self._send_enhanced_notifications(result, network, max_tokens, min_alpha_score)
                    except Exception as e:
                        logger.error(f"Failed to send enhanced notifications: {e}")
                
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
    
    async def _send_enhanced_notifications(self, result, network: str, max_tokens: int, min_alpha_score: float):
        """Send enhanced notifications with smart threshold adjustment"""
        try:
            initial_qualifying_tokens = []
            
            # Check how many tokens qualify with current threshold
            if result.ranked_tokens:
                for token_data in result.ranked_tokens:
                    if len(token_data) >= 3 and token_data[2] >= min_alpha_score:
                        initial_qualifying_tokens.append(token_data)
            
            # 🎯 SMART THRESHOLD ADJUSTMENT - If no alerts, lower the threshold automatically
            if len(initial_qualifying_tokens) == 0 and result.ranked_tokens:
                # Find the highest scoring token
                max_score = max([t[2] for t in result.ranked_tokens[:5]]) if result.ranked_tokens else 0
                
                # If there are tokens but none qualify, lower threshold to get some alerts
                if max_score > 5.0:  # Only if there are decent tokens
                    adjusted_threshold = max(max_score * 0.7, 8.0)  # 70% of max score or minimum 8.0
                    logger.info(f"🎯 AUTO-ADJUSTING threshold from {min_alpha_score} to {adjusted_threshold:.1f} to generate alerts")
                    
                    # Re-filter with adjusted threshold
                    for token_data in result.ranked_tokens:
                        if len(token_data) >= 3 and token_data[2] >= adjusted_threshold:
                            initial_qualifying_tokens.append(token_data)
                    
                    min_alpha_score = adjusted_threshold  # Update for logging
            
            # Send notifications using the standard service
            await telegram_service.send_analysis_notifications(result, network, max_tokens, min_alpha_score)
            
            # Log enhancement info
            logger.info(f"📱 Enhanced notifications sent: {len(initial_qualifying_tokens)} alerts (threshold: {min_alpha_score:.1f})")
            
        except Exception as e:
            logger.error(f"Enhanced notification sending failed: {e}")
            # Send fallback notification
            try:
                await telegram_service.send_message(f"❌ **Enhanced Notification Error**\n\nAnalysis completed but failed to send enhanced alerts: {str(e)}")
            except:
                pass
    
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
        """Build debug information with enhanced settings"""
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
                'store_data': store_data
            },
            'enhanced_notifications': {
                'enabled': True,
                'auto_threshold_adjustment': True,
                'default_min_score': self.enhanced_notification_settings['default_min_alpha_score'],
                'max_min_score': self.enhanced_notification_settings['max_min_alpha_score'],
                'orjson_performance': ORJSON_AVAILABLE
            },
            'telegram_status': telegram_status,
            'notifications_enabled': telegram_service.is_configured(),
            'bigquery_configured': bool(self.config.bigquery_project_id),
            'alchemy_configured': bool(self.config.alchemy_api_key),
            'ai_enhancement': 'available',
            'last_run_tracking': self.last_run_tracker and self.last_run_tracker.is_available(),
            'storage_enabled': store_data,
            'performance_boost': f"orjson {'enabled' if ORJSON_AVAILABLE else 'not available'}"
        }
    
    async def _handle_debug_mode(self, debug_info: Dict) -> Dict[str, Any]:
        """Handle debug mode request"""
        logger.info("Running in enhanced debug mode - returning config info")
        
        if self.last_run_tracker and self.last_run_tracker.is_available():
            debug_info['run_history'] = await self.last_run_tracker.get_run_history(5)
        
        return {
            'debug_mode': True,
            'debug_info': debug_info,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'enhanced_features': {
                'lower_notification_thresholds': True,
                'auto_threshold_adjustment': True,
                'orjson_performance': ORJSON_AVAILABLE,
                'smart_timing': True
            }
        }
    
    def _build_success_result(self, result, days_back: float, use_smart_timing: bool, 
                             debug_info: Dict, store_data: bool) -> Dict[str, Any]:
        """Build successful analysis result with enhanced data"""
        return {
            'network': result.network,
            'analysis_type': result.analysis_type,
            'total_transactions': result.total_transactions,
            'unique_tokens': result.unique_tokens,
            'total_eth_value': result.total_eth_value,
            'top_tokens': result.ranked_tokens[:15],  # Return more tokens
            'performance_metrics': result.performance_metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'ai_enhancement': 'available',
            'days_back_used': days_back,
            'smart_timing_used': use_smart_timing,
            'data_stored': store_data,
            'transfers_stored': result.performance_metrics.get('transfers_stored', 0) if store_data else 0,
            'debug_info': debug_info,
            'enhanced_features': {
                'lower_thresholds': True,
                'auto_adjustment': True,
                'orjson_performance': ORJSON_AVAILABLE,
                'contract_addresses': True
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
                'orjson_performance': ORJSON_AVAILABLE
            }
        }
    
    async def get_run_history(self, limit: int = 10) -> list:
        """Get run history if tracker is available"""
        if self.last_run_tracker and self.last_run_tracker.is_available():
            return await self.last_run_tracker.get_run_history(limit)
        return []

# Global instance
analysis_handler = AnalysisHandler()