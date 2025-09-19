import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import asyncio
import concurrent.futures
from utils.config import Config

logger = logging.getLogger(__name__)

class AlertMomentumTracker:
    """Track alerts over time with score-based momentum calculation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.table_id = f"{config.bigquery_project_id}.{config.bigquery_dataset_id}.alert_history"
        self.retention_days = 5
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Momentum scoring weights
        self.momentum_weights = {
            'buy_multiplier': 1.0,      # Buy scores add full value
            'sell_multiplier': -0.8,    # Sell scores subtract
            'time_decay_factor': 0.95,  # Newer alerts have more weight
            'volume_boost': 0.1,        # ETH volume adds bonus
            'wallet_quality_boost': 0.05 # High-scoring wallets add bonus
        }
        
    async def initialize(self):
        """Initialize BigQuery client and ensure table exists"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            logger.info("Alert momentum tracker initialized with score-based analysis")
        except Exception as e:
            logger.error(f"Failed to initialize alert tracker: {e}")
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization"""
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create or update alert history table"""
        try:
            # Try to get existing table
            table = self.client.get_table(self.table_id)
            logger.info(f"Alert history table exists: {self.table_id}")
            
            # Check if we need to add new fields (for momentum upgrade)
            existing_schema = [field.name for field in table.schema]
            
            # New fields for momentum tracking
            new_fields = [
                'alert_score', 'volume_score', 'quality_score', 'ai_score', 
                'momentum_score', 'transaction_count', 'avg_wallet_score', 
                'smart_money_ratio', 'time_weight'
            ]
            
            missing_fields = [field for field in new_fields if field not in existing_schema]
            
            if missing_fields:
                logger.info(f"Adding new momentum fields to existing table: {missing_fields}")
                # For now, we'll work with existing schema and add fields gradually
                # BigQuery allows schema evolution but we'll keep it simple
            
        except NotFound:
            logger.info(f"Creating new alert history table: {self.table_id}")
            
            # Create new table with basic schema (compatible with existing data)
            schema = [
                bigquery.SchemaField("alert_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("token_symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("contract_address", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("network", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("analysis_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("alert_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("score", "FLOAT64", mode="REQUIRED"),  # Keep original field name
                bigquery.SchemaField("eth_volume", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("wallet_count", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("confidence", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("ai_enhanced", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("is_verified", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("has_liquidity", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("honeypot_risk", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("whale_coordination", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("pump_signals", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("smart_money_active", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            table = bigquery.Table(self.table_id, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="alert_timestamp"
            )
            table.clustering_fields = ["token_symbol", "network", "analysis_type"]
            
            table = self.client.create_table(table)
            logger.info(f"Created alert history table: {table.table_id}")

    async def store_alert(self, alert_data: Dict) -> bool:
        """Store alert with momentum scoring"""
        try:
            if not self.client:
                return False
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self._sync_store_alert,
                alert_data
            )
            return success
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            return False
    
    def _sync_store_alert(self, alert_data: Dict) -> bool:
        """Store alert with basic momentum scoring"""
        try:
            data = alert_data.get('data', {})
            ai_data = alert_data.get('ai_data', {})
            
            alert_id = f"{alert_data['token']}_{alert_data['network']}_{alert_data['alert_type']}_{int(datetime.now().timestamp())}"
            
            # FIXED: Safe type conversion functions
            def safe_float(value, default=0.0):
                """Safely convert to float, handling numpy types"""
                try:
                    if hasattr(value, 'item'):  # numpy scalar
                        return float(value.item())
                    elif value is None:
                        return default
                    else:
                        return float(value)
                except (ValueError, TypeError, AttributeError):
                    return default
            
            def safe_int(value, default=0):
                """Safely convert to int, handling numpy types"""
                try:
                    if hasattr(value, 'item'):  # numpy scalar
                        return int(value.item())
                    elif value is None:
                        return default
                    else:
                        return int(value)
                except (ValueError, TypeError, AttributeError):
                    return default
            
            def safe_bool(value, default=False):
                """Safely convert to bool, handling numpy types"""
                try:
                    if hasattr(value, 'item'):  # numpy bool
                        return bool(value.item())
                    elif value is None:
                        return default
                    else:
                        return bool(value)
                except (ValueError, TypeError, AttributeError):
                    return default
            
            def safe_str(value, default=""):
                """Safely convert to string"""
                try:
                    if value is None:
                        return default
                    else:
                        return str(value)
                except:
                    return default
            
            # Extract and safely convert main score
            main_score = safe_float(data.get('alpha_score', data.get('sell_pressure_score', data.get('total_score', 0))))
            
            # Calculate simple momentum score
            if alert_data['alert_type'] == 'buy':
                momentum_score = main_score * 1.0  # Positive for buys
            else:  # sell
                momentum_score = main_score * -0.8  # Negative for sells
            
            # FIXED: Convert all values with safe conversion
            eth_volume = safe_float(data.get('total_eth_spent', data.get('total_eth_received', data.get('total_eth_value', 0))))
            wallet_count = safe_int(data.get('wallet_count', 0))
            confidence_value = safe_float(ai_data.get('confidence')) if ai_data.get('confidence') is not None else None
            
            # Build row data with safe type conversion
            row_data = {
                "alert_id": safe_str(alert_id),
                "token_symbol": safe_str(alert_data['token']),
                "contract_address": safe_str(data.get('contract_address', data.get('ca', ''))),
                "network": safe_str(alert_data['network']),
                "analysis_type": safe_str(alert_data['alert_type']),
                "alert_timestamp": datetime.utcnow().isoformat(),
                "score": safe_float(main_score),
                "eth_volume": eth_volume,
                "wallet_count": wallet_count,
                "confidence": confidence_value,
                "ai_enhanced": safe_bool(ai_data.get('ai_enhanced', False)),
                "is_verified": safe_bool(ai_data.get('is_verified')) if 'is_verified' in ai_data else None,
                "has_liquidity": safe_bool(ai_data.get('has_liquidity')) if 'has_liquidity' in ai_data else None,
                "honeypot_risk": safe_float(ai_data.get('honeypot_risk', 0)) if ai_data.get('honeypot_risk') else None,
                "whale_coordination": safe_bool(ai_data.get('whale_coordination_detected')),
                "pump_signals": safe_bool(ai_data.get('pump_signals_detected')),
                "smart_money_active": safe_bool(ai_data.get('has_smart_money')),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store to BigQuery
            table = self.client.get_table(self.table_id)
            errors = self.client.insert_rows_json(table, [row_data])
            
            if not errors:
                logger.info(f"Stored momentum alert: {alert_data['token']} ({alert_data['alert_type']}) - Score: {momentum_score:.1f}")
                return True
            else:
                logger.error(f"Error inserting alert: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Alert storage failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    async def get_token_momentum(self, token_symbol: str, network: str, days_back: int = 5) -> Dict:
        """Get momentum analysis with combined scores"""
        try:
            if not self.client:
                return {}
            
            loop = asyncio.get_event_loop()
            momentum = await loop.run_in_executor(
                self.executor,
                self._sync_get_token_momentum,
                token_symbol,
                network,
                days_back
            )
            return momentum
            
        except Exception as e:
            logger.error(f"Error getting momentum: {e}")
            return {}
    
    def _sync_get_token_momentum(self, token_symbol: str, network: str, days_back: int) -> Dict:
        """Calculate momentum using existing schema fields"""
        try:
            query = f"""
            SELECT 
                alert_timestamp,
                analysis_type,
                score,
                eth_volume,
                wallet_count,
                confidence,
                whale_coordination,
                pump_signals,
                smart_money_active
            FROM `{self.table_id}`
            WHERE token_symbol = @token_symbol
            AND network = @network
            AND alert_timestamp >= @cutoff_date
            ORDER BY alert_timestamp DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("token_symbol", "STRING", token_symbol),
                    bigquery.ScalarQueryParameter("network", "STRING", network),
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", 
                                                datetime.utcnow() - timedelta(days=days_back))
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if not results:
                return {"momentum_detected": False, "net_momentum_score": 0, "alert_count": 0}
            
            # Calculate momentum using scores
            buy_momentum = 0
            sell_momentum = 0
            total_alerts = len(results)
            
            buy_alerts = []
            sell_alerts = []
            
            for result in results:
                # Apply time decay
                hours_ago = (datetime.utcnow() - result.alert_timestamp.replace(tzinfo=None)).total_seconds() / 3600
                time_decay = 0.95 ** (hours_ago / 24)  # Daily decay
                
                score = float(result.score)
                
                if result.analysis_type == 'buy':
                    adjusted_score = score * time_decay
                    buy_momentum += adjusted_score
                    buy_alerts.append(result)
                else:  # sell
                    adjusted_score = score * 0.8 * time_decay  # Discount sell scores
                    sell_momentum += adjusted_score
                    sell_alerts.append(result)
            
            # Net momentum = Buys - Sells
            net_momentum_score = buy_momentum - sell_momentum
            
            # Momentum classification
            momentum_strength = "NEUTRAL"
            if net_momentum_score >= 100:
                momentum_strength = "VERY_BULLISH"
            elif net_momentum_score >= 50:
                momentum_strength = "BULLISH"
            elif net_momentum_score >= 20:
                momentum_strength = "SLIGHTLY_BULLISH"
            elif net_momentum_score <= -50:
                momentum_strength = "BEARISH"
            elif net_momentum_score <= -20:
                momentum_strength = "SLIGHTLY_BEARISH"
            
            # Calculate velocity
            if len(results) >= 2:
                recent_alerts = [r for r in results if (datetime.utcnow() - r.alert_timestamp.replace(tzinfo=None)).total_seconds() / 3600 <= 12]
                momentum_velocity = len(recent_alerts) / max(len(results), 1)
            else:
                momentum_velocity = 0
            
            # Detect signals
            has_whale_activity = any(r.whale_coordination for r in results if r.whale_coordination)
            has_pump_signals = any(r.pump_signals for r in results if r.pump_signals)
            has_smart_money = any(r.smart_money_active for r in results if r.smart_money_active)
            
            # Calculate averages
            avg_confidence = sum(float(r.confidence) for r in results if r.confidence) / len([r for r in results if r.confidence]) if any(r.confidence for r in results) else 0
            total_volume = sum(float(r.eth_volume) for r in results)
            
            return {
                "momentum_detected": abs(net_momentum_score) >= 20,
                "net_momentum_score": round(net_momentum_score, 2),
                "momentum_strength": momentum_strength,
                "momentum_velocity": round(momentum_velocity, 3),
                "buy_momentum": round(buy_momentum, 2),
                "sell_momentum": round(sell_momentum, 2),
                "alert_count": total_alerts,
                "buy_alerts": len(buy_alerts),
                "sell_alerts": len(sell_alerts),
                "total_volume": round(total_volume, 4),
                "avg_confidence": round(avg_confidence, 3),
                "days_analyzed": days_back,
                "whale_activity": has_whale_activity,
                "pump_activity": has_pump_signals,
                "smart_money_activity": has_smart_money,
                "trending_direction": "UP" if net_momentum_score > 0 else "DOWN" if net_momentum_score < 0 else "SIDEWAYS"
            }
            
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    async def get_trending_tokens(self, network: str = None, hours_back: int = 24, limit: int = 10) -> List[Dict]:
        """Get trending tokens with momentum scoring"""
        try:
            if not self.client:
                return []
            
            loop = asyncio.get_event_loop()
            trending = await loop.run_in_executor(
                self.executor,
                self._sync_get_trending_tokens,
                network,
                hours_back,
                limit
            )
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending tokens: {e}")
            return []
    
    def _sync_get_trending_tokens(self, network: str, hours_back: int, limit: int) -> List[Dict]:
        """Get trending tokens using existing schema"""
        try:
            conditions = ["alert_timestamp >= @cutoff_time"]
            params = [
                bigquery.ScalarQueryParameter("cutoff_time", "TIMESTAMP",
                                            datetime.utcnow() - timedelta(hours=hours_back)),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
            
            if network:
                conditions.append("network = @network")
                params.append(bigquery.ScalarQueryParameter("network", "STRING", network))
            
            where_clause = " AND ".join(conditions)
            
            # Use existing schema fields
            query = f"""
            SELECT 
                token_symbol,
                network,
                COUNT(*) as alert_count,
                
                -- Calculate momentum using existing score field
                SUM(CASE WHEN analysis_type = 'buy' THEN score ELSE 0 END) as buy_momentum,
                SUM(CASE WHEN analysis_type = 'sell' THEN score * 0.8 ELSE 0 END) as sell_momentum,
                SUM(CASE WHEN analysis_type = 'buy' THEN score ELSE -score * 0.8 END) as net_momentum,
                
                -- Other metrics
                SUM(eth_volume) as total_volume,
                AVG(confidence) as avg_confidence,
                
                -- Counts
                COUNT(CASE WHEN analysis_type = 'buy' THEN 1 END) as buy_count,
                COUNT(CASE WHEN analysis_type = 'sell' THEN 1 END) as sell_count,
                
                -- Latest activity
                MAX(alert_timestamp) as latest_alert,
                
                -- Signal detection
                SUM(CASE WHEN whale_coordination THEN 1 ELSE 0 END) as whale_signals,
                SUM(CASE WHEN pump_signals THEN 1 ELSE 0 END) as pump_signals,
                SUM(CASE WHEN smart_money_active THEN 1 ELSE 0 END) as smart_money_signals
                
            FROM `{self.table_id}`
            WHERE {where_clause}
            GROUP BY token_symbol, network
            HAVING alert_count >= 1
            ORDER BY 
                ABS(net_momentum) DESC,
                total_volume DESC,
                alert_count DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            trending = []
            for row in results:
                net_momentum = float(row.net_momentum) if row.net_momentum else 0
                
                # Determine momentum indicator
                if net_momentum >= 50:
                    momentum_indicator = "🚀 STRONG BUY"
                elif net_momentum >= 20:
                    momentum_indicator = "📈 BULLISH"
                elif net_momentum >= 5:
                    momentum_indicator = "⬆️ SLIGHT BUY"
                elif net_momentum <= -50:
                    momentum_indicator = "📉 STRONG SELL"
                elif net_momentum <= -20:
                    momentum_indicator = "⬇️ BEARISH"
                elif net_momentum <= -5:
                    momentum_indicator = "📉 SLIGHT SELL"
                else:
                    momentum_indicator = "➡️ NEUTRAL"
                
                trending.append({
                    "token_symbol": row.token_symbol,
                    "network": row.network,
                    "net_momentum_score": round(net_momentum, 2),
                    "buy_momentum": round(float(row.buy_momentum), 2),
                    "sell_momentum": round(float(row.sell_momentum), 2),
                    "momentum_indicator": momentum_indicator,
                    "alert_count": row.alert_count,
                    "buy_count": row.buy_count,
                    "sell_count": row.sell_count,
                    "total_volume": round(float(row.total_volume), 4),
                    "avg_confidence": round(float(row.avg_confidence) if row.avg_confidence else 0, 3),
                    "latest_alert": row.latest_alert.isoformat(),
                    "whale_signals": row.whale_signals,
                    "pump_signals": row.pump_signals,
                    "smart_money_signals": row.smart_money_signals,
                    "hours_tracked": hours_back
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Trending calculation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def cleanup_old_alerts(self):
        """Remove alerts older than retention period"""
        try:
            if not self.client:
                return
            
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            query = f"""
            DELETE FROM `{self.table_id}`
            WHERE alert_timestamp < @cutoff_date
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", cutoff_date)
                ]
            )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(query, job_config=job_config).result()
            )
            
            logger.info(f"Cleaned up alerts older than {self.retention_days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()
        if self.executor:
            self.executor.shutdown(wait=True)