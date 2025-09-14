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
    """Track alerts over time to identify momentum patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.table_id = f"{config.bigquery_project_id}.{config.bigquery_dataset_id}.alert_history"
        self.retention_days = 5  # Keep 5 days of alert history
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self):
        """Initialize BigQuery client and ensure table exists"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            logger.info("Alert momentum tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize alert tracker: {e}")
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization"""
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create alert history table if it doesn't exist"""
        try:
            self.client.get_table(self.table_id)
            logger.info(f"Alert history table exists: {self.table_id}")
        except NotFound:
            logger.info(f"Creating alert history table: {self.table_id}")
            
            schema = [
                bigquery.SchemaField("alert_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("token_symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("contract_address", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("network", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("analysis_type", "STRING", mode="REQUIRED"),  # 'buy' or 'sell'
                bigquery.SchemaField("alert_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("score", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("eth_volume", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("wallet_count", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("confidence", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("ai_enhanced", "BOOLEAN", mode="REQUIRED"),
                
                # Web3 intelligence fields
                bigquery.SchemaField("is_verified", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("has_liquidity", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("honeypot_risk", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("token_age_hours", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("holder_count", "INTEGER", mode="NULLABLE"),
                
                # Momentum indicators
                bigquery.SchemaField("whale_coordination", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("pump_signals", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("smart_money_active", "BOOLEAN", mode="NULLABLE"),
                
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            table = bigquery.Table(self.table_id, schema=schema)
            
            # Partition by day for efficient queries and cleanup
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="alert_timestamp"
            )
            
            # Cluster for momentum queries
            table.clustering_fields = ["token_symbol", "network", "analysis_type"]
            
            table = self.client.create_table(table)
            logger.info(f"Created alert history table: {table.table_id}")
    
    async def store_alert(self, alert_data: Dict) -> bool:
        """Store an alert for momentum tracking"""
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
        """Synchronously store alert"""
        try:
            # Extract alert information
            data = alert_data.get('data', {})
            ai_data = alert_data.get('ai_data', {})
            
            # Generate unique alert ID
            alert_id = f"{alert_data['token']}_{alert_data['network']}_{alert_data['alert_type']}_{int(datetime.now().timestamp())}"
            
            # Build row data
            row_data = {
                "alert_id": alert_id,
                "token_symbol": alert_data['token'],
                "contract_address": data.get('contract_address') or data.get('ca', ''),
                "network": alert_data['network'],
                "analysis_type": alert_data['alert_type'],
                "alert_timestamp": datetime.utcnow().isoformat(),
                "score": float(data.get('alpha_score', data.get('sell_pressure_score', data.get('total_score', 0)))),
                "eth_volume": float(data.get('total_eth_spent', data.get('total_eth_received', data.get('total_eth_value', 0)))),
                "wallet_count": int(data.get('wallet_count', 0)),
                "confidence": float(ai_data.get('confidence', 0.5)) if ai_data.get('confidence') else None,
                "ai_enhanced": bool(ai_data.get('ai_enhanced', False)),
                
                # Web3 intelligence
                "is_verified": ai_data.get('is_verified') if 'is_verified' in ai_data else None,
                "has_liquidity": ai_data.get('has_liquidity') if 'has_liquidity' in ai_data else None,
                "honeypot_risk": float(ai_data.get('honeypot_risk', 0)) if ai_data.get('honeypot_risk') else None,
                "token_age_hours": float(ai_data.get('token_age_hours')) if ai_data.get('token_age_hours') else None,
                "holder_count": int(ai_data.get('holder_count', 0)) if ai_data.get('holder_count') else None,
                
                # Momentum signals
                "whale_coordination": ai_data.get('whale_coordination_detected'),
                "pump_signals": ai_data.get('pump_signals_detected'), 
                "smart_money_active": ai_data.get('has_smart_money'),
                
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Insert into BigQuery
            table = self.client.get_table(self.table_id)
            errors = self.client.insert_rows_json(table, [row_data])
            
            if not errors:
                logger.info(f"Stored alert for momentum tracking: {alert_data['token']}")
                return True
            else:
                logger.error(f"Error inserting alert: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Sync store alert failed: {e}")
            return False
    
    async def get_token_momentum(self, token_symbol: str, network: str, days_back: int = 5) -> Dict:
        """Get momentum analysis for a specific token"""
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
            logger.error(f"Error getting token momentum: {e}")
            return {}
    
    def _sync_get_token_momentum(self, token_symbol: str, network: str, days_back: int) -> Dict:
        """Synchronous momentum analysis"""
        try:
            query = f"""
            SELECT 
                alert_timestamp,
                analysis_type,
                score,
                eth_volume,
                confidence,
                ai_enhanced,
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
                return {"momentum_detected": False, "alert_count": 0}
            
            # Analyze momentum
            buy_alerts = [r for r in results if r.analysis_type == 'buy']
            sell_alerts = [r for r in results if r.analysis_type == 'sell']
            
            total_alerts = len(results)
            avg_score = sum(r.score for r in results) / total_alerts
            total_volume = sum(r.eth_volume for r in results)
            
            # Momentum indicators
            momentum_score = 0
            if total_alerts >= 2:  # Multiple alerts = momentum
                momentum_score += 30
            if len(buy_alerts) > len(sell_alerts):  # More buys than sells
                momentum_score += 25
            if avg_score > 50:  # High average score
                momentum_score += 20
            if any(r.whale_coordination for r in results if r.whale_coordination):
                momentum_score += 15
            if any(r.pump_signals for r in results if r.pump_signals):
                momentum_score += 10
            
            momentum_detected = momentum_score >= 50
            
            return {
                "momentum_detected": momentum_detected,
                "momentum_score": momentum_score,
                "alert_count": total_alerts,
                "buy_alerts": len(buy_alerts),
                "sell_alerts": len(sell_alerts), 
                "avg_score": avg_score,
                "total_volume": total_volume,
                "days_analyzed": days_back,
                "whale_activity": any(r.whale_coordination for r in results if r.whale_coordination),
                "pump_activity": any(r.pump_signals for r in results if r.pump_signals),
                "smart_money_activity": any(r.smart_money_active for r in results if r.smart_money_active)
            }
            
        except Exception as e:
            logger.error(f"Sync momentum analysis failed: {e}")
            return {}
    
    async def get_trending_tokens(self, network: str = None, hours_back: int = 24, limit: int = 10) -> List[Dict]:
        """Get tokens with the most momentum in recent hours"""
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
        """Get trending tokens with momentum"""
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
            
            query = f"""
            SELECT 
                token_symbol,
                network,
                COUNT(*) as alert_count,
                AVG(score) as avg_score,
                SUM(eth_volume) as total_volume,
                COUNT(DISTINCT analysis_type) as signal_diversity,
                MAX(alert_timestamp) as latest_alert,
                COALESCE(SUM(CASE WHEN whale_coordination THEN 1 ELSE 0 END), 0) as whale_signals,
                COALESCE(SUM(CASE WHEN pump_signals THEN 1 ELSE 0 END), 0) as pump_signals
            FROM `{self.table_id}`
            WHERE {where_clause}
            GROUP BY token_symbol, network
            HAVING alert_count >= 2  -- Must have at least 2 alerts for momentum
            ORDER BY 
                alert_count DESC,
                avg_score DESC,
                total_volume DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            trending = []
            for row in results:
                momentum_score = (
                    row.alert_count * 20 +  # More alerts = more momentum
                    (row.avg_score / 10) +  # Higher scores = better momentum
                    row.signal_diversity * 15 +  # Both buy/sell signals = interesting
                    row.whale_signals * 10 +
                    row.pump_signals * 5
                )
                
                trending.append({
                    "token_symbol": row.token_symbol,
                    "network": row.network,
                    "momentum_score": momentum_score,
                    "alert_count": row.alert_count,
                    "avg_score": float(row.avg_score),
                    "total_volume": float(row.total_volume),
                    "latest_alert": row.latest_alert.isoformat(),
                    "whale_signals": row.whale_signals,
                    "pump_signals": row.pump_signals,
                    "hours_tracked": hours_back
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Sync trending tokens failed: {e}")
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