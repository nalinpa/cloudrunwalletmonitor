# services/database/bigquery_service.py
import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from api.models.data_models import WalletInfo
from utils.config import Config
import concurrent.futures

logger = logging.getLogger(__name__)

# Check if BigQuery is available
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("BigQuery library not available")

class BigQueryService:
    """Unified BigQuery service for wallets and alerts - all-in-one"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[bigquery.Client] = None
        self.dataset_id = config.bigquery_dataset_id
        
        # Table references
        self.wallets_table = "smart_wallets"
        self.alerts_table = "alert_history"
        
        self.wallets_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.wallets_table}"
        self.alerts_table_id = f"{config.bigquery_project_id}.{self.dataset_id}.{self.alerts_table}"
        
        self._initialized = False
        
        # Thread executor for sync BigQuery operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Alert momentum settings
        self.retention_days = 5
        self.momentum_weights = {
            'buy_multiplier': 1.0,
            'sell_multiplier': -0.8,
            'time_decay_factor': 0.95,
        }
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize(self):
        """Initialize BigQuery client and ensure all tables exist"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_initialize)
            
            self._initialized = True
            logger.info("✅ BigQuery service initialized (wallets + alerts)")
            
        except Exception as e:
            logger.error(f"BigQuery service initialization failed: {e}")
            self._initialized = False
            raise
    
    def _sync_initialize(self):
        """Synchronous initialization"""
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery library not available")
        
        # Initialize client
        self.client = bigquery.Client(project=self.config.bigquery_project_id)
        logger.info(f"BigQuery client initialized for project: {self.config.bigquery_project_id}")
        
        # Ensure dataset exists
        self._ensure_dataset_exists()
        
        # Ensure tables exist
        self._ensure_wallets_table_exists()
        self._ensure_alerts_table_exists()
    
    def _ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            dataset_ref = f"{self.config.bigquery_project_id}.{self.dataset_id}"
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_ref} exists")
        except NotFound:
            logger.info(f"Creating dataset {dataset_ref}")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.config.bigquery_location
            self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_ref}")
    
    def _ensure_wallets_table_exists(self):
        """Ensure smart_wallets table exists"""
        try:
            self.client.get_table(self.wallets_table_id)
            logger.info(f"Wallets table exists: {self.wallets_table_id}")
        except NotFound:
            logger.warning(f"Wallets table not found: {self.wallets_table_id}")
            logger.info("Wallets table should be created by wallet scoring system")
    
    def _ensure_alerts_table_exists(self):
        """Ensure alert_history table exists"""
        try:
            self.client.get_table(self.alerts_table_id)
            logger.info(f"Alerts table exists: {self.alerts_table_id}")
        except NotFound:
            logger.info(f"Creating alerts table: {self.alerts_table_id}")
            
            schema = [
                bigquery.SchemaField("alert_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("token_symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("contract_address", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("network", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("analysis_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("alert_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("score", "FLOAT64", mode="REQUIRED"),
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
            
            table = bigquery.Table(self.alerts_table_id, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="alert_timestamp"
            )
            table.clustering_fields = ["token_symbol", "network", "analysis_type"]
            
            self.client.create_table(table)
            logger.info(f"Created alerts table: {self.alerts_table_id}")
    
    # ========================================================================
    # WALLET OPERATIONS
    # ========================================================================
    
    async def get_top_wallets(self, network: str, limit: int = 50) -> List[WalletInfo]:
        """Get top wallets from smart_wallets table"""
        if not self._initialized or not self.client:
            logger.error("BigQuery client not initialized")
            return []
        
        try:
            loop = asyncio.get_event_loop()
            wallets = await loop.run_in_executor(
                self.executor,
                self._sync_get_top_wallets,
                network,
                limit
            )
            return wallets
        except Exception as e:
            logger.error(f"Error getting top wallets: {e}")
            return []
    
    def _sync_get_top_wallets(self, network: str, limit: int) -> List[WalletInfo]:
        """Synchronously get top wallets"""
        try:
            query = f"""
            SELECT 
                address,
                score
            FROM `{self.wallets_table_id}`
            WHERE address IS NOT NULL
              AND address LIKE '0x%'
              AND LENGTH(address) = 42
              AND score IS NOT NULL
              AND score > 0
              AND COALESCE(is_active, TRUE) = TRUE
            ORDER BY score DESC
            LIMIT @row_limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("row_limit", "INT64", limit)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            wallets = []
            for row in results:
                wallet = WalletInfo(
                    address=row.address,
                    score=float(row.score) if row.score else 0.0,
                    network=network
                )
                wallets.append(wallet)
            
            logger.info(f"Retrieved {len(wallets)} wallets from BigQuery")
            return wallets
            
        except Exception as e:
            logger.error(f"Get top wallets failed: {e}")
            return []
    
    async def get_wallet_stats(self) -> Dict:
        """Get wallet statistics"""
        if not self._initialized or not self.client:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.executor,
                self._sync_get_wallet_stats
            )
            return stats
        except Exception as e:
            logger.error(f"Error getting wallet stats: {e}")
            return {}
    
    def _sync_get_wallet_stats(self) -> Dict:
        """Get wallet statistics synchronously"""
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_wallets,
                COUNT(CASE WHEN COALESCE(is_active, TRUE) = TRUE THEN 1 END) as active_wallets,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                COUNT(CASE WHEN score >= 50 THEN 1 END) as high_score_wallets,
                COUNT(CASE WHEN score >= 20 THEN 1 END) as medium_score_wallets
            FROM `{self.wallets_table_id}`
            WHERE address IS NOT NULL 
              AND address LIKE '0x%'
              AND LENGTH(address) = 42
              AND score IS NOT NULL
            """
            
            query_job = self.client.query(query)
            results = list(query_job.result())
            
            if results:
                row = results[0]
                return {
                    "total_wallets": int(row.total_wallets or 0),
                    "active_wallets": int(row.active_wallets or 0),
                    "avg_score": float(row.avg_score or 0),
                    "min_score": int(row.min_score or 0),
                    "max_score": int(row.max_score or 0),
                    "high_score_wallets": int(row.high_score_wallets or 0),
                    "medium_score_wallets": int(row.medium_score_wallets or 0)
                }
            return {}
            
        except Exception as e:
            logger.error(f"Wallet stats query failed: {e}")
            return {}
    
    # ========================================================================
    # ALERT OPERATIONS
    # ========================================================================
    
    async def store_alert(self, alert_data: Dict) -> bool:
        """Store alert with momentum scoring"""
        try:
            if not self.client:
                logger.warning("BigQuery client not initialized - cannot store alert")
                return False
            
            # Validate required fields
            required_fields = ['token', 'network', 'alert_type', 'data']
            missing_fields = [f for f in required_fields if f not in alert_data]
            if missing_fields:
                logger.error(f"Missing required alert fields: {missing_fields}")
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
        """Store alert synchronously"""
        try:
            data = alert_data.get('data', {})
            ai_data = alert_data.get('ai_data', {})
            
            # Generate unique alert ID
            timestamp_ms = int(datetime.now().timestamp() * 1000)
            alert_id = f"{alert_data['token']}_{alert_data['network']}_{alert_data['alert_type']}_{timestamp_ms}"
            
            # Extract main score
            main_score = self._safe_float(
                data.get('alpha_score') or 
                data.get('sell_pressure_score') or 
                data.get('total_score') or 
                ai_data.get('total_score'),
                0.0
            )
            
            # Extract contract address
            contract_address = (
                data.get('contract_address') or 
                data.get('ca') or 
                ai_data.get('contract_address') or 
                ai_data.get('ca') or
                ''
            )
            
            # Build row data
            row_data = {
                "alert_id": str(alert_id),
                "token_symbol": str(alert_data['token']),
                "contract_address": str(contract_address)[:42],
                "network": str(alert_data['network']),
                "analysis_type": str(alert_data['alert_type']),
                "alert_timestamp": datetime.utcnow().isoformat(),
                "score": self._safe_float(main_score),
                "eth_volume": self._safe_float(
                    data.get('total_eth_spent') or 
                    data.get('total_eth_received') or 
                    data.get('total_eth_value')
                ),
                "wallet_count": self._safe_int(data.get('wallet_count')),
                "confidence": self._safe_float(ai_data.get('confidence')),
                "ai_enhanced": self._safe_bool(ai_data.get('ai_enhanced', False)),
                "is_verified": self._safe_bool(ai_data.get('is_verified')),
                "has_liquidity": self._safe_bool(ai_data.get('has_liquidity')),
                "honeypot_risk": self._safe_float(ai_data.get('honeypot_risk')),
                "whale_coordination": self._safe_bool(ai_data.get('whale_coordination_detected')),
                "pump_signals": self._safe_bool(ai_data.get('pump_signals_detected')),
                "smart_money_active": self._safe_bool(ai_data.get('has_smart_money')),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store to BigQuery
            table = self.client.get_table(self.alerts_table_id)
            errors = self.client.insert_rows_json(table, [row_data])
            
            if not errors:
                logger.info(
                    f"✅ Stored alert: {alert_data['token']} "
                    f"({alert_data['alert_type']}) - "
                    f"Score: {main_score:.1f}, "
                    f"ETH: {row_data['eth_volume']:.4f}"
                )
                return True
            else:
                logger.error(f"BigQuery insert errors: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Alert storage failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def get_token_momentum(self, token_symbol: str, network: str, days_back: int = 5) -> Dict:
        """Get momentum analysis for a token"""
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
        """Calculate token momentum synchronously"""
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
            FROM `{self.alerts_table_id}`
            WHERE token_symbol = @token_symbol
            AND network = @network
            AND alert_timestamp >= @cutoff_date
            ORDER BY alert_timestamp DESC
            LIMIT 100
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
                return {
                    "momentum_detected": False, 
                    "net_momentum_score": 0, 
                    "alert_count": 0,
                    "reason": "no_alerts_found"
                }
            
            # Calculate momentum
            buy_momentum = 0
            sell_momentum = 0
            
            for result in results:
                hours_ago = (datetime.utcnow() - result.alert_timestamp.replace(tzinfo=None)).total_seconds() / 3600
                time_decay = self.momentum_weights['time_decay_factor'] ** (hours_ago / 24)
                score = float(result.score)
                
                if result.analysis_type == 'buy':
                    buy_momentum += score * time_decay * self.momentum_weights['buy_multiplier']
                else:
                    sell_momentum += score * time_decay * abs(self.momentum_weights['sell_multiplier'])
            
            net_momentum = buy_momentum - sell_momentum
            
            # Classification
            if net_momentum >= 100:
                strength, emoji = "VERY_BULLISH", "🚀"
            elif net_momentum >= 50:
                strength, emoji = "BULLISH", "📈"
            elif net_momentum >= 20:
                strength, emoji = "SLIGHTLY_BULLISH", "⬆️"
            elif net_momentum <= -50:
                strength, emoji = "BEARISH", "📉"
            elif net_momentum <= -20:
                strength, emoji = "SLIGHTLY_BEARISH", "⬇️"
            else:
                strength, emoji = "NEUTRAL", "➡️"
            
            return {
                "momentum_detected": abs(net_momentum) >= 20,
                "net_momentum_score": round(net_momentum, 2),
                "momentum_strength": strength,
                "momentum_emoji": emoji,
                "buy_momentum": round(buy_momentum, 2),
                "sell_momentum": round(sell_momentum, 2),
                "alert_count": len(results),
                "days_analyzed": days_back
            }
            
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            return {"momentum_detected": False, "error": str(e)}
    
    async def get_trending_tokens(self, network: str = None, hours_back: int = 24, limit: int = 10) -> List[Dict]:
        """Get trending tokens"""
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
        """Get trending tokens synchronously"""
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
                SUM(CASE WHEN analysis_type = 'buy' THEN score ELSE 0 END) as buy_momentum,
                SUM(CASE WHEN analysis_type = 'sell' THEN score * 0.8 ELSE 0 END) as sell_momentum,
                SUM(CASE WHEN analysis_type = 'buy' THEN score ELSE -score * 0.8 END) as net_momentum,
                SUM(eth_volume) as total_volume,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN analysis_type = 'buy' THEN 1 END) as buy_count,
                COUNT(CASE WHEN analysis_type = 'sell' THEN 1 END) as sell_count,
                MAX(alert_timestamp) as latest_alert
            FROM `{self.alerts_table_id}`
            WHERE {where_clause}
            GROUP BY token_symbol, network
            HAVING alert_count >= 1
            ORDER BY ABS(net_momentum) DESC, total_volume DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            trending = []
            for row in results:
                net_momentum = float(row.net_momentum) if row.net_momentum else 0
                
                if net_momentum >= 50:
                    indicator = "🚀 STRONG BUY"
                elif net_momentum >= 20:
                    indicator = "📈 BULLISH"
                elif net_momentum <= -50:
                    indicator = "📉 STRONG SELL"
                elif net_momentum <= -20:
                    indicator = "⬇️ BEARISH"
                else:
                    indicator = "➡️ NEUTRAL"
                
                trending.append({
                    "token_symbol": row.token_symbol,
                    "network": row.network,
                    "net_momentum_score": round(net_momentum, 2),
                    "momentum_indicator": indicator,
                    "alert_count": row.alert_count,
                    "buy_count": row.buy_count,
                    "sell_count": row.sell_count,
                    "total_volume": round(float(row.total_volume), 4),
                    "hours_tracked": hours_back
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Trending calculation failed: {e}")
            return []
    
    async def cleanup_old_alerts(self):
        """Remove alerts older than retention period"""
        try:
            if not self.client:
                return
            
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            query = f"""
            DELETE FROM `{self.alerts_table_id}`
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
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert to float"""
        if value is None:
            return default
        try:
            if hasattr(value, 'item'):
                return float(value.item())
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert to int"""
        if value is None:
            return default
        try:
            if hasattr(value, 'item'):
                return int(value.item())
            return int(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    def _safe_bool(self, value: Any, default: bool = False) -> bool:
        """Safely convert to bool"""
        if value is None:
            return default
        try:
            if hasattr(value, 'item'):
                return bool(value.item())
            return bool(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    async def test_connection(self) -> bool:
        """Test BigQuery connection"""
        try:
            if not self.client:
                return False
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_test_connection
            )
            return result
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _sync_test_connection(self) -> bool:
        """Test connection synchronously"""
        try:
            query = f"SELECT COUNT(*) as count FROM `{self.wallets_table_id}` LIMIT 1"
            query_job = self.client.query(query)
            results = list(query_job.result())
            return len(results) > 0
        except Exception as e:
            logger.error(f"Connection test query failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup BigQuery client"""
        if self.client:
            try:
                self.client.close()
                logger.info("BigQuery connection closed")
            except Exception as e:
                logger.error(f"Error closing BigQuery connection: {e}")
            finally:
                self.client = None
                self._initialized = False
        
        if self.executor:
            self.executor.shutdown(wait=True)