import logging
from datetime import datetime
from typing import Dict, List, Optional
from utils.config import Config

logger = logging.getLogger(__name__)

# Import BigQuery safely
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("BigQuery library not available for last run tracking")

class LastRunTracker:
    """Track last run times for different analysis types using BigQuery"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.table_id = f"{config.bigquery_project_id}.{config.bigquery_dataset_id}.last_runs"
        self._initialized = False
        
    async def initialize(self):
        """Initialize BigQuery client and ensure table exists"""
        if not BIGQUERY_AVAILABLE:
            logger.warning("BigQuery not available - last run tracking disabled")
            return
            
        try:
            self.client = bigquery.Client(project=self.config.bigquery_project_id)
            await self._ensure_table_exists()
            self._initialized = True
            logger.info("Last run tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize last run tracker: {e}")
            self.client = None
    
    async def _ensure_table_exists(self):
        """Create last_runs table if it doesn't exist"""
        try:
            self.client.get_table(self.table_id)
            logger.info(f"Last runs table exists: {self.table_id}")
        except NotFound:
            logger.info(f"Creating last runs table: {self.table_id}")
            
            schema = [
                bigquery.SchemaField("job_key", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("network", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("analysis_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("last_run_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("days_back_used", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            
            table = bigquery.Table(self.table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created last runs table: {table.table_id}")
    
    def _get_job_key(self, network: str, analysis_type: str) -> str:
        """Generate unique job key"""
        return f"{network}_{analysis_type}"
    
    async def get_days_since_last_run(self, network: str, analysis_type: str, default_days: float = 1.0) -> float:
        """Calculate days since last successful run"""
        if not self.client:
            logger.warning("Last run tracker not available, using default days")
            return default_days
        
        try:
            job_key = self._get_job_key(network, analysis_type)
            
            query = f"""
            SELECT last_run_time, days_back_used
            FROM `{self.table_id}`
            WHERE job_key = @job_key
              AND status = 'success'
            ORDER BY last_run_time DESC
            LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("job_key", "STRING", job_key)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if results:
                last_run_time = results[0].last_run_time
                
                # Calculate time difference
                now = datetime.utcnow()
                time_diff = now - last_run_time.replace(tzinfo=None)
                days_since = time_diff.total_seconds() / (24 * 3600)
                
                # Round to 1 significant figure
                days_back = self._round_to_one_sig_fig(days_since)
                
                logger.info(f"Last run was {days_since:.3f} days ago, using {days_back} days back")
                return max(days_back, 0.1)  # Minimum 0.1 days (2.4 hours)
            else:
                logger.info(f"No previous run found for {job_key}, using default {default_days} days")
                return default_days
                
        except Exception as e:
            logger.error(f"Error getting last run time: {e}")
            return default_days
    
    def _round_to_one_sig_fig(self, value: float) -> float:
        """Round to 1 significant figure"""
        if value <= 0:
            return 0.1
        
        import math
        
        # Find the order of magnitude
        magnitude = math.floor(math.log10(abs(value)))
        
        # Scale to get the first significant digit
        scaled = value / (10 ** magnitude)
        
        # Round to nearest integer and scale back
        rounded = round(scaled) * (10 ** magnitude)
        
        return max(rounded, 0.1)  # Ensure minimum
    
    async def record_run(self, network: str, analysis_type: str, days_back: float, status: str = "success"):
        """Record a completed run"""
        if not self.client:
            logger.warning("Last run tracker not available, cannot record run")
            return
        
        try:
            job_key = self._get_job_key(network, analysis_type)
            now = datetime.utcnow()
            
            # Insert new record
            row_data = {
                "job_key": job_key,
                "network": network,
                "analysis_type": analysis_type,
                "last_run_time": now.isoformat(),
                "days_back_used": float(days_back),
                "status": status,
                "created_at": now.isoformat()
            }
            
            table = self.client.get_table(self.table_id)
            errors = self.client.insert_rows_json(table, [row_data])
            
            if not errors:
                logger.info(f"Recorded run: {job_key} at {now.isoformat()} ({days_back} days back)")
            else:
                logger.error(f"Error recording run: {errors}")
                
        except Exception as e:
            logger.error(f"Error recording run: {e}")
    
    async def get_run_history(self, limit: int = 10) -> List[Dict]:
        """Get recent run history"""
        if not self.client:
            return []
        
        try:
            query = f"""
            SELECT job_key, network, analysis_type, last_run_time, days_back_used, status
            FROM `{self.table_id}`
            ORDER BY last_run_time DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("limit", "INT64", limit)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            history = []
            for row in results:
                history.append({
                    "job_key": row.job_key,
                    "network": row.network,
                    "analysis_type": row.analysis_type,
                    "last_run_time": row.last_run_time.isoformat(),
                    "days_back_used": float(row.days_back_used),
                    "status": row.status
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting run history: {e}")
            return []

    def is_available(self) -> bool:
        """Check if tracker is available and initialized"""
        return self._initialized and self.client is not None