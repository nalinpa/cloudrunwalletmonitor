import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Import your services
from services.notifications import (
    send_alert_notifications, 
    send_test_notification, 
    check_notification_config,
    format_alert_message
)
from core.analysis.buy_analyzer import BuyAnalyzer
from core.analysis.sell_analyzer import SellAnalyzer

# Load environment variables
load_dotenv('.env.production')

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/crypto_monitor.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Global variables for monitoring
monitoring_active = False
analysis_results_cache = {}
last_analysis_time = None

# FastAPI app
app = FastAPI(
    title="Crypto Monitor",
    description="Smart wallet analysis and monitoring system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_compatibility_headers(request: Request, call_next):
    response = await call_next(request)
    # Add headers for better HTTP/2 compatibility
    response.headers["Connection"] = "keep-alive"
    response.headers["Keep-Alive"] = "timeout=60, max=1000" 
    # Remove caching headers that can cause issues with HTTP/2
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    # Add HTTP/2 specific headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database connection
mongo_client = None
db = None

class AnalysisRequest(BaseModel):
    network: str = "ethereum"
    analysis_type: str = "buy" 
    num_wallets: int = 50
    days_back: float = 1.0

class AlertSettings(BaseModel):
    min_wallets: int = 1
    min_eth: float = 0.01
    min_score: float = 10.0
    networks: List[str] = ["ethereum", "base"]

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and services"""
    global mongo_client, db
    
    try:
        # MongoDB connection
        mongo_uri = os.getenv('MONGO_URI')
        if mongo_uri:
            mongo_client = AsyncIOMotorClient(mongo_uri)
            db_name = os.getenv('DB_NAME', 'crypto_tracker')
            db = mongo_client[db_name]
            logger.info("Database connected successfully")
        
        # Test Telegram configuration
        if check_notification_config():
            logger.info("Telegram notifications configured")
        else:
            logger.warning("Telegram not configured - notifications disabled")
            
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
            
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections"""
    global mongo_client, monitoring_active
    
    monitoring_active = False
    
    if mongo_client:
        mongo_client.close()
        logger.info("Database connection closed")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard"""
    base_url = str(request.base_url).rstrip('/')
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "base_url": base_url
    })

@app.get("/control-panel", response_class=HTMLResponse) 
async def control_panel(request: Request):
    """Control panel interface"""
    base_url = str(request.base_url).rstrip('/')
    return templates.TemplateResponse("control_panel.html", {
        "request": request,
        "base_url": base_url
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "monitoring_active": monitoring_active,
        "telegram_configured": check_notification_config(),
        "database_connected": db is not None,
        "environment": os.getenv('ENVIRONMENT', 'development')
    }
    return JSONResponse(content=status)

@app.get("/telegram/status")
async def telegram_status():
    """Check Telegram configuration status"""
    configured = check_notification_config()
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    return JSONResponse(content={
        "configured": configured,
        "bot_token_length": len(bot_token) if bot_token else 0,
        "chat_id_set": bool(chat_id),
        "status": "ready" if configured else "not_configured"
    })

@app.post("/telegram/test")
async def test_telegram():
    """Send test Telegram notification"""
    try:
        success = await send_test_notification()
        return JSONResponse(content={
            "success": success,
            "message": "Test notification sent" if success else "Failed to send notification"
        })
    except Exception as e:
        logger.error(f"Telegram test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/run")
async def run_analysis(request: AnalysisRequest):
    """Run crypto analysis"""
    global analysis_results_cache, last_analysis_time
    
    try:
        logger.info(f"Running {request.analysis_type} analysis for {request.network}")
        
        if request.analysis_type == "buy":
            async with BuyAnalyzer(request.network) as analyzer:
                result = await analyzer.analyze_wallets_concurrent(
                    num_wallets=request.num_wallets,
                    days_back=request.days_back
                )
        elif request.analysis_type == "sell":
            async with SellAnalyzer(request.network) as analyzer:
                result = await analyzer.analyze_wallets_concurrent(
                    num_wallets=request.num_wallets, 
                    days_back=request.days_back
                )
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        # Cache results
        cache_key = f"{request.network}_{request.analysis_type}"
        analysis_results_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow(),
            "request": request.dict()
        }
        last_analysis_time = datetime.utcnow()
        
        return JSONResponse(content={
            "success": True,
            "network": result.network,
            "analysis_type": result.analysis_type,
            "total_transactions": result.total_transactions,
            "unique_tokens": result.unique_tokens,
            "total_eth_value": result.total_eth_value,
            "top_tokens": result.ranked_tokens[:10],
            "performance_metrics": result.performance_metrics,
            "timestamp": last_analysis_time.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/results")
async def get_analysis_results():
    """Get cached analysis results"""
    if not analysis_results_cache:
        return JSONResponse(content={
            "message": "No analysis results available. Run analysis first.",
            "results": {},
            "last_run": None
        })
    
    formatted_results = {}
    for key, data in analysis_results_cache.items():
        result = data["result"]
        formatted_results[key] = {
            "network": result.network,
            "analysis_type": result.analysis_type,
            "total_transactions": result.total_transactions,
            "unique_tokens": result.unique_tokens, 
            "total_eth_value": result.total_eth_value,
            "top_tokens": result.ranked_tokens[:10],
            "timestamp": data["timestamp"].isoformat(),
            "web3_enhanced": result.web3_enhanced
        }
    
    return JSONResponse(content={
        "results": formatted_results,
        "last_run": last_analysis_time.isoformat() if last_analysis_time else None,
        "cache_size": len(analysis_results_cache)
    })

@app.post("/monitor/start")
async def start_monitoring():
    """Start the monitoring service"""
    global monitoring_active
    
    if monitoring_active:
        return JSONResponse(content={"message": "Monitoring already active", "status": "running"})
    
    if not check_notification_config():
        raise HTTPException(status_code=400, detail="Telegram not configured")
    
    monitoring_active = True
    logger.info("Monitoring service started")
    
    return JSONResponse(content={
        "message": "Monitoring started successfully",
        "status": "running",
        "telegram_ready": True
    })

@app.post("/monitor/stop")
async def stop_monitoring():
    """Stop the monitoring service"""
    global monitoring_active
    
    monitoring_active = False
    logger.info("Monitoring service stopped")
    
    return JSONResponse(content={
        "message": "Monitoring stopped",
        "status": "stopped"
    })

@app.get("/monitor/status")
async def monitor_status():
    """Get monitoring status"""
    return JSONResponse(content={
        "monitoring_active": monitoring_active,
        "telegram_configured": check_notification_config(),
        "last_analysis": last_analysis_time.isoformat() if last_analysis_time else None,
        "cached_results": len(analysis_results_cache),
        "uptime": datetime.utcnow().isoformat()
    })

@app.post("/alerts/simulate")
async def simulate_alert():
    """Simulate alert for testing"""
    if not check_notification_config():
        raise HTTPException(status_code=400, detail="Telegram not configured")
    
    # Create sample alert
    sample_alert = {
        "token": "TEST_TOKEN",
        "network": "ethereum",
        "alert_type": "new_token",
        "confidence": "high",
        "data": {
            "alpha_score": 85.5,
            "total_eth_spent": 12.5,
            "wallet_count": 8,
            "platforms": ["Uniswap", "1inch"]
        }
    }
    
    try:
        await send_alert_notifications([sample_alert])
        return JSONResponse(content={
            "success": True,
            "message": "Sample alert sent to Telegram"
        })
    except Exception as e:
        logger.error(f"Alert simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduler/script")
async def get_scheduler_script(request: Request):
    """Generate Cloud Scheduler setup script"""
    service_url = str(request.base_url).rstrip('/')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
    
    script_content = f"""
# Cloud Scheduler setup script
# Run this to create scheduled analysis jobs

PROJECT_ID="{project_id}"
SERVICE_URL="{service_url}"
REGION="asia-southeast1"  # Change as needed

# Create buy analysis job (every 30 minutes)
gcloud scheduler jobs create http crypto-buy-analysis \\
    --location=$REGION \\
    --schedule="*/30 * * * *" \\
    --uri="$SERVICE_URL/analysis/run" \\
    --http-method=POST \\
    --headers="Content-Type=application/json" \\
    --message-body='{{"network": "ethereum", "analysis_type": "buy", "num_wallets": 50, "days_back": 1.0}}'

# Create sell analysis job (every 45 minutes) 
gcloud scheduler jobs create http crypto-sell-analysis \\
    --location=$REGION \\
    --schedule="*/45 * * * *" \\
    --uri="$SERVICE_URL/analysis/run" \\
    --http-method=POST \\
    --headers="Content-Type=application/json" \\
    --message-body='{{"network": "ethereum", "analysis_type": "sell", "num_wallets": 50, "days_back": 1.0}}'

echo "Scheduler jobs created successfully!"
echo "View jobs: gcloud scheduler jobs list --location=$REGION"
    """
    

@app.post("/admin/clear-cache")
async def clear_cache():
    """Clear analysis cache"""
    global analysis_results_cache, last_analysis_time
    
    analysis_results_cache.clear()
    last_analysis_time = None
    
    return JSONResponse(content={
        "success": True,
        "message": "Cache cleared successfully"
    })

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Crypto Monitor on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False
    )