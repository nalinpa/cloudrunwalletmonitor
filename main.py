import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Cloud Run imports (NOT Cloud Functions)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Your existing imports
from utils.logger import setup_logging
from utils.config import Config
from core.analysis.buy_analyzer import CloudBuyAnalyzer
from core.analysis.sell_analyzer import CloudSellAnalyzer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Monitor",
    description="Smart wallet analysis and monitoring system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instances (reused across requests)
_analyzers = {}

# Pydantic model for request validation
class AnalysisRequest(BaseModel):
    network: str = "ethereum"
    analysis_type: str = "buy"
    num_wallets: int = 50
    days_back: float = 1.0

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

# Convert your Cloud Function to FastAPI endpoints
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Crypto Monitor is running!",
        "status": "healthy",
        "version": "2.0.0",
        "service": "crypto-analysis"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "analyzers_loaded": len(_analyzers),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "service": "crypto-monitor"
    }

@app.post("/analysis")
async def run_analysis(request: AnalysisRequest):
    """Main analysis endpoint - converted from Cloud Function"""
    try:
        logger.info(f"Running {request.analysis_type} analysis for {request.network}")
        
        # Run the analysis logic from your original function
        result = await _run_analysis(request.dict())
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/")
async def main_endpoint(request: Request):
    """Main endpoint for compatibility - handles POST requests like Cloud Function"""
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return JSONResponse(
                content="",
                status_code=204,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Max-Age': '3600'
                }
            )
        
        # Parse JSON request
        try:
            request_json = await request.json()
        except:
            raise HTTPException(status_code=400, detail="No JSON data provided")
        
        # Run analysis
        result = await _run_analysis(request_json)
        
        return JSONResponse(
            content=result,
            headers={'Access-Control-Allow-Origin': '*'}
        )
        
    except Exception as e:
        logger.error(f"Main endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
