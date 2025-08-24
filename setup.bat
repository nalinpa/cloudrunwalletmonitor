Write-Host "`n⚙️ Creating environment configuration..." -ForegroundColor Cyan

# Get additional required information
$MONGO_URI = Read-Host "Enter your MongoDB URI (e.g., mongodb+srv://user:pass@cluster.mongodb.net/db)"
$ALCHEMY_KEY = Read-Host "Enter your Alchemy API Key"

# Create .env.production file
$envContent = @"
# REQUIRED - Database
MONGO_URI=$MONGO_URI

# REQUIRED - Alchemy API
ALCHEMY_API_KEY=$ALCHEMY_KEY

# REQUIRED - Telegram (MANDATORY)
TELEGRAM_BOT_TOKEN=$BOT_TOKEN
TELEGRAM_CHAT_ID=$CHAT_ID

# Optional - App Settings
DB_NAME=crypto_tracker
WALLETS_COLLECTION=smart_wallets
ENVIRONMENT=production
LOG_LEVEL=INFO

# Analysis Settings
DEFAULT_WALLET_COUNT=50
MIN_ETH_VALUE=0.01
MIN_ETH_VALUE_BASE=0.005

# Monitor Settings
MONITOR_INTERVAL_MINUTES=30
ALERT_MIN_WALLETS=1
ALERT_MIN_ETH=0.01
ALERT_MIN_SCORE=10.0

# Security
REQUIRE_AUTH=false
APP_PASSWORD=admin123
SECRET_KEY=your-secret-key-change-this-$(Get-Random)
"@

$envContent | Out-File -FilePath ".env.production" -Encoding UTF8

# Create template file
$templateContent = @"
# Copy this to .env.production and fill in your values

# REQUIRED - MongoDB connection
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database

# REQUIRED - Alchemy API key  
ALCHEMY_API_KEY=your_alchemy_api_key_here

# REQUIRED - Telegram bot credentials (MANDATORY)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Optional settings with defaults
ENVIRONMENT=production
LOG_LEVEL=INFO
DEFAULT_WALLET_COUNT=50
MONITOR_INTERVAL_MINUTES=30
"@

$templateContent | Out-File -FilePath ".env.template" -Encoding UTF8

Write-Host "✅ Environment files created" -ForegroundColor Green

# =============================================================================
# Step 5: Create Core Files (Windows)
# =============================================================================

Write-Host "`n📁 Creating core application files..." -ForegroundColor Cyan

# Create requirements.txt
$requirements = @"
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
httpx==0.25.2
motor==3.3.2
pymongo==4.6.0
pandas==2.1.4
numpy==1.25.2
scipy==1.11.4
python-dotenv==1.0.0
aiofiles==23.2.0
orjson==3.9.10
jinja2==3.1.2
python-multipart==0.0.6
google-cloud-scheduler==2.13.3
google-cloud-pubsub==2.18.4
"@

$requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8

# Create Dockerfile
$dockerfile = @"
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs static/css static/js static/icons templates

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "main.py"]
"@

$dockerfile | Out-File -FilePath "Dockerfile" -Encoding UTF8

# Create .dockerignore
$dockerignore = @"
.git
.gitignore
README.md
.env.local
.env.template
*.log
.pytest_cache
__pycache__
*.pyc
.vscode
.idea
node_modules
"@

$dockerignore | Out-File -FilePath ".dockerignore" -Encoding UTF8

# Create .gcloudignore
$dockerignore | Out-File -FilePath ".gcloudignore" -Encoding UTF8

Write-Host "✅ Docker files created" -ForegroundColor Green

# =============================================================================
# Step 6: Create Notifications Service (Windows)
# =============================================================================

$notificationService = @'
import httpx
import asyncio
import logging
from typing import Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramClient:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        if self.bot_token and self.chat_id:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def send_message(self, message: str) -> bool:
        if not self._client or not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = await self._client.post(url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

telegram_client = TelegramClient()

async def send_alert_notifications(alerts: list):
    """Send alert notifications via Telegram"""
    if not alerts:
        return
    
    async with telegram_client:
        for alert in alerts:
            message = format_alert_message(alert)
            await telegram_client.send_message(message)
            await asyncio.sleep(1)  # Rate limiting

async def send_test_notification():
    """Send test notification"""
    test_message = f"🧪 **TEST NOTIFICATION**\n\n✅ Crypto Monitor is working!\n🕐 {datetime.now().strftime('%H:%M:%S')}"
    
    async with telegram_client:
        return await telegram_client.send_message(test_message)

def check_notification_config():
    """Check if Telegram is properly configured"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return bool(bot_token and chat_id and len(bot_token) > 40)

def format_alert_message(alert: dict) -> str:
    """Format alert for Telegram"""
    data = alert.get('data', {})
    alert_type = alert.get('alert_type', 'unknown')
    
    if alert_type == 'new_token':
        emoji = "🆕"
        score = data.get('alpha_score', 0)
    else:
        emoji = "📉"
        score = data.get('sell_score', 0)
    
    eth_value = data.get('total_eth_spent') or data.get('total_eth_value', 0)
    
    message = f"""
{emoji} **{alert_type.replace('_', ' ').upper()}**

🪙 **Token:** `{alert['token']}`
🌐 **Network:** {alert['network'].upper()}
📊 **Score:** {score:.1f}
💰 **ETH:** {eth_value:.4f}
👥 **Wallets:** {data.get('wallet_count', 0)}
🎯 **Confidence:** {alert['confidence']}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
    
    return message.strip()
'@

$notificationService | Out-File -FilePath "services\notifications.py" -Encoding UTF8

Write-Host "✅ Telegram notifications service created" -ForegroundColor Green

# =============================================================================
# Step 7: Create Placeholder Analysis Services (Windows)
# =============================================================================

# Create models
$models = @'
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class Purchase:
    transaction_hash: str
    token_bought: str
    amount_received: float
    eth_spent: float
    wallet_address: str
    platform: str
    block_number: int
    timestamp: datetime
    sophistication_score: Optional[float] = None
    web3_analysis: Optional[Dict] = None

@dataclass
class AnalysisResult:
    network: str
    analysis_type: str
    total_transactions: int
    unique_tokens: int
    total_eth_value: float
    ranked_tokens: List[tuple]
    performance_metrics: Dict[str, Any]
    web3_enhanced: bool = False
'@

$models | Out-File -FilePath "core\data\models.py" -Encoding UTF8

# Create buy analyzer placeholder
$buyAnalyzer = @'
import logging
from typing import List, Dict
from datetime import datetime
from core.data.models import AnalysisResult

logger = logging.getLogger(__name__)

class BuyAnalyzer:
    def __init__(self, network: str):
        self.network = network
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def analyze_wallets_concurrent(self, num_wallets: int = 50, days_back: float = 1.0) -> AnalysisResult:
        """Placeholder buy analysis - replace with your full implementation"""
        logger.info(f"Running placeholder buy analysis for {self.network}")
        
        # Placeholder data - replace with real analysis
        ranked_tokens = [
            ("PLACEHOLDER_TOKEN", {"total_eth_spent": 0.1, "wallet_count": 2, "platforms": ["Uniswap"]}, 25.0)
        ]
        
        return AnalysisResult(
            network=self.network,
            analysis_type="buy",
            total_transactions=1,
            unique_tokens=1,
            total_eth_value=0.1,
            ranked_tokens=ranked_tokens,
            performance_metrics={"placeholder": True},
            web3_enhanced=False
        )
'@

$buyAnalyzer | Out-File -FilePath "core\analysis\buy_analyzer.py" -Encoding UTF8

# Create sell analyzer placeholder
$sellAnalyzer = @'
import logging
from typing import List, Dict
from datetime import datetime
from core.data.models import AnalysisResult

logger = logging.getLogger(__name__)

class SellAnalyzer:
    def __init__(self, network: str):
        self.network = network
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def analyze_wallets_concurrent(self, num_wallets: int = 50, days_back: float = 1.0) -> AnalysisResult:
        """Placeholder sell analysis - replace with your full implementation"""
        logger.info(f"Running placeholder sell analysis for {self.network}")
        
        # Placeholder data - replace with real analysis
        ranked_tokens = [
            ("PLACEHOLDER_TOKEN", {"total_eth_value": 0.05, "wallet_count": 1, "methods": ["Transfer"]}, 15.0)
        ]
        
        return AnalysisResult(
            network=self.network,
            analysis_type="sell", 
            total_transactions=1,
            unique_tokens=1,
            total_eth_value=0.05,
            ranked_tokens=ranked_tokens,
            performance_metrics={"placeholder": True},
            web3_enhanced=False
        )
'@

$sellAnalyzer | Out-File -FilePath "core\analysis\sell_analyzer.py" -Encoding UTF8

Write-Host "✅ Placeholder analysis services created" -ForegroundColor Green
Write-Host "⚠️ Replace these with your full analysis implementations" -ForegroundColor Yellow

# =============================================================================
# Step 8: Create Deploy Script (Windows PowerShell)
# =============================================================================

$deployScript = @'
# PowerShell deployment script for Windows
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "asia-southeast"
)

$SERVICE_NAME = "crypto-monitor"

Write-Host "🚀 Deploying Crypto Monitor with Mandatory Telegram" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Service: $SERVICE_NAME" -ForegroundColor White

# Validate environment file exists
if (!(Test-Path ".env.production")) {
    Write-Host "❌ ERROR: .env.production file not found" -ForegroundColor Red
    Write-Host "Please create .env.production with your configuration" -ForegroundColor Yellow
    exit 1
}

# Read environment variables
$envVars = Get-Content ".env.production" | Where-Object { $_ -match "^[^#]" -and $_ -match "=" }
$envString = ($envVars -join ",")

# Validate Telegram configuration
$telegramBotToken = ($envVars | Where-Object { $_ -match "TELEGRAM_BOT_TOKEN=" }) -replace "TELEGRAM_BOT_TOKEN=", ""
$telegramChatId = ($envVars | Where-Object { $_ -match "TELEGRAM_CHAT_ID=" }) -replace "TELEGRAM_CHAT_ID=", ""

if ([string]::IsNullOrEmpty($telegramBotToken) -or [string]::IsNullOrEmpty($telegramChatId)) {
    Write-Host "❌ ERROR: Telegram configuration missing in .env.production" -ForegroundColor Red
    Write-Host "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Telegram configuration found" -ForegroundColor Green

# Set project and enable APIs
gcloud config set project $ProjectId
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Build and deploy
Write-Host "🔨 Building and deploying..." -ForegroundColor Cyan

gcloud run deploy $SERVICE_NAME `
    --source . `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8080 `
    --memory 2Gi `
    --cpu 2 `
    --timeout 900 `
    --max-instances 10 `
    --set-env-vars $envString

# Get service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $Region --format "value(status.url)"

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "🌐 Service URL: $SERVICE_URL" -ForegroundColor White
Write-Host "🔧 Control Panel: $SERVICE_URL/control-panel" -ForegroundColor White
Write-Host "📱 Telegram Status: $SERVICE_URL/telegram/status" -ForegroundColor White
Write-Host "🏥 Health Check: $SERVICE_URL/health" -ForegroundColor White

# Test the deployment
Write-Host "🧪 Testing deployment..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

try {
    $healthCheck = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing
    if ($healthCheck.StatusCode -eq 200) {
        Write-Host "✅ Service is healthy!" -ForegroundColor Green
        
        # Test Telegram
        try {
            $telegramStatus = Invoke-RestMethod -Uri "$SERVICE_URL/telegram/status"
            if ($telegramStatus.configured) {
                Write-Host "✅ Telegram notifications ready!" -ForegroundColor Green
            } else {
                Write-Host "⚠️ WARNING: Telegram not properly configured" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️ Could not check Telegram status" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Health check failed" -ForegroundColor Red
    Write-Host "🔍 Check logs: gcloud logs read --service=$SERVICE_NAME --limit=20" -ForegroundColor Yellow
}

Write-Host "`n📝 Next steps:" -ForegroundColor Yellow
Write-Host "1. Open control panel: $SERVICE_URL/control-panel" -ForegroundColor White
Write-Host "2. Test Telegram: Invoke-RestMethod -Uri '$SERVICE_URL/telegram/test' -Method Post" -ForegroundColor White
Write-Host "3. Start monitoring: Invoke-RestMethod -Uri '$SERVICE_URL/monitor/start' -Method Post" -ForegroundColor White
Write-Host "4. Setup scheduler: Use the control panel to generate scheduler script" -ForegroundColor White

Write-Host "`n🔧 Management commands:" -ForegroundColor Yellow
Write-Host "  View logs: gcloud logs tail --service=$SERVICE_NAME" -ForegroundColor White
Write-Host "  Update service: gcloud run services update $SERVICE_NAME --region=$Region" -ForegroundColor White
Write-Host "  Delete service: gcloud run services delete $SERVICE_NAME --region=$Region" -ForegroundColor White
'@

$deployScript | Out-File -FilePath "deploy.ps1" -Encoding UTF8

Write-Host "✅ PowerShell deployment script created (deploy.ps1)" -ForegroundColor Green

# =============================================================================
# Step 9: Create Windows Test Script
# =============================================================================

$testScript = @'
# Windows PowerShell test script

Write-Host "🧪 Testing Crypto Monitor Setup" -ForegroundColor Cyan

# Test 1: Environment validation
Write-Host "1️⃣ Testing environment configuration..." -ForegroundColor White
if (Test-Path ".env.production") {
    $envContent = Get-Content ".env.production"
    
    $telegramBot = $envContent | Where-Object { $_ -match "TELEGRAM_BOT_TOKEN=" }
    $telegramChat = $envContent | Where-Object { $_ -match "TELEGRAM_CHAT_ID=" }
    
    if ($telegramBot -and $telegramChat) {
        Write-Host "   ✅ Telegram configuration found" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Telegram configuration missing" -ForegroundColor Red
        exit 1
    }
    
    $mongoUri = $envContent | Where-Object { $_ -match "MONGO_URI=" }
    $alchemyKey = $envContent | Where-Object { $_ -match "ALCHEMY_API_KEY=" }
    
    if ($mongoUri -and $alchemyKey) {
        Write-Host "   ✅ Database and API configuration found" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ Database or API configuration missing" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ❌ .env.production file missing" -ForegroundColor Red
    exit 1
}

# Test 2: File structure
Write-Host "2️⃣ Testing file structure..." -ForegroundColor White
$requiredFiles = @(
    "main.py"
    "requirements.txt" 
    "Dockerfile"
    "deploy.ps1"
    ".env.production"
    "services\notifications.py"
    "core\data\models.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file missing" -ForegroundColor Red
    }
}

Write-Host "`n✅ Setup test complete!" -ForegroundColor Green
Write-Host "`n🚀 Ready to deploy:" -ForegroundColor Cyan
Write-Host "   .\deploy.ps1 -ProjectId YOUR_PROJECT_ID" -ForegroundColor White
Write-Host "`n📱 Your Telegram bot is configured and ready" -ForegroundColor Green
Write-Host "🌐 All required files are present" -ForegroundColor Green
'@

$testScript | Out-File -FilePath "test-setup.ps1" -Encoding UTF8

Write-Host "✅ Windows test script created (test-setup.ps1)" -ForegroundColor Green

# =============================================================================
# Final Steps
# =============================================================================

Write-Host "`n🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "`nProject created in: $(Get-Location)" -ForegroundColor White
Write-Host "Project ID: $PROJECT_ID" -ForegroundColor White

Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Copy the main.py file from the artifact above" -ForegroundColor White
Write-Host "2. Test your setup: .\test-setup.ps1" -ForegroundColor White  
Write-Host "3. Deploy to Cloud Run: .\deploy.ps1 -ProjectId $PROJECT_ID" -ForegroundColor White
Write-Host "4. Open control panel and setup scheduler" -ForegroundColor White

Write-Host "`n💾 Important files created:" -ForegroundColor Cyan
Write-Host "  📱 .env.production - Your configuration (KEEP SECURE!)" -ForegroundColor White
Write-Host "  🚀 deploy.ps1 - Deployment script" -ForegroundColor White
Write-Host "  🧪 test-setup.ps1 - Test script" -ForegroundColor White
Write-Host "  📋 requirements.txt - Python dependencies" -ForegroundColor White
Write-Host "  🐳 Dockerfile - Container configuration" -ForegroundColor White

Write-Host "`n⚠️ IMPORTANT:" -ForegroundColor Red
Write-Host "You still need to add the main.py file from the artifact above!" -ForegroundColor Yellow
Write-Host "Copy the entire 'Enhanced Cloud Run Service (main.py)' content to main.py" -ForegroundColor Yellow

Write-Host "`nTelegram Bot Token: $BOT_TOKEN" -ForegroundColor Green
Write-Host "Telegram Chat ID: $CHAT_ID" -ForegroundColor Green
Write-Host "Project ID: $PROJECT_ID" -ForegroundColor Green