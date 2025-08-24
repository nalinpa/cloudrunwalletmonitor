# PowerShell deployment script for Windows
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "asia-southeast"
)

$SERVICE_NAME = "crypto-monitor"

Write-Host " Deploying Crypto Monitor with Mandatory Telegram" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Service: $SERVICE_NAME" -ForegroundColor White

# Validate environment file exists
if (!(Test-Path ".env.production")) {
    Write-Host " ERROR: .env.production file not found" -ForegroundColor Red
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
    Write-Host " ERROR: Telegram configuration missing in .env.production" -ForegroundColor Red
    Write-Host "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required" -ForegroundColor Yellow
    exit 1
}

Write-Host " Telegram configuration found" -ForegroundColor Green

# Set project and enable APIs
gcloud config set project $ProjectId
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Build and deploy
Write-Host " Building and deploying..." -ForegroundColor Cyan

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

Write-Host " Deployment complete!" -ForegroundColor Green
Write-Host " Service URL: $SERVICE_URL" -ForegroundColor White
Write-Host " Control Panel: $SERVICE_URL/control-panel" -ForegroundColor White
Write-Host " Telegram Status: $SERVICE_URL/telegram/status" -ForegroundColor White
Write-Host " Health Check: $SERVICE_URL/health" -ForegroundColor White

# Test the deployment
Write-Host " Testing deployment..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

try {
    $healthCheck = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing
    if ($healthCheck.StatusCode -eq 200) {
        Write-Host " Service is healthy!" -ForegroundColor Green
        
        # Test Telegram
        try {
            $telegramStatus = Invoke-RestMethod -Uri "$SERVICE_URL/telegram/status"
            if ($telegramStatus.configured) {
                Write-Host " Telegram notifications ready!" -ForegroundColor Green
            } else {
                Write-Host " WARNING: Telegram not properly configured" -ForegroundColor Yellow
            }
        } catch {
            Write-Host " Could not check Telegram status" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host " Health check failed" -ForegroundColor Red
    Write-Host " Check logs: gcloud logs read --service=$SERVICE_NAME --limit=20" -ForegroundColor Yellow
}

Write-Host "`n Next steps:" -ForegroundColor Yellow
Write-Host "1. Open control panel: $SERVICE_URL/control-panel" -ForegroundColor White
Write-Host "2. Test Telegram: Invoke-RestMethod -Uri '$SERVICE_URL/telegram/test' -Method Post" -ForegroundColor White
Write-Host "3. Start monitoring: Invoke-RestMethod -Uri '$SERVICE_URL/monitor/start' -Method Post" -ForegroundColor White
Write-Host "4. Setup scheduler: Use the control panel to generate scheduler script" -ForegroundColor White

Write-Host "`n Management commands:" -ForegroundColor Yellow
Write-Host "  View logs: gcloud logs tail --service=$SERVICE_NAME" -ForegroundColor White
Write-Host "  Update service: gcloud run services update $SERVICE_NAME --region=$Region" -ForegroundColor White
Write-Host "  Delete service: gcloud run services delete $SERVICE_NAME --region=$Region" -ForegroundColor White
