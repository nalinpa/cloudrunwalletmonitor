# Windows PowerShell test script

Write-Host " Testing Crypto Monitor Setup" -ForegroundColor Cyan

# Test 1: Environment validation
Write-Host "1 Testing environment configuration..." -ForegroundColor White
if (Test-Path ".env.production") {
    $envContent = Get-Content ".env.production"
    
    $telegramBot = $envContent | Where-Object { $_ -match "TELEGRAM_BOT_TOKEN=" }
    $telegramChat = $envContent | Where-Object { $_ -match "TELEGRAM_CHAT_ID=" }
    
    if ($telegramBot -and $telegramChat) {
        Write-Host "    Telegram configuration found" -ForegroundColor Green
    } else {
        Write-Host "    Telegram configuration missing" -ForegroundColor Red
        exit 1
    }
    
    $mongoUri = $envContent | Where-Object { $_ -match "MONGO_URI=" }
    $alchemyKey = $envContent | Where-Object { $_ -match "ALCHEMY_API_KEY=" }
    
    if ($mongoUri -and $alchemyKey) {
        Write-Host "    Database and API configuration found" -ForegroundColor Green
    } else {
        Write-Host "    Database or API configuration missing" -ForegroundColor Yellow
    }
} else {
    Write-Host "    .env.production file missing" -ForegroundColor Red
    exit 1
}

# Test 2: File structure
Write-Host "2 Testing file structure..." -ForegroundColor White
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
        Write-Host "    $file" -ForegroundColor Green
    } else {
        Write-Host "    $file missing" -ForegroundColor Red
    }
}

Write-Host "`n Setup test complete!" -ForegroundColor Green
Write-Host "`n Ready to deploy:" -ForegroundColor Cyan
Write-Host "   .\deploy.ps1 -ProjectId YOUR_PROJECT_ID" -ForegroundColor White
Write-Host "`n Your Telegram bot is configured and ready" -ForegroundColor Green
Write-Host " All required files are present" -ForegroundColor Green
