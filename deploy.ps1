param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "asia-southeast1",
    [string]$FunctionName = "crypto-analysis-function"
)

Write-Host "🚀 Enhanced Crypto Monitor Cloud Functions Deployment with Telegram" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Function: $FunctionName" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan

# Function to check if a command exists
function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Validate prerequisites
Write-Host "`n🔍 Checking prerequisites..." -ForegroundColor Yellow

if (!(Test-CommandExists "gcloud")) {
    Write-Host "❌ ERROR: gcloud CLI not found. Please install Google Cloud SDK first." -ForegroundColor Red
    Write-Host "Download from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ gcloud CLI found" -ForegroundColor Green

# Check required files
$requiredFiles = @("main.py", "requirements.txt", ".env.production")
foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Host "❌ ERROR: Required file missing: $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Found $file" -ForegroundColor Green
}

# Validate main.py is configured for Cloud Functions
Write-Host "`n🔍 Validating Cloud Functions configuration..." -ForegroundColor Yellow
$mainContent = Get-Content "main.py" -Raw
if ($mainContent -like "*functions_framework*" -and $mainContent -like "*@functions_framework.http*") {
    Write-Host "✅ main.py is configured for Cloud Functions" -ForegroundColor Green
} else {
    Write-Host "❌ ERROR: main.py is not configured for Cloud Functions" -ForegroundColor Red
    exit 1
}

# Enhanced environment variable validation with Telegram focus
Write-Host "`n📱 Processing environment variables with Telegram validation..." -ForegroundColor Yellow

if (!(Test-Path ".env.production")) {
    Write-Host "❌ ERROR: .env.production file not found" -ForegroundColor Red
    exit 1
}

$envContent = Get-Content ".env.production" -Encoding UTF8
$validEnvVars = @()
$requiredVars = @("MONGO_URI", "ALCHEMY_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
$foundVars = @{}

# Parse environment variables
foreach ($line in $envContent) {
    if ($line -and $line -notmatch "^\s*#" -and $line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        
        # Remove surrounding quotes
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or 
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        
        $validEnvVars += "$key=$value"
        $foundVars[$key] = $value
        Write-Host "✓ $key" -ForegroundColor Gray
    }
}

Write-Host "✅ Found $($validEnvVars.Count) environment variables" -ForegroundColor Green

# Validate required variables with special focus on Telegram
$missingVars = @()
$telegramVars = @("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

foreach ($reqVar in $requiredVars) {
    if (!$foundVars.ContainsKey($reqVar) -or [string]::IsNullOrEmpty($foundVars[$reqVar])) {
        $missingVars += $reqVar
    } else {
        if ($reqVar -in $telegramVars) {
            Write-Host "✅ Telegram variable $reqVar is set" -ForegroundColor Green
        } else {
            Write-Host "✅ Required variable $reqVar is set" -ForegroundColor Green
        }
    }
}

# Special Telegram validation
if ($foundVars.ContainsKey("TELEGRAM_BOT_TOKEN")) {
    $botToken = $foundVars["TELEGRAM_BOT_TOKEN"]
    if ($botToken.Length -lt 40 -or !$botToken.Contains(":")) {
        Write-Host "⚠️ WARNING: TELEGRAM_BOT_TOKEN may be invalid (should be like 123456789:ABCdef...)" -ForegroundColor Yellow
    } else {
        Write-Host "✅ TELEGRAM_BOT_TOKEN format looks valid" -ForegroundColor Green
    }
}

if ($foundVars.ContainsKey("TELEGRAM_CHAT_ID")) {
    $chatId = $foundVars["TELEGRAM_CHAT_ID"]
    if ($chatId -match "^\-?\d+$") {
        Write-Host "✅ TELEGRAM_CHAT_ID format looks valid" -ForegroundColor Green
    } else {
        Write-Host "⚠️ WARNING: TELEGRAM_CHAT_ID should be a number (like -123456789 or 987654321)" -ForegroundColor Yellow
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "❌ ERROR: Missing required environment variables:" -ForegroundColor Red
    foreach ($missingVar in $missingVars) {
        Write-Host "  - $missingVar" -ForegroundColor Red
        
        if ($missingVar -eq "TELEGRAM_BOT_TOKEN") {
            Write-Host "    Get this from @BotFather on Telegram" -ForegroundColor Gray
        } elseif ($missingVar -eq "TELEGRAM_CHAT_ID") {
            Write-Host "    Get this by messaging @userinfobot on Telegram" -ForegroundColor Gray
        }
    }
    exit 1
}

# Set up Google Cloud project
Write-Host "`n☁️ Setting up Google Cloud..." -ForegroundColor Yellow

try {
    Write-Host "Setting project to $ProjectId..." -ForegroundColor Gray
    gcloud config set project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Failed to set project" }
    
    Write-Host "Enabling required APIs..." -ForegroundColor Gray
    gcloud services enable cloudfunctions.googleapis.com --quiet
    gcloud services enable cloudbuild.googleapis.com --quiet
    gcloud services enable artifactregistry.googleapis.com --quiet
    
    Write-Host "✅ Google Cloud setup complete" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to setup Google Cloud project" -ForegroundColor Red
    exit 1
}

# Deploy main function
Write-Host "`n🚀 Deploying main analysis function..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

$deploymentSuccess = $false
$envVarsString = $validEnvVars -join ","

try {
    gcloud functions deploy $FunctionName `
        --gen2 `
        --runtime=python311 `
        --region=$Region `
        --source=. `
        --entry-point=crypto_analysis_function `
        --trigger-http `
        --allow-unauthenticated `
        --memory=2GB `
        --timeout=540s `
        --max-instances=10 `
        --min-instances=0 `
        --concurrency=1 `
        --set-env-vars=$envVarsString `
        --quiet

    if ($LASTEXITCODE -eq 0) {
        $deploymentSuccess = $true
        Write-Host "✅ Main function deployment successful!" -ForegroundColor Green
    } else {
        throw "Main function deployment failed"
    }
    
} catch {
    Write-Host "❌ ERROR: Main function deployment failed" -ForegroundColor Red
    exit 1
}

# Deploy Telegram-specific functions
Write-Host "`n📱 Deploying Telegram support functions..." -ForegroundColor Yellow

$telegramFunctions = @(
    @{Name="telegram-status"; Entry="telegram_status"; Description="Telegram status check"},
    @{Name="telegram-test"; Entry="telegram_test"; Description="Telegram test notifications"},
    @{Name="telegram-thresholds"; Entry="telegram_thresholds"; Description="Alert thresholds management"},
    @{Name="telegram-simulate"; Entry="telegram_simulate_alert"; Description="Simulate alerts"}
)

$telegramDeployedFunctions = @()

foreach ($func in $telegramFunctions) {
    try {
        Write-Host "Deploying $($func.Name)..." -ForegroundColor Gray
        
        gcloud functions deploy $func.Name `
            --gen2 `
            --runtime=python311 `
            --region=$Region `
            --source=. `
            --entry-point=$($func.Entry) `
            --trigger-http `
            --allow-unauthenticated `
            --memory=512MB `
            --timeout=120s `
            --set-env-vars=$envVarsString `
            --quiet

        if ($LASTEXITCODE -eq 0) {
            $telegramDeployedFunctions += $func
            Write-Host "✅ $($func.Name) deployed successfully" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $($func.Name) deployment failed (non-critical)" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "⚠️ Failed to deploy $($func.Name): $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host "✅ Deployed $($telegramDeployedFunctions.Count) of $($telegramFunctions.Count) Telegram functions" -ForegroundColor Green

# Get function URLs
Write-Host "`n📡 Getting function URLs..." -ForegroundColor Yellow

try {
    $MAIN_FUNCTION_URL = gcloud functions describe $FunctionName --region=$Region --format="value(serviceConfig.uri)"
    
    if ($MAIN_FUNCTION_URL) {
        Write-Host "`n🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
        Write-Host "=============================================" -ForegroundColor Green
        Write-Host "🌐 Main Function URL: $MAIN_FUNCTION_URL" -ForegroundColor White
        Write-Host "🏥 Health Check: $MAIN_FUNCTION_URL" -ForegroundColor White
        Write-Host "🔍 Debug Info: ${MAIN_FUNCTION_URL}?debug=true" -ForegroundColor White
        Write-Host "📊 Analysis Endpoint: $MAIN_FUNCTION_URL (POST)" -ForegroundColor White
        Write-Host "=============================================" -ForegroundColor Green
        
        # Show Telegram function URLs
        if ($telegramDeployedFunctions.Count -gt 0) {
            Write-Host "`n📱 TELEGRAM FUNCTION URLs:" -ForegroundColor Cyan
            Write-Host "=============================================" -ForegroundColor Cyan
            
            foreach ($func in $telegramDeployedFunctions) {
                try {
                    $funcUrl = gcloud functions describe $func.Name --region=$Region --format="value(serviceConfig.uri)" 2>$null
                    if ($funcUrl) {
                        Write-Host "🔗 $($func.Description): $funcUrl" -ForegroundColor White
                    }
                } catch {
                    Write-Host "⚠️ Could not get URL for $($func.Name)" -ForegroundColor Yellow
                }
            }
            Write-Host "=============================================" -ForegroundColor Cyan
        }
        
    } else {
        Write-Host "⚠️ Deployment may have succeeded but couldn't retrieve function URL" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "⚠️ Could not retrieve function information: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test the deployment with enhanced Telegram testing
if ($MAIN_FUNCTION_URL) {
    Write-Host "`n🧪 Testing deployment..." -ForegroundColor Yellow
    Write-Host "Waiting for functions to be ready..." -ForegroundColor Gray
    Start-Sleep -Seconds 20
    
    # Test main function health
    try {
        Write-Host "Testing main function..." -ForegroundColor Gray
        $healthResponse = Invoke-WebRequest -Uri $MAIN_FUNCTION_URL -UseBasicParsing -TimeoutSec 30
        
        if ($healthResponse.StatusCode -eq 200) {
            Write-Host "✅ Main function is healthy!" -ForegroundColor Green
            
            try {
                $healthData = $healthResponse.Content | ConvertFrom-Json
                Write-Host "✅ Status: $($healthData.status)" -ForegroundColor Green
                Write-Host "✅ Version: $($healthData.version)" -ForegroundColor Green
                Write-Host "📱 Telegram Configured: $($healthData.telegram_configured)" -ForegroundColor $(if ($healthData.telegram_configured) {'Green'} else {'Yellow'})
            } catch {
                Write-Host "✅ Function responded successfully" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "⚠️ Main function health check failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    # Test Telegram functionality
    Write-Host "`n📱 Testing Telegram integration..." -ForegroundColor Yellow
    
    # Test Telegram status
    $telegramStatusFunc = $telegramDeployedFunctions | Where-Object { $_.Name -eq "telegram-status" }
    if ($telegramStatusFunc) {
        try {
            $telegramStatusUrl = gcloud functions describe "telegram-status" --region=$Region --format="value(serviceConfig.uri)" 2>$null
            if ($telegramStatusUrl) {
                $telegramResponse = Invoke-RestMethod -Uri $telegramStatusUrl -TimeoutSec 30
                
                Write-Host "📊 Telegram Status Check:" -ForegroundColor Cyan
                Write-Host "  Configured: $($telegramResponse.configured)" -ForegroundColor $(if ($telegramResponse.configured) {'Green'} else {'Red'})
                Write-Host "  Bot Token Set: $($telegramResponse.bot_token_set)" -ForegroundColor $(if ($telegramResponse.bot_token_set) {'Green'} else {'Red'})
                Write-Host "  Chat ID Set: $($telegramResponse.chat_id_set)" -ForegroundColor $(if ($telegramResponse.chat_id_set) {'Green'} else {'Red'})
                
                if ($telegramResponse.configured) {
                    Write-Host "🎊 Telegram is fully configured and ready!" -ForegroundColor Green
                } else {
                    Write-Host "⚠️ Telegram configuration needs attention" -ForegroundColor Yellow
                }
            }
        } catch {
            Write-Host "⚠️ Telegram status check failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    # Test Telegram notifications
    $telegramTestFunc = $telegramDeployedFunctions | Where-Object { $_.Name -eq "telegram-test" }
    if ($telegramTestFunc -and $foundVars.ContainsKey("TELEGRAM_BOT_TOKEN")) {
        try {
            $telegramTestUrl = gcloud functions describe "telegram-test" --region=$Region --format="value(serviceConfig.uri)" 2>$null
            if ($telegramTestUrl) {
                Write-Host "Sending Telegram test notification..." -ForegroundColor Gray
                
                $testResponse = Invoke-RestMethod -Uri $telegramTestUrl -Method POST -ContentType "application/json" -Body "{}" -TimeoutSec 30
                
                if ($testResponse.success) {
                    Write-Host "✅ Telegram test notification sent successfully!" -ForegroundColor Green
                    Write-Host "📱 Check your Telegram app for the test message" -ForegroundColor Cyan
                } else {
                    Write-Host "❌ Telegram test notification failed: $($testResponse.error)" -ForegroundColor Red
                }
            }
        } catch {
            Write-Host "⚠️ Telegram test failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Provide comprehensive next steps
Write-Host "`n📋 NEXT STEPS & USAGE:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

if ($MAIN_FUNCTION_URL) {
    Write-Host "1. 🌐 Test main function: $MAIN_FUNCTION_URL" -ForegroundColor White
    Write-Host "2. 🔍 Check configuration: ${MAIN_FUNCTION_URL}?debug=true" -ForegroundColor White
    Write-Host "3. 📱 Test Telegram notifications:" -ForegroundColor White
    
    $telegramTestFunc = $telegramDeployedFunctions | Where-Object { $_.Name -eq "telegram-test" }
    if ($telegramTestFunc) {
        $telegramTestUrl = gcloud functions describe "telegram-test" --region=$Region --format="value(serviceConfig.uri)" 2>$null
        if ($telegramTestUrl) {
            Write-Host "   POST to: $telegramTestUrl" -ForegroundColor Gray
        }
    }
    
    Write-Host "4. 📊 Run sample analysis:" -ForegroundColor White
    Write-Host '   POST to main function with JSON:' -ForegroundColor Gray
    Write-Host '   {"network": "ethereum", "analysis_type": "buy", "notifications": true}' -ForegroundColor Gray
    Write-Host "5. 🎯 Configure alert thresholds using telegram-thresholds endpoint" -ForegroundColor White
    Write-Host "6. ⏰ Set up Cloud Scheduler for automated analysis" -ForegroundColor White
}

Write-Host "`n📱 TELEGRAM NOTIFICATION FEATURES:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "✅ Real-time analysis summaries" -ForegroundColor Green
Write-Host "✅ High-alpha token alerts" -ForegroundColor Green
Write-Host "✅ Sell pressure warnings" -ForegroundColor Green
Write-Host "✅ System error notifications" -ForegroundColor Green
Write-Host "✅ Configurable alert thresholds" -ForegroundColor Green
Write-Host "✅ Test notifications" -ForegroundColor Green
Write-Host "✅ System status updates" -ForegroundColor Green

Write-Host "`n🎯 TELEGRAM ALERT THRESHOLDS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow
if ($foundVars.ContainsKey("TELEGRAM_BOT_TOKEN")) {
    Write-Host "Current thresholds (you can adjust these):" -ForegroundColor White
    Write-Host "• Minimum Alpha Score: 25.0" -ForegroundColor Gray
    Write-Host "• Minimum Sell Score: 20.0" -ForegroundColor Gray
    Write-Host "• Minimum ETH Value: 0.05 ETH" -ForegroundColor Gray
    Write-Host "• Minimum Smart Wallets: 3" -ForegroundColor Gray
    Write-Host "" -ForegroundColor White
    Write-Host "Adjust thresholds via the telegram-thresholds endpoint" -ForegroundColor Cyan
} else {
    Write-Host "Configure Telegram to enable customizable thresholds" -ForegroundColor Yellow
}

Write-Host "`n🔧 MANAGEMENT COMMANDS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "View logs (main function):" -ForegroundColor White
Write-Host "  gcloud functions logs read $FunctionName --region=$Region --limit=20" -ForegroundColor Gray

Write-Host "`nStream logs (real-time):" -ForegroundColor White
Write-Host "  gcloud functions logs tail $FunctionName --region=$Region" -ForegroundColor Gray

Write-Host "`nUpdate functions:" -ForegroundColor White
Write-Host "  gcloud functions deploy $FunctionName --region=$Region --source=." -ForegroundColor Gray

Write-Host "`nDelete all functions:" -ForegroundColor White
Write-Host "  gcloud functions delete $FunctionName --region=$Region" -ForegroundColor Gray
foreach ($func in $telegramDeployedFunctions) {
    Write-Host "  gcloud functions delete $($func.Name) --region=$Region" -ForegroundColor Gray
}

Write-Host "`n🧪 TESTING COMMANDS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

if ($MAIN_FUNCTION_URL) {
    Write-Host "PowerShell testing examples:" -ForegroundColor White
    
    Write-Host "`n# Health check with Telegram status" -ForegroundColor Gray
    Write-Host "Invoke-RestMethod -Uri '$MAIN_FUNCTION_URL'" -ForegroundColor Cyan
    
    Write-Host "`n# Debug mode" -ForegroundColor Gray
    Write-Host "Invoke-RestMethod -Uri '${MAIN_FUNCTION_URL}?debug=true'" -ForegroundColor Cyan
    
    Write-Host "`n# Run analysis with notifications" -ForegroundColor Gray
    Write-Host '$analysisData = @{' -ForegroundColor Cyan
    Write-Host '    network = "ethereum"' -ForegroundColor Cyan
    Write-Host '    analysis_type = "buy"' -ForegroundColor Cyan
    Write-Host '    num_wallets = 50' -ForegroundColor Cyan
    Write-Host '    days_back = 1.0' -ForegroundColor Cyan
    Write-Host '    notifications = $true' -ForegroundColor Cyan
    Write-Host '} | ConvertTo-Json' -ForegroundColor Cyan
    Write-Host '' -ForegroundColor Cyan
    Write-Host "Invoke-RestMethod -Uri '$MAIN_FUNCTION_URL' -Method POST -Body `$analysisData -ContentType 'application/json'" -ForegroundColor Cyan
    
    $telegramTestFunc = $telegramDeployedFunctions | Where-Object { $_.Name -eq "telegram-test" }
    if ($telegramTestFunc) {
        $telegramTestUrl = gcloud functions describe "telegram-test" --region=$Region --format="value(serviceConfig.uri)" 2>$null
        if ($telegramTestUrl) {
            Write-Host "`n# Send Telegram test notification" -ForegroundColor Gray
            Write-Host "Invoke-RestMethod -Uri '$telegramTestUrl' -Method POST -ContentType 'application/json' -Body '{}'" -ForegroundColor Cyan
        }
    }
}

Write-Host "`n🎊 ENHANCED CRYPTO ANALYSIS WITH TELEGRAM DEPLOYMENT COMPLETE!" -ForegroundColor Green
if ($MAIN_FUNCTION_URL) {
    Write-Host "Main Function: $MAIN_FUNCTION_URL" -ForegroundColor Cyan
}
Write-Host "Telegram Functions: $($telegramDeployedFunctions.Count) deployed" -ForegroundColor Cyan
Write-Host "Telegram Configured: $(if ($foundVars.ContainsKey('TELEGRAM_BOT_TOKEN')) {'✅ YES'} else {'❌ NO'})" -ForegroundColor $(if ($foundVars.ContainsKey('TELEGRAM_BOT_TOKEN')) {'Green'} else {'Red'})
Write-Host "=========================================" -ForegroundColor Green

# Final Telegram setup reminder
if (!$foundVars.ContainsKey("TELEGRAM_BOT_TOKEN") -or !$foundVars.ContainsKey("TELEGRAM_CHAT_ID")) {
    Write-Host "`n📱 TELEGRAM SETUP REMINDER:" -ForegroundColor Red
    Write-Host "=============================================" -ForegroundColor Red
    Write-Host "To enable Telegram notifications:" -ForegroundColor Yellow
    Write-Host "1. Create a bot: Message @BotFather on Telegram" -ForegroundColor White
    Write-Host "2. Get your chat ID: Message @userinfobot on Telegram" -ForegroundColor White
    Write-Host "3. Add these to your .env.production file:" -ForegroundColor White
    Write-Host "   TELEGRAM_BOT_TOKEN=your_bot_token_here" -ForegroundColor Gray
    Write-Host "   TELEGRAM_CHAT_ID=your_chat_id_here" -ForegroundColor Gray
    Write-Host "4. Redeploy using this script" -ForegroundColor White
    Write-Host "=============================================" -ForegroundColor Red
} else {
    Write-Host "`n📱 Your crypto analysis system is now ready with Telegram alerts!" -ForegroundColor Green
    Write-Host "Check your Telegram app for notifications! 🚀" -ForegroundColor Cyan
}