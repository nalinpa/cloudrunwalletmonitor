param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "asia-southeast1",
    [string]$FunctionName = "crypto-analysis-function"
)

Write-Host "🚀 Complete Crypto Monitor Cloud Functions Deployment" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Function: $FunctionName" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan

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

# Check required files (Cloud Functions doesn't need Dockerfile)
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
    Write-Host "   Please ensure main.py uses functions-framework and @functions_framework.http" -ForegroundColor Yellow
    exit 1
}

# Check requirements.txt for Cloud Functions dependencies
$reqContent = Get-Content "requirements.txt" -Raw
if ($reqContent -like "*functions-framework*") {
    Write-Host "✅ requirements.txt includes functions-framework" -ForegroundColor Green
} else {
    Write-Host "⚠️ WARNING: requirements.txt may be missing functions-framework" -ForegroundColor Yellow
}

# Read and validate environment variables
Write-Host "`n🔧 Processing environment variables..." -ForegroundColor Yellow

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
        
        # Remove surrounding quotes if they exist
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

# Validate required variables
$missingVars = @()
foreach ($reqVar in $requiredVars) {
    if (!$foundVars.ContainsKey($reqVar) -or [string]::IsNullOrEmpty($foundVars[$reqVar])) {
        $missingVars += $reqVar
    } else {
        Write-Host "✅ Required variable $reqVar is set" -ForegroundColor Green
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "❌ ERROR: Missing required environment variables:" -ForegroundColor Red
    foreach ($missingVar in $missingVars) {
        Write-Host "  - $missingVar" -ForegroundColor Red
    }
    Write-Host "`nPlease add these to your .env.production file" -ForegroundColor Yellow
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
    Write-Host "Please verify your project ID and permissions" -ForegroundColor Yellow
    exit 1
}

# Deploy to Cloud Functions
Write-Host "`n🚀 Deploying to Cloud Functions..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

$deploymentSuccess = $false
$envVarsString = $validEnvVars -join ","

try {
    # First attempt: Deploy with environment variables
    Write-Host "Attempting deployment with environment variables..." -ForegroundColor Gray
    
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
        Write-Host "✅ Deployment with environment variables successful!" -ForegroundColor Green
    } else {
        throw "Deployment with env vars failed"
    }
    
} catch {
    Write-Host "⚠️ Environment variables deployment failed, trying basic deployment..." -ForegroundColor Yellow
    
    try {
        # Second attempt: Basic deployment without env vars
        Write-Host "Attempting basic deployment (env vars will need manual setup)..." -ForegroundColor Gray
        
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
            --quiet

        if ($LASTEXITCODE -eq 0) {
            $deploymentSuccess = $true
            Write-Host "✅ Basic deployment successful!" -ForegroundColor Green
            Write-Host "⚠️ Environment variables need to be set manually in Google Cloud Console" -ForegroundColor Yellow
        } else {
            throw "All deployment attempts failed"
        }
        
    } catch {
        Write-Host "❌ ERROR: All deployment attempts failed" -ForegroundColor Red
        Write-Host "Please check the logs above for specific error details" -ForegroundColor Yellow
        
        # Show troubleshooting tips
        Write-Host "`n🔧 TROUBLESHOOTING TIPS:" -ForegroundColor Yellow
        Write-Host "1. Check function logs: gcloud functions logs read $FunctionName --region=$Region --limit=20" -ForegroundColor White
        Write-Host "2. Verify entry point exists: 'crypto_analysis_function' in main.py" -ForegroundColor White
        Write-Host "3. Check requirements.txt for missing dependencies" -ForegroundColor White
        Write-Host "4. Ensure all imports in main.py are available" -ForegroundColor White
        exit 1
    }
}

# Get function information
if ($deploymentSuccess) {
    Write-Host "`n📡 Getting function information..." -ForegroundColor Yellow
    
    try {
        $FUNCTION_URL = gcloud functions describe $FunctionName --region=$Region --format="value(serviceConfig.uri)"
        
        if ($FUNCTION_URL) {
            Write-Host "`n🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
            Write-Host "=============================================" -ForegroundColor Green
            Write-Host "🌐 Function URL: $FUNCTION_URL" -ForegroundColor White
            Write-Host "🏥 Health Check: $FUNCTION_URL" -ForegroundColor White
            Write-Host "🔍 Debug Info: ${FUNCTION_URL}?debug=true" -ForegroundColor White
            Write-Host "📊 Analysis Endpoint: $FUNCTION_URL (POST with JSON)" -ForegroundColor White
            Write-Host "=============================================" -ForegroundColor Green
            
            # Test the deployment
            Write-Host "`n🧪 Testing deployment..." -ForegroundColor Yellow
            Write-Host "Waiting for function to be ready..." -ForegroundColor Gray
            Start-Sleep -Seconds 15
            
            try {
                $healthResponse = Invoke-WebRequest -Uri $FUNCTION_URL -UseBasicParsing -TimeoutSec 30
                if ($healthResponse.StatusCode -eq 200) {
                    Write-Host "✅ Function is healthy and responding!" -ForegroundColor Green
                    
                    # Parse health response to check components
                    try {
                        $healthData = $healthResponse.Content | ConvertFrom-Json
                        
                        Write-Host "✅ Status: $($healthData.status)" -ForegroundColor Green
                        Write-Host "✅ Version: $($healthData.version)" -ForegroundColor Green
                        Write-Host "✅ Service: $($healthData.service)" -ForegroundColor Green
                        
                    } catch {
                        Write-Host "✅ Function responded successfully" -ForegroundColor Green
                    }
                    
                } else {
                    Write-Host "⚠️ Function deployed but health check returned HTTP $($healthResponse.StatusCode)" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "⚠️ Health check failed, but function may still be starting up" -ForegroundColor Yellow
                Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Gray
            }
            
            # Test debug endpoint
            Write-Host "`n🔍 Testing debug endpoint..." -ForegroundColor Yellow
            try {
                $debugResponse = Invoke-RestMethod -Uri "${FUNCTION_URL}?debug=true" -TimeoutSec 30
                if ($debugResponse.debug_info) {
                    Write-Host "🔧 Configuration Check:" -ForegroundColor Cyan
                    Write-Host "  MongoDB: $($debugResponse.debug_info.mongo_configured)" -ForegroundColor $(if ($debugResponse.debug_info.mongo_configured) {'Green'} else {'Red'})
                    Write-Host "  Alchemy: $($debugResponse.debug_info.alchemy_configured)" -ForegroundColor $(if ($debugResponse.debug_info.alchemy_configured) {'Green'} else {'Red'})
                    Write-Host "  Networks: $($debugResponse.debug_info.supported_networks -join ', ')" -ForegroundColor White
                } else {
                    Write-Host "✅ Debug endpoint accessible" -ForegroundColor Green
                }
            } catch {
                Write-Host "⚠️ Debug endpoint test failed: $($_.Exception.Message)" -ForegroundColor Yellow
            }
            
        } else {
            Write-Host "⚠️ Deployment may have succeeded but couldn't retrieve function URL" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "⚠️ Could not retrieve function information: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Provide next steps and management information
Write-Host "`n📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

if ($FUNCTION_URL) {
    Write-Host "1. 🌐 Test your function: $FUNCTION_URL" -ForegroundColor White
    Write-Host "2. 🔍 Debug configuration: ${FUNCTION_URL}?debug=true" -ForegroundColor White
    Write-Host "3. 📊 Run analysis test:" -ForegroundColor White
    Write-Host "   POST to $FUNCTION_URL with JSON:" -ForegroundColor Gray
    Write-Host '   {"network": "ethereum", "analysis_type": "buy", "debug": true}' -ForegroundColor Gray
    Write-Host "4. 🗄️ Check database connectivity and wallet data" -ForegroundColor White
    Write-Host "5. ⏰ Set up Cloud Scheduler to call function periodically" -ForegroundColor White
} else {
    Write-Host "1. 🌐 Check your function at: https://console.cloud.google.com/functions/list?project=$ProjectId" -ForegroundColor White
    Write-Host "2. 🔧 Verify the function is deployed and get the URL" -ForegroundColor White
    Write-Host "3. 📱 Test the function endpoints manually" -ForegroundColor White
}

if (!$deploymentSuccess -or !$foundVars.ContainsKey("MONGO_URI")) {
    Write-Host "`n⚠️ ENVIRONMENT VARIABLES:" -ForegroundColor Yellow
    Write-Host "If environment variables weren't set automatically:" -ForegroundColor White
    Write-Host "1. Go to: https://console.cloud.google.com/functions/details/$Region/$FunctionName?project=$ProjectId" -ForegroundColor White
    Write-Host "2. Click 'Edit'" -ForegroundColor White
    Write-Host "3. Go to 'Runtime, build, connections and security settings'" -ForegroundColor White
    Write-Host "4. Add the variables from your .env.production file" -ForegroundColor White
    Write-Host "5. Click 'Deploy' to update the function" -ForegroundColor White
}

Write-Host "`n🔧 MANAGEMENT COMMANDS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "View logs:" -ForegroundColor White
Write-Host "  gcloud functions logs read $FunctionName --region=$Region --limit=20" -ForegroundColor Gray

Write-Host "`nStream logs (real-time):" -ForegroundColor White
Write-Host "  gcloud functions logs tail $FunctionName --region=$Region" -ForegroundColor Gray

Write-Host "`nUpdate function:" -ForegroundColor White
Write-Host "  gcloud functions deploy $FunctionName --region=$Region --source=." -ForegroundColor Gray

Write-Host "`nDelete function:" -ForegroundColor White
Write-Host "  gcloud functions delete $FunctionName --region=$Region" -ForegroundColor Gray

Write-Host "`nView function details:" -ForegroundColor White
Write-Host "  gcloud functions describe $FunctionName --region=$Region" -ForegroundColor Gray

Write-Host "`nTest function locally:" -ForegroundColor White
Write-Host "  functions-framework --target=crypto_analysis_function --debug" -ForegroundColor Gray

Write-Host "`n📊 TESTING EXAMPLES:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

if ($FUNCTION_URL) {
    Write-Host "PowerShell test commands:" -ForegroundColor White
    Write-Host "`n# Health check" -ForegroundColor Gray
    Write-Host "Invoke-RestMethod -Uri '$FUNCTION_URL'" -ForegroundColor Cyan
    
    Write-Host "`n# Debug mode" -ForegroundColor Gray
    Write-Host "Invoke-RestMethod -Uri '${FUNCTION_URL}?debug=true'" -ForegroundColor Cyan
    
    Write-Host "`n# Run analysis" -ForegroundColor Gray
    Write-Host '$testData = @{' -ForegroundColor Cyan
    Write-Host '    network = "ethereum"' -ForegroundColor Cyan
    Write-Host '    analysis_type = "buy"' -ForegroundColor Cyan
    Write-Host '    num_wallets = 50' -ForegroundColor Cyan
    Write-Host '    days_back = 1.0' -ForegroundColor Cyan
    Write-Host '    debug = $true' -ForegroundColor Cyan
    Write-Host '} | ConvertTo-Json' -ForegroundColor Cyan
    Write-Host '' -ForegroundColor Cyan
    Write-Host "Invoke-RestMethod -Uri '$FUNCTION_URL' -Method POST -Body `$testData -ContentType 'application/json'" -ForegroundColor Cyan
}

Write-Host "`n🎊 CLOUD FUNCTIONS DEPLOYMENT COMPLETE!" -ForegroundColor Green
if ($FUNCTION_URL) {
    Write-Host "Your Crypto Analysis Function is now running at: $FUNCTION_URL" -ForegroundColor Cyan
} else {
    Write-Host "Check Google Cloud Console for your function URL" -ForegroundColor Cyan
}
Write-Host "=========================================" -ForegroundColor Green