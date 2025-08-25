param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "asia-southeast1",
    [string]$ServiceName = "crypto-monitor"
)

Write-Host "🚀 Complete Crypto Monitor Deployment" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Service: $ServiceName" -ForegroundColor White
Write-Host "=============================================" -ForegroundColor Cyan

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
$requiredFiles = @("main.py", "requirements.txt", "Dockerfile", ".env.production")
foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Host "❌ ERROR: Required file missing: $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Found $file" -ForegroundColor Green
}

# Check templates
if (!(Test-Path "templates")) {
    Write-Host "⚠️ Templates directory missing, creating..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Name "templates" -Force
}

if (!(Test-Path "templates/dashboard.html") -or !(Test-Path "templates/control_panel.html")) {
    Write-Host "⚠️ Template files missing, this may cause issues" -ForegroundColor Yellow
}

# Read and validate environment variables
Write-Host "`n🔧 Processing environment variables..." -ForegroundColor Yellow

if (!(Test-Path ".env.production")) {
    Write-Host "❌ ERROR: .env.production file not found" -ForegroundColor Red
    exit 1
}

$envContent = Get-Content ".env.production" -Encoding UTF8
$validEnvVars = @()
$requiredVars = @("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ALCHEMY_API_KEY")
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

# Create env.yaml file for gcloud
Write-Host "`n📄 Creating env.yaml for deployment..." -ForegroundColor Yellow

$yamlLines = @()
foreach ($line in $envContent) {
    if ($line -and $line -notmatch "^\s*#" -and $line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        
        # Remove surrounding quotes if they exist
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or 
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        
        # Always quote the value in YAML to handle special characters
        $yamlLines += "$key`: `"$value`""
    }
}

# Write YAML file with UTF-8 encoding (no BOM)
$yamlContent = $yamlLines -join "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("env.yaml", $yamlContent, $utf8NoBom)

Write-Host "✅ Created env.yaml with $($yamlLines.Count) variables" -ForegroundColor Green

# Set up Google Cloud project
Write-Host "`n☁️ Setting up Google Cloud..." -ForegroundColor Yellow

try {
    Write-Host "Setting project to $ProjectId..." -ForegroundColor Gray
    gcloud config set project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Failed to set project" }
    
    Write-Host "Enabling required APIs..." -ForegroundColor Gray
    gcloud services enable cloudbuild.googleapis.com --quiet
    gcloud services enable run.googleapis.com --quiet
    gcloud services enable artifactregistry.googleapis.com --quiet
    
    Write-Host "✅ Google Cloud setup complete" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to setup Google Cloud project" -ForegroundColor Red
    Write-Host "Please verify your project ID and permissions" -ForegroundColor Yellow
    exit 1
}

# Deploy to Cloud Run
Write-Host "`n🚀 Deploying to Cloud Run..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

$deploymentSuccess = $false

try {
    # First attempt: Deploy with env.yaml file
    Write-Host "Attempting deployment with environment file..." -ForegroundColor Gray
    
    gcloud run deploy $ServiceName `
        --source . `
        --region $Region `
        --env-vars-file env.yaml `
        --platform managed `
        --allow-unauthenticated `
        --port 8080 `
        --memory 2Gi `
        --cpu 2 `
        --timeout 900 `
        --max-instances 10 `
        --min-instances 0 `
        --concurrency 80 `
        --quiet

    if ($LASTEXITCODE -eq 0) {
        $deploymentSuccess = $true
        Write-Host "✅ Deployment with environment file successful!" -ForegroundColor Green
    } else {
        throw "Deployment with env file failed"
    }
    
} catch {
    Write-Host "⚠️ Environment file deployment failed, trying alternative..." -ForegroundColor Yellow
    
    try {
        # Second attempt: Deploy with --set-env-vars
        Write-Host "Attempting deployment with direct environment variables..." -ForegroundColor Gray
        $envString = ($validEnvVars -join ",")
        
        gcloud run deploy $ServiceName `
            --source . `
            --region $Region `
            --set-env-vars $envString `
            --platform managed `
            --allow-unauthenticated `
            --port 8080 `
            --memory 2Gi `
            --cpu 2 `
            --timeout 900 `
            --max-instances 10 `
            --quiet

        if ($LASTEXITCODE -eq 0) {
            $deploymentSuccess = $true
            Write-Host "✅ Deployment with direct env vars successful!" -ForegroundColor Green
        } else {
            throw "Direct env vars deployment failed"
        }
        
    } catch {
        Write-Host "⚠️ Direct environment variables failed, trying basic deployment..." -ForegroundColor Yellow
        
        try {
            # Third attempt: Basic deployment without env vars
            Write-Host "Attempting basic deployment (env vars will need manual setup)..." -ForegroundColor Gray
            
            gcloud run deploy $ServiceName `
                --source . `
                --region $Region `
                --platform managed `
                --allow-unauthenticated `
                --port 8080 `
                --memory 2Gi `
                --cpu 2 `
                --timeout 900 `
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
            exit 1
        }
    }
}

# Get service information
if ($deploymentSuccess) {
    Write-Host "`n📡 Getting service information..." -ForegroundColor Yellow
    
    try {
        $SERVICE_URL = gcloud run services describe $ServiceName --region $Region --format="value(status.url)"
        
        if ($SERVICE_URL) {
            Write-Host "`n🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
            Write-Host "=============================================" -ForegroundColor Green
            Write-Host "🌐 Service URL: $SERVICE_URL" -ForegroundColor White
            Write-Host "🔧 Control Panel: $SERVICE_URL/control-panel" -ForegroundColor White
            Write-Host "📱 Telegram Status: $SERVICE_URL/telegram/status" -ForegroundColor White
            Write-Host "🏥 Health Check: $SERVICE_URL/health" -ForegroundColor White
            Write-Host "📊 Analysis Results: $SERVICE_URL/analysis/results" -ForegroundColor White
            Write-Host "⏰ Scheduler Script: $SERVICE_URL/scheduler/script" -ForegroundColor White
            Write-Host "=============================================" -ForegroundColor Green
            
            # Test the deployment
            Write-Host "`n🧪 Testing deployment..." -ForegroundColor Yellow
            Write-Host "Waiting for service to be ready..." -ForegroundColor Gray
            Start-Sleep -Seconds 15
            
            try {
                $healthResponse = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing -TimeoutSec 30
                if ($healthResponse.StatusCode -eq 200) {
                    Write-Host "✅ Service is healthy and responding!" -ForegroundColor Green
                    
                    # Parse health response to check components
                    try {
                        $healthData = $healthResponse.Content | ConvertFrom-Json
                        
                        if ($healthData.telegram_configured) {
                            Write-Host "✅ Telegram is properly configured!" -ForegroundColor Green
                        } else {
                            Write-Host "⚠️ Telegram configuration needs verification" -ForegroundColor Yellow
                        }
                        
                        if ($healthData.database_connected) {
                            Write-Host "✅ Database connection is working!" -ForegroundColor Green
                        } else {
                            Write-Host "⚠️ Database connection may need setup" -ForegroundColor Yellow
                        }
                        
                        if ($healthData.monitoring_active) {
                            Write-Host "✅ Monitoring is active!" -ForegroundColor Green
                        } else {
                            Write-Host "ℹ️ Monitoring is currently stopped (normal for new deployment)" -ForegroundColor Cyan
                        }
                        
                    } catch {
                        Write-Host "ℹ️ Health check successful, but couldn't parse detailed status" -ForegroundColor Cyan
                    }
                    
                } else {
                    Write-Host "⚠️ Service deployed but health check returned HTTP $($healthResponse.StatusCode)" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "⚠️ Health check failed, but service may still be starting up" -ForegroundColor Yellow
                Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Gray
            }
            
        } else {
            Write-Host "⚠️ Deployment may have succeeded but couldn't retrieve service URL" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "⚠️ Could not retrieve service information: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Provide next steps and management information
Write-Host "`n📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

if ($SERVICE_URL) {
    Write-Host "1. 🌐 Open your service: $SERVICE_URL" -ForegroundColor White
    Write-Host "2. 🔧 Access control panel: $SERVICE_URL/control-panel" -ForegroundColor White
    Write-Host "3. 📱 Test Telegram notifications in the control panel" -ForegroundColor White
    Write-Host "4. 📊 Run sample analysis to verify everything works" -ForegroundColor White
    Write-Host "5. ⏰ Set up Cloud Scheduler using: $SERVICE_URL/scheduler/script" -ForegroundColor White
} else {
    Write-Host "1. 🌐 Check your service at: https://console.cloud.google.com/run?project=$ProjectId" -ForegroundColor White
    Write-Host "2. 🔧 Verify the service is running and get the URL" -ForegroundColor White
    Write-Host "3. 📱 Test the service endpoints manually" -ForegroundColor White
}

if (!$deploymentSuccess -or !$foundVars.ContainsKey("TELEGRAM_BOT_TOKEN")) {
    Write-Host "`n⚠️ ENVIRONMENT VARIABLES:" -ForegroundColor Yellow
    Write-Host "If environment variables weren't set automatically:" -ForegroundColor White
    Write-Host "1. Go to: https://console.cloud.google.com/run/detail/$Region/$ServiceName/revisions?project=$ProjectId" -ForegroundColor White
    Write-Host "2. Click 'Edit & Deploy New Revision'" -ForegroundColor White
    Write-Host "3. Go to 'Variables & Secrets' tab" -ForegroundColor White
    Write-Host "4. Add the variables from your .env.production file" -ForegroundColor White
}

Write-Host "`n🔧 MANAGEMENT COMMANDS:" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "View logs:" -ForegroundColor White
Write-Host "  gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName' --limit=20" -ForegroundColor Gray

Write-Host "`nUpdate service:" -ForegroundColor White
Write-Host "  gcloud run services update $ServiceName --region=$Region" -ForegroundColor Gray

Write-Host "`nRedeploy:" -ForegroundColor White
Write-Host "  gcloud run deploy $ServiceName --source . --region=$Region" -ForegroundColor Gray

Write-Host "`nDelete service:" -ForegroundColor White
Write-Host "  gcloud run services delete $ServiceName --region=$Region" -ForegroundColor Gray

Write-Host "`nView service details:" -ForegroundColor White
Write-Host "  gcloud run services describe $ServiceName --region=$Region" -ForegroundColor Gray

# Clean up temporary files
Write-Host "`n🧹 Cleaning up..." -ForegroundColor Yellow
if (Test-Path "env.yaml") {
    Remove-Item "env.yaml" -Force
    Write-Host "✅ Cleaned up temporary env.yaml file" -ForegroundColor Green
}

Write-Host "`n🎊 DEPLOYMENT COMPLETE!" -ForegroundColor Green
if ($SERVICE_URL) {
    Write-Host "Your Crypto Monitor is now running at: $SERVICE_URL" -ForegroundColor Cyan
} else {
    Write-Host "Check Google Cloud Console for your service URL" -ForegroundColor Cyan
}
Write-Host "=============================================" -ForegroundColor Green