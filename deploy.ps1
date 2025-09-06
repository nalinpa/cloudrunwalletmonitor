# Enhanced Deploy Script for Cloud Functions with pandas-ta
param(
    [string]$ProjectId = "crypto-tracker-cloudrun",
    [string]$FunctionName = "crypto-analysis-function",
    [string]$Region = "asia-southeast1"
)

Write-Host "☁️ CLOUD FUNCTIONS DEPLOYMENT WITH PANDAS-TA" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Function: $FunctionName" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White

# Step 1: Validate files exist
Write-Host "`n1️⃣ Validating files..." -ForegroundColor Yellow

$requiredFiles = @("Dockerfile", "requirements.txt", "main.py", ".env.production")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file missing" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Missing required files: $($missingFiles -join ', ')" -ForegroundColor Red
    exit 1
}

# Step 2: Check environment variables
Write-Host "`n2️⃣ Checking environment configuration..." -ForegroundColor Yellow

if (Test-Path ".env.production") {
    $envContent = Get-Content ".env.production" -Encoding UTF8
    
    # Parse environment variables for Cloud Functions
    $envVars = @()
    foreach ($line in $envContent) {
        if ($line -and $line -notmatch "^\s*#" -and $line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove surrounding quotes
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or 
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            $envVars += "$key=$value"
        }
    }
    
    Write-Host "✅ Found $($envVars.Count) environment variables" -ForegroundColor Green
} else {
    Write-Host "❌ .env.production not found!" -ForegroundColor Red
    exit 1
}

# Step 3: Set up gcloud
Write-Host "`n3️⃣ Setting up Google Cloud..." -ForegroundColor Yellow
gcloud config set project $ProjectId
gcloud services enable cloudbuild.googleapis.com cloudfunctions.googleapis.com

# Step 4: Build and deploy Cloud Function
Write-Host "`n4️⃣ Deploying Cloud Function..." -ForegroundColor Yellow
Write-Host "⏰ This may take 3-5 minutes..." -ForegroundColor Gray

$deployArgs = @(
    "functions", "deploy", $FunctionName,
    "--gen2",
    "--runtime", "python311",
    "--region", $Region,
    "--source", ".",
    "--entry-point", "main",
    "--trigger-http",
    "--allow-unauthenticated",
    "--memory", "2Gi",
    "--cpu", "1",
    "--timeout", "540s",
    "--max-instances", "10"
)

# Add environment variables
if ($envVars.Count -gt 0) {
    foreach ($envVar in $envVars) {
        $deployArgs += "--set-env-vars"
        $deployArgs += $envVar
    }
}

& gcloud @deployArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    
    # Get function URL
    $functionUrl = gcloud functions describe $FunctionName --region $Region --gen2 --format="value(serviceConfig.uri)"
    Write-Host "🌐 Function URL: $functionUrl" -ForegroundColor White
    
    # Test the deployment
    Write-Host "`n🧪 Testing deployment..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
    
    try {
        $healthResponse = Invoke-RestMethod -Uri "$functionUrl" -TimeoutSec 30
        if ($healthResponse.status -eq "healthy") {
            Write-Host "✅ Function is healthy and running!" -ForegroundColor Green
            Write-Host "🤖 AI Features: $($healthResponse.features -join ', ')" -ForegroundColor White
        }
    } catch {
        Write-Host "⚠️ Health check failed, but function may still be starting" -ForegroundColor Yellow
    }
    
    Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Test AI analysis: POST to $functionUrl" -ForegroundColor White
    Write-Host "2. Check logs: gcloud functions logs read $FunctionName --region=$Region" -ForegroundColor White
    Write-Host "3. Monitor: https://console.cloud.google.com/functions/details/$Region/$FunctionName" -ForegroundColor White
    
} else {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n🚀 Cloud Function deployment complete!" -ForegroundColor Green