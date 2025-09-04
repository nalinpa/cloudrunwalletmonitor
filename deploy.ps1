# Force Fresh Deployment Script
# This ensures the new code is actually deployed

Write-Host "🔄 FORCING FRESH DEPLOYMENT" -ForegroundColor Cyan

$PROJECT_ID = "crypto-tracker-cloudrun"
$FUNCTION_NAME = "crypto-analysis-function" 
$REGION = "asia-southeast1"

# Step 1: Delete the existing function to force fresh deployment
Write-Host "`n1️⃣ Deleting existing function..." -ForegroundColor Yellow
try {
    gcloud functions delete $FUNCTION_NAME --region=$REGION --quiet
    Write-Host "✅ Function deleted successfully" -ForegroundColor Green
    Start-Sleep -Seconds 5
} catch {
    Write-Host "⚠️ Function might not exist or already deleted" -ForegroundColor Yellow
}

# Step 2: Verify files are updated
Write-Host "`n2️⃣ Verifying files..." -ForegroundColor Yellow

$buyAnalyzerPath = "core\analysis\buy_analyzer.py"
if (Test-Path $buyAnalyzerPath) {
    $content = Get-Content $buyAnalyzerPath -Raw
    if ($content -match "STARTING BUY ANALYSIS") {
        Write-Host "✅ buy_analyzer.py has debug logging" -ForegroundColor Green
    } else {
        Write-Host "❌ buy_analyzer.py still has old code!" -ForegroundColor Red
        Write-Host "Replace core/analysis/buy_analyzer.py with the debug version" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "❌ buy_analyzer.py not found!" -ForegroundColor Red
    exit 1
}

# Step 3: Clean any cached builds
Write-Host "`n3️⃣ Cleaning cache..." -ForegroundColor Yellow
if (Test-Path ".gcloudignore") {
    Write-Host "✅ .gcloudignore exists" -ForegroundColor Green
} else {
    Write-Host "Creating .gcloudignore..." -ForegroundColor Gray
    @"
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
.env
venv/
.venv/
.git/
.gitignore
README.md
*.log
.pytest_cache/
.coverage
.vscode/
.idea/
node_modules/
"@ | Out-File -FilePath ".gcloudignore" -Encoding UTF8
}

# Step 4: Set project and ensure APIs are enabled
Write-Host "`n4️⃣ Setting up project..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID
gcloud services enable cloudfunctions.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet

# Step 5: Deploy fresh function
Write-Host "`n5️⃣ Deploying fresh function..." -ForegroundColor Yellow

# Check for environment file
if (Test-Path ".env.production") {
    Write-Host "✅ Found .env.production" -ForegroundColor Green
    
    # Process environment variables
    $envContent = Get-Content ".env.production" -Encoding UTF8
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
    
    if ($envVars.Count -gt 0) {
        $envString = ($envVars -join ",")
        Write-Host "✅ Found $($envVars.Count) environment variables" -ForegroundColor Green
    }
} else {
    Write-Host "❌ .env.production not found!" -ForegroundColor Red
    exit 1
}

# Deploy command
Write-Host "`nDeploying with these settings:" -ForegroundColor White
Write-Host "  Project: $PROJECT_ID" -ForegroundColor Gray
Write-Host "  Function: $FUNCTION_NAME" -ForegroundColor Gray
Write-Host "  Region: $REGION" -ForegroundColor Gray
Write-Host "  Memory: 2Gi" -ForegroundColor Gray
Write-Host "  Timeout: 540s" -ForegroundColor Gray

$deployArgs = @(
    "functions", "deploy", $FUNCTION_NAME,
    "--gen2",
    "--runtime", "python311", 
    "--region", $REGION,
    "--source", ".",
    "--entry-point", "crypto_analysis_function",
    "--trigger-http",
    "--allow-unauthenticated",
    "--memory", "2Gi",
    "--timeout", "540s",
    "--max-instances", "10"
)

if ($envVars.Count -gt 0) {
    $deployArgs += "--set-env-vars"
    $deployArgs += $envString
}

& gcloud @deployArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n🎉 FRESH DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    
    # Get function URL
    $FUNCTION_URL = gcloud functions describe $FUNCTION_NAME --region $REGION --format="value(serviceConfig.uri)"
    Write-Host "🌐 Function URL: $FUNCTION_URL" -ForegroundColor White
    
} else {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✅ Fresh deployment complete!" -ForegroundColor Green