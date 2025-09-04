param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId = "crypto-tracker-cloudrun",
    
    [string]$Region = "asia-southeast1",
    [string]$FunctionName = "crypto-monitor"
)

Write-Host "🚀 Deploying to Cloud Functions" -ForegroundColor Cyan
Write-Host "Project: $ProjectId | Region: $Region | Function: $FunctionName" -ForegroundColor White

# Set Google Cloud project
Write-Host "`n☁️ Setting up Google Cloud..." -ForegroundColor Yellow
gcloud config set project $ProjectId
gcloud services enable cloudfunctions.googleapis.com cloudbuild.googleapis.com --quiet

# Process environment variables if .env.production exists
$envArgs = @()
if (Test-Path ".env.production") {
    Write-Host "📄 Processing environment variables..." -ForegroundColor Yellow
    
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
        $envArgs = @("--set-env-vars", $envString)
        Write-Host "✅ Found $($envVars.Count) environment variables" -ForegroundColor Green
    }
}

# Deploy to Cloud Functions
Write-Host "`n🚀 Deploying to Cloud Functions..." -ForegroundColor Yellow

$deployArgs = @(
    "functions", "deploy", $FunctionName,
    "--gen2",
    "--runtime", "python311",
    "--region", $Region,
    "--source", ".",
    "--entry-point", "crypto_analysis_function",
    "--trigger-http",
    "--allow-unauthenticated",
    "--memory", "2Gi",
    "--timeout", "540s",
    "--max-instances", "10",
    "--quiet"
)

if ($envArgs.Count -gt 0) {
    $deployArgs += $envArgs
}

& gcloud @deployArgs

if ($LASTEXITCODE -eq 0) {
    # Get function URL
    $FUNCTION_URL = gcloud functions describe $FunctionName --region $Region --format="value(serviceConfig.uri)"
    
    Write-Host "`n🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "🌐 Function URL: $FUNCTION_URL" -ForegroundColor White
    Write-Host "🔧 Health Check: $FUNCTION_URL" -ForegroundColor White
    Write-Host "📊 Analysis (POST): $FUNCTION_URL" -ForegroundColor White
} else {
    Write-Host "❌ Deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Done!" -ForegroundColor Green