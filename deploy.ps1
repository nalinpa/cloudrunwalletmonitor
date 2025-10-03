# Test Deploy Script (based on your working one)
param(
    [string]$ProjectId = "crypto-tracker-cloudrun",
    [string]$FunctionName = "crypto-analysis-function-test",  # Changed to test
    [string]$Region = "asia-southeast1"
)

Write-Host "Deploying test function..." -ForegroundColor Cyan

# Parse environment variables
$envVars = @()
if (Test-Path ".env.production") {
    Get-Content ".env.production" | Where-Object { $_ -match "^[^#].*=" } | ForEach-Object {
        if ($_ -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or 
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            $envVars += "$key=$value"
        }
    }
}

gcloud config set project $ProjectId

# Same deployment as production, just different name
gcloud functions deploy $FunctionName `
    --gen2 `
    --runtime python311 `
    --region $Region `
    --source . `
    --entry-point main `
    --trigger-http `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 1 `
    --timeout 540s `
    --max-instances 10 `
    --set-env-vars ($envVars -join ',')

if ($LASTEXITCODE -eq 0) {
    $url = gcloud functions describe $FunctionName --region $Region --gen2 --format="value(serviceConfig.uri)"
    Write-Host "`nSUCCESS - Test function deployed" -ForegroundColor Green
    Write-Host "URL: $url" -ForegroundColor Yellow
    
    Write-Host "`nTest payload:" -ForegroundColor Cyan
    Write-Host 'Invoke-RestMethod -Uri "' $url '" -Method POST -ContentType "application/json" -Body ''{"network":"ethereum","analysis_type":"buy","num_wallets":50,"store_verified_trades":true,"notifications":false}''' -ForegroundColor White
} else {
    Write-Host "FAILED" -ForegroundColor Red
}