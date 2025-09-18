# Fix Test Function Authentication
param(
    [string]$ProjectId = "crypto-tracker-cloudrun",
    [string]$Region = "asia-southeast1"
)

$TEST_FUNCTION = "crypto-analysis-function-test"

Write-Host "🔐 FIXING TEST FUNCTION AUTHENTICATION" -ForegroundColor Cyan
Write-Host "Function: $TEST_FUNCTION" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White

# Set project
gcloud config set project $ProjectId

# Add public access to the function
Write-Host "`n1️⃣ Adding public access..." -ForegroundColor Yellow
gcloud functions add-iam-policy-binding $TEST_FUNCTION `
    --region=$Region `
    --member="allUsers" `
    --role="roles/cloudfunctions.invoker"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Public access granted!" -ForegroundColor Green
    
    # Get the function URL
    $testUrl = gcloud functions describe $TEST_FUNCTION --region=$Region --gen2 --format="value(serviceConfig.uri)"
    
    Write-Host "`n🌐 Test Function URL: $testUrl" -ForegroundColor Green
    Write-Host "`n🧪 Test it now:" -ForegroundColor Cyan
    Write-Host "Invoke-RestMethod -Uri '$testUrl'" -ForegroundColor White
    
    # Test it
    Write-Host "`n⏱️ Testing access..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri $testUrl -TimeoutSec 15
        Write-Host "✅ SUCCESS! Function is now accessible" -ForegroundColor Green
        Write-Host "Status: $($response.status)" -ForegroundColor White
    } catch {
        Write-Host "❌ Still having issues: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "ℹ️ Wait 1-2 minutes for permissions to propagate" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "❌ Failed to set permissions" -ForegroundColor Red
    Write-Host "Try running the deploy script again" -ForegroundColor Yellow
}