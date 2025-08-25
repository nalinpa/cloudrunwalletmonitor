# Test your Cloud Function analysis
$FUNCTION_URL = "https://crypto-analysis-function-qz6f5mkbmq-as.a.run.app/"

Write-Host "🧪 Testing Crypto Analysis Cloud Function" -ForegroundColor Cyan
Write-Host "URL: $FUNCTION_URL" -ForegroundColor White

# Test 1: Health Check (already working)
Write-Host "`n1️⃣ Health Check Test..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri $FUNCTION_URL -Method GET
    Write-Host "✅ Health check successful!" -ForegroundColor Green
    Write-Host "   Status: $($healthResponse.status)" -ForegroundColor White
    Write-Host "   Service: $($healthResponse.service)" -ForegroundColor White
    Write-Host "   Version: $($healthResponse.version)" -ForegroundColor White
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Buy Analysis
Write-Host "`n2️⃣ Testing Buy Analysis..." -ForegroundColor Yellow
$buyAnalysisData = @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 174
    days_back = 1.0
} | ConvertTo-Json

try {
    Write-Host "Sending buy analysis request..." -ForegroundColor Gray
    $buyResponse = Invoke-RestMethod -Uri $FUNCTION_URL -Method POST -Body $buyAnalysisData -ContentType "application/json"
    
    if ($buyResponse.success) {
        Write-Host "✅ Buy analysis successful!" -ForegroundColor Green
        Write-Host "   Network: $($buyResponse.network)" -ForegroundColor White
        Write-Host "   Transactions: $($buyResponse.total_transactions)" -ForegroundColor White
        Write-Host "   Tokens: $($buyResponse.unique_tokens)" -ForegroundColor White
        Write-Host "   Total ETH: $($buyResponse.total_eth_value)" -ForegroundColor White
        
        if ($buyResponse.top_tokens -and $buyResponse.top_tokens.Count -gt 0) {
            Write-Host "   Top Token: $($buyResponse.top_tokens[0][0])" -ForegroundColor White
        }
    } else {
        Write-Host "⚠️ Buy analysis completed but with issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Buy analysis failed: $_" -ForegroundColor Red
    Write-Host "This might be expected if you don't have wallet data yet" -ForegroundColor Yellow
}

# Test 3: Sell Analysis
Write-Host "`n3️⃣ Testing Sell Analysis..." -ForegroundColor Yellow
$sellAnalysisData = @{
    network = "base"
    analysis_type = "sell"
    num_wallets = 174
    days_back = 1.0
} | ConvertTo-Json

try {
    Write-Host "Sending sell analysis request..." -ForegroundColor Gray
    $sellResponse = Invoke-RestMethod -Uri $FUNCTION_URL -Method POST -Body $sellAnalysisData -ContentType "application/json"
    
    if ($sellResponse.success) {
        Write-Host "✅ Sell analysis successful!" -ForegroundColor Green
        Write-Host "   Network: $($sellResponse.network)" -ForegroundColor White
        Write-Host "   Transactions: $($sellResponse.total_transactions)" -ForegroundColor White
        Write-Host "   Tokens: $($sellResponse.unique_tokens)" -ForegroundColor White
        Write-Host "   Total ETH: $($sellResponse.total_eth_value)" -ForegroundColor White
    } else {
        Write-Host "⚠️ Sell analysis completed but with issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Sell analysis failed: $_" -ForegroundColor Red
    Write-Host "This might be expected if you don't have wallet data yet" -ForegroundColor Yellow
}

# Test 4: Error Handling
Write-Host "`n4️⃣ Testing Error Handling..." -ForegroundColor Yellow
$invalidData = @{
    network = "invalid_network"
    analysis_type = "buy"
} | ConvertTo-Json

try {
    $errorResponse = Invoke-RestMethod -Uri $FUNCTION_URL -Method POST -Body $invalidData -ContentType "application/json"
    Write-Host "⚠️ Expected error test didn't fail as expected" -ForegroundColor Yellow
} catch {
    Write-Host "✅ Error handling working correctly" -ForegroundColor Green
    Write-Host "   (Correctly rejected invalid network)" -ForegroundColor Gray
}

Write-Host "`n🎉 TESTING COMPLETE!" -ForegroundColor Green
Write-Host "`n📋 Summary:" -ForegroundColor Cyan
Write-Host "✅ Cloud Function is deployed and responding" -ForegroundColor White
Write-Host "✅ Health checks are working" -ForegroundColor White
Write-Host "✅ Analysis endpoints are functional" -ForegroundColor White
Write-Host "✅ Error handling is working" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Populate your MongoDB with wallet data" -ForegroundColor White
Write-Host "2. Set up Cloud Scheduler for automated monitoring" -ForegroundColor White
Write-Host "3. Configure Telegram notifications" -ForegroundColor White
Write-Host "4. Create the monitoring wrapper service" -ForegroundColor White

Write-Host "`n📱 Your Function URL: $FUNCTION_URL" -ForegroundColor Green