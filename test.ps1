# Cloud Functions Comprehensive Test Script
param(
    [string]$FunctionUrl = "",
    [string]$ProjectId = "crypto-tracker-cloudrun",
    [string]$Region = "asia-southeast1", 
    [string]$FunctionName = "crypto-analysis-function"
)

Write-Host "🧪 COMPREHENSIVE CLOUD FUNCTIONS TEST SUITE" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Get function URL if not provided
if ([string]::IsNullOrEmpty($FunctionUrl)) {
    Write-Host "🔍 Getting function URL..." -ForegroundColor Yellow
    try {
        $FunctionUrl = gcloud functions describe $FunctionName --region=$Region --format="value(serviceConfig.uri)" 2>$null
        if ([string]::IsNullOrEmpty($FunctionUrl)) {
            Write-Host "❌ Could not get function URL. Please provide it manually." -ForegroundColor Red
            Write-Host "Usage: .\test-function.ps1 -FunctionUrl 'YOUR_FUNCTION_URL'" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "✅ Function URL: $FunctionUrl" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to get function URL: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Function URL: $FunctionUrl" -ForegroundColor White
Write-Host "Project: $ProjectId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Function: $FunctionName" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan

$global:testResults = @()
$global:testNumber = 1

# Helper function to run test
function Run-Test {
    param(
        [string]$TestName,
        [string]$Method = "GET",
        [string]$Endpoint = "",
        [hashtable]$Body = $null,
        [int]$ExpectedStatus = 200,
        [string]$Description = ""
    )
    
    Write-Host "`n$global:testNumber️⃣ $TestName" -ForegroundColor Yellow
    if ($Description) {
        Write-Host "   $Description" -ForegroundColor Gray
    }
    
    $url = $FunctionUrl + $Endpoint
    $success = $false
    $responseData = $null
    $statusCode = 0
    $errorMessage = ""
    
    try {
        Write-Host "   🔄 Testing: $Method $url" -ForegroundColor Gray
        
        if ($Method -eq "GET") {
            $response = Invoke-WebRequest -Uri $url -Method GET -UseBasicParsing -TimeoutSec 30
        } else {
            $jsonBody = $Body | ConvertTo-Json -Depth 10
            Write-Host "   📤 Body: $($jsonBody.Substring(0, [Math]::Min(100, $jsonBody.Length)))..." -ForegroundColor Gray
            $response = Invoke-WebRequest -Uri $url -Method $Method -Body $jsonBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 60
        }
        
        $statusCode = $response.StatusCode
        
        if ($statusCode -eq $ExpectedStatus) {
            Write-Host "   ✅ SUCCESS: HTTP $statusCode" -ForegroundColor Green
            $success = $true
            
            # Try to parse JSON response
            try {
                $responseData = $response.Content | ConvertFrom-Json
                
                # Show key response fields
                if ($responseData.status) {
                    Write-Host "   📊 Status: $($responseData.status)" -ForegroundColor White
                }
                if ($responseData.message) {
                    Write-Host "   💬 Message: $($responseData.message)" -ForegroundColor White
                }
                if ($responseData.success -ne $null) {
                    Write-Host "   🎯 Success: $($responseData.success)" -ForegroundColor White
                }
                if ($responseData.version) {
                    Write-Host "   🏷️ Version: $($responseData.version)" -ForegroundColor White
                }
                
            } catch {
                Write-Host "   📝 Response: $($response.Content.Substring(0, [Math]::Min(200, $response.Content.Length)))..." -ForegroundColor White
            }
            
        } else {
            Write-Host "   ❌ FAILED: Expected HTTP $ExpectedStatus, got HTTP $statusCode" -ForegroundColor Red
            $errorMessage = "Wrong status code"
        }
        
    } catch {
        Write-Host "   ❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $errorMessage = $_.Exception.Message
        
        # Try to get status code from exception
        if ($_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
            Write-Host "   🔢 HTTP Status: $statusCode" -ForegroundColor Red
        }
    }
    
    # Store test result
    $testResult = [PSCustomObject]@{
        Number = $global:testNumber
        Name = $TestName
        Success = $success
        StatusCode = $statusCode
        ErrorMessage = $errorMessage
        ResponseData = $responseData
    }
    $global:testResults = $global:testResults + $testResult
    
    $global:testNumber++
    Start-Sleep -Seconds 1
}

# Start testing
Write-Host "`n🚀 STARTING TESTS..." -ForegroundColor Green

# Test 1: Basic Health Check
Run-Test -TestName "Health Check" -Description "Basic function availability test"

# Test 2: Health Check with Debug
Run-Test -TestName "Debug Health Check" -Endpoint "?debug=true" -Description "Health check with configuration info"

# Test 3: CORS Preflight
Run-Test -TestName "CORS Preflight" -Method "OPTIONS" -Description "CORS preflight request" -ExpectedStatus 204

# Test 4: Invalid Method
Run-Test -TestName "Invalid Method" -Method "PUT" -Description "Test error handling" -ExpectedStatus 405

# Test 5: Empty POST Body
Run-Test -TestName "Empty POST Body" -Method "POST" -Description "POST without JSON body" -ExpectedStatus 400

# Test 6: Basic Analysis - Ethereum Buy (Debug Mode)
Run-Test -TestName "Ethereum Buy Analysis (Debug)" -Method "POST" -Body @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 50
    days_back = 1.0
    debug = $true
} -Description "Debug analysis to check database/API connectivity"

# Test 7: Basic Analysis - Base Buy (Debug Mode)  
Run-Test -TestName "Base Buy Analysis (Debug)" -Method "POST" -Body @{
    network = "base"
    analysis_type = "buy"
    num_wallets = 50
    days_back = 1.0
    debug = $true
} -Description "Debug analysis for Base network"

# Test 8: Longer Timeframe Test
Run-Test -TestName "Extended Timeframe Test" -Method "POST" -Body @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 100
    days_back = 7.0
    debug = $false
} -Description "7-day analysis to find more activity"

# Test 9: Sell Analysis
Run-Test -TestName "Sell Analysis" -Method "POST" -Body @{
    network = "ethereum"  
    analysis_type = "sell"
    num_wallets = 50
    days_back = 3.0
    debug = $false
} -Description "Test sell analysis functionality"

# Test 10: Invalid Network
Run-Test -TestName "Invalid Network" -Method "POST" -Body @{
    network = "invalid"
    analysis_type = "buy"
    num_wallets = 10
    days_back = 1.0
} -Description "Error handling for invalid network" -ExpectedStatus 500

# Test 11: Invalid Analysis Type
Run-Test -TestName "Invalid Analysis Type" -Method "POST" -Body @{
    network = "ethereum"
    analysis_type = "invalid"
    num_wallets = 10
    days_back = 1.0
} -Description "Error handling for invalid analysis type" -ExpectedStatus 500

# Test 12: Large Wallet Count
Run-Test -TestName "Large Wallet Count" -Method "POST" -Body @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 200
    days_back = 1.0
    debug = $false
} -Description "Test with maximum wallet count"

# Test 13: Performance Test
Write-Host "`n1️3️⃣ Performance Test" -ForegroundColor Yellow
Write-Host "   Testing function response time under load" -ForegroundColor Gray

$performanceTimes = @()
$performanceBody = @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 50
    days_back = 1.0
} | ConvertTo-Json

for ($i = 1; $i -le 3; $i++) {
    Write-Host "   🔄 Performance run $i/3..." -ForegroundColor Gray
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        $perfResponse = Invoke-RestMethod -Uri $FunctionUrl -Method POST -Body $performanceBody -ContentType "application/json" -TimeoutSec 90
        $stopwatch.Stop()
        $responseTime = $stopwatch.ElapsedMilliseconds
        $performanceTimes += $responseTime
        Write-Host "   ⏱️ Response time: ${responseTime}ms" -ForegroundColor White
    } catch {
        $stopwatch.Stop()
        Write-Host "   ❌ Performance test $i failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2
}

if ($performanceTimes.Count -gt 0) {
    $avgTime = [int]($performanceTimes | Measure-Object -Average).Average
    $minTime = ($performanceTimes | Measure-Object -Minimum).Minimum
    $maxTime = ($performanceTimes | Measure-Object -Maximum).Maximum
    
    Write-Host "   📊 Performance Summary:" -ForegroundColor Cyan
    Write-Host "      Average: ${avgTime}ms" -ForegroundColor White
    Write-Host "      Min: ${minTime}ms" -ForegroundColor White  
    Write-Host "      Max: ${maxTime}ms" -ForegroundColor White
}

$global:testNumber++

# Generate Test Report
Write-Host "`n📊 TEST RESULTS SUMMARY" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

$passedTests = ($global:testResults | Where-Object { $_.Success }).Count
$totalTests = $global:testResults.Count

if ($totalTests -gt 0) {
    $successRate = [math]::Round(($passedTests / $totalTests) * 100, 1)
} else {
    $successRate = 0
}

Write-Host "✅ Passed: $passedTests" -ForegroundColor Green
Write-Host "❌ Failed: $($totalTests - $passedTests)" -ForegroundColor Red
Write-Host "📈 Success Rate: $successRate%" -ForegroundColor White
Write-Host "🕒 Total Tests: $totalTests" -ForegroundColor White

# Show failed tests details
$failedTests = $global:testResults | Where-Object { -not $_.Success }
if ($failedTests.Count -gt 0) {
    Write-Host "`n❌ FAILED TESTS:" -ForegroundColor Red
    Write-Host "=================" -ForegroundColor Red
    
    foreach ($test in $failedTests) {
        Write-Host "• $($test.Name) (HTTP $($test.StatusCode)): $($test.ErrorMessage)" -ForegroundColor Red
    }
}

# Show successful debug tests details
$debugTests = $global:testResults | Where-Object { $_.Name -like "*Debug*" -and $_.Success -and $_.ResponseData.debug_info }
if ($debugTests.Count -gt 0) {
    Write-Host "`n🔍 DEBUG INFORMATION:" -ForegroundColor Cyan
    Write-Host "=====================" -ForegroundColor Cyan
    
    foreach ($test in $debugTests) {
        $debugInfo = $test.ResponseData.debug_info
        Write-Host "`n📋 $($test.Name):" -ForegroundColor Yellow
        
        if ($debugInfo.database_debug) {
            $dbDebug = $debugInfo.database_debug
            Write-Host "   🗄️ Database:" -ForegroundColor White
            Write-Host "      Connected: $($dbDebug.database_connected)" -ForegroundColor $(if ($dbDebug.database_connected) {'Green'} else {'Red'})
            if ($dbDebug.total_wallets) {
                Write-Host "      Total Wallets: $($dbDebug.total_wallets)" -ForegroundColor White
            }
            if ($dbDebug.fetched_wallets_count -ne $null) {
                Write-Host "      Fetched Wallets: $($dbDebug.fetched_wallets_count)" -ForegroundColor White
            }
            if ($dbDebug.field_existence) {
                Write-Host "      Field Existence:" -ForegroundColor White
                $dbDebug.field_existence.PSObject.Properties | ForEach-Object {
                    Write-Host "         $($_.Name): $($_.Value)" -ForegroundColor Gray
                }
            }
        }
        
        if ($debugInfo.alchemy_debug) {
            $alchemyDebug = $debugInfo.alchemy_debug
            Write-Host "   🔗 Alchemy API:" -ForegroundColor White
            Write-Host "      API Responsive: $($alchemyDebug.alchemy_api_responsive)" -ForegroundColor $(if ($alchemyDebug.alchemy_api_responsive) {'Green'} else {'Red'})
            if ($alchemyDebug.blocks_in_range) {
                Write-Host "      Blocks in Range: $($alchemyDebug.blocks_in_range)" -ForegroundColor White
            }
            if ($alchemyDebug.alchemy_error) {
                Write-Host "      Error: $($alchemyDebug.alchemy_error)" -ForegroundColor Red
            }
        }
    }
}

# Show analysis results
$analysisTests = $global:testResults | Where-Object { $_.Name -like "*Analysis*" -and $_.Success -and -not ($_.Name -like "*Debug*") }
if ($analysisTests.Count -gt 0) {
    Write-Host "`n📊 ANALYSIS RESULTS:" -ForegroundColor Cyan
    Write-Host "====================" -ForegroundColor Cyan
    
    foreach ($test in $analysisTests) {
        $result = $test.ResponseData
        Write-Host "`n🔍 $($test.Name):" -ForegroundColor Yellow
        if ($result.network) { Write-Host "   Network: $($result.network)" -ForegroundColor White }
        if ($result.total_transactions -ne $null) { Write-Host "   Transactions: $($result.total_transactions)" -ForegroundColor White }
        if ($result.unique_tokens -ne $null) { Write-Host "   Unique Tokens: $($result.unique_tokens)" -ForegroundColor White }
        if ($result.total_eth_value -ne $null) { Write-Host "   Total ETH: $($result.total_eth_value)" -ForegroundColor White }
        if ($result.top_tokens -and $result.top_tokens.Count -gt 0) {
            Write-Host "   Top Token: $($result.top_tokens[0][0])" -ForegroundColor White
        }
    }
}

# Final recommendations
Write-Host "`n🎯 RECOMMENDATIONS:" -ForegroundColor Yellow
Write-Host "===================" -ForegroundColor Yellow

if ($successRate -ge 80) {
    Write-Host "✅ Function is working well overall!" -ForegroundColor Green
} else {
    Write-Host "❌ Function has significant issues that need attention" -ForegroundColor Red
}

# Check for specific issues
$hasDbIssues = $debugTests | Where-Object { $_.ResponseData.debug_info.database_debug.database_connected -eq $false }
$hasApiIssues = $debugTests | Where-Object { $_.ResponseData.debug_info.alchemy_debug.alchemy_api_responsive -eq $false }
$hasNoData = $analysisTests | Where-Object { $_.ResponseData.total_transactions -eq 0 }

if ($hasDbIssues) {
    Write-Host "🗄️ Database connection issues detected - check MongoDB URI and credentials" -ForegroundColor Red
}

if ($hasApiIssues) {
    Write-Host "🔗 Alchemy API issues detected - check API key and network access" -ForegroundColor Red
}

if ($hasNoData) {
    Write-Host "📊 Analysis returning 0 results - check wallet data and time ranges" -ForegroundColor Yellow
    Write-Host "   • Try longer time periods (7-14 days)" -ForegroundColor Gray
    Write-Host "   • Verify wallet data exists in database" -ForegroundColor Gray
    Write-Host "   • Check field names (address vs wallet_address, score vs rating)" -ForegroundColor Gray
}

Write-Host "`n🔧 NEXT STEPS:" -ForegroundColor Cyan
if ($hasDbIssues -or $hasApiIssues -or $hasNoData) {
    Write-Host "1. Review debug information above" -ForegroundColor White
    Write-Host "2. Check function logs: gcloud functions logs read $FunctionName --region=$Region --limit=50" -ForegroundColor White
    Write-Host "3. Verify environment variables in Google Cloud Console" -ForegroundColor White
    Write-Host "4. Test database connectivity separately" -ForegroundColor White
} else {
    Write-Host "1. Set up monitoring and alerting" -ForegroundColor White
    Write-Host "2. Configure Cloud Scheduler for periodic analysis" -ForegroundColor White
    Write-Host "3. Set up Telegram notifications" -ForegroundColor White
}

Write-Host "`n🎉 TESTING COMPLETE!" -ForegroundColor Green
Write-Host "Function URL: $FunctionUrl" -ForegroundColor Cyan
Write-Host "Overall Health: $(if ($successRate -ge 80) {'✅ HEALTHY'} else {'❌ NEEDS ATTENTION'})" -ForegroundColor $(if ($successRate -ge 80) {'Green'} else {'Red'})
Write-Host "============================================" -ForegroundColor Green