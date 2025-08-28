# Comprehensive Telegram Notifications Test Script
param(
    [string]$MainFunctionUrl = "",
    [string]$ProjectId = "",
    [string]$Region = "asia-southeast1"
)

Write-Host "📱 COMPREHENSIVE TELEGRAM NOTIFICATIONS TEST SUITE" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Get function URLs if not provided
if ([string]::IsNullOrEmpty($MainFunctionUrl) -and ![string]::IsNullOrEmpty($ProjectId)) {
    Write-Host "🔍 Getting function URLs..." -ForegroundColor Yellow
    try {
        $MainFunctionUrl = gcloud functions describe "crypto-analysis-function" --region=$Region --format="value(serviceConfig.uri)" 2>$null
        if ([string]::IsNullOrEmpty($MainFunctionUrl)) {
            Write-Host "❌ Could not get main function URL. Please provide it manually." -ForegroundColor Red
            Write-Host "Usage: .\test-telegram.ps1 -MainFunctionUrl 'YOUR_FUNCTION_URL'" -ForegroundColor Yellow
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to get function URL. Please provide it manually." -ForegroundColor Red
        exit 1
    }
}

if ([string]::IsNullOrEmpty($MainFunctionUrl)) {
    $MainFunctionUrl = Read-Host "Please enter your main function URL"
}

Write-Host "Main Function URL: $MainFunctionUrl" -ForegroundColor White
Write-Host "Testing Telegram integration..." -ForegroundColor White
Write-Host "=================================================" -ForegroundColor Cyan

$global:testResults = @()
$global:testNumber = 1

# Helper function to run test
function Run-TelegramTest {
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
    
    # Try to get specific function URL for Telegram endpoints
    if ($Endpoint.StartsWith("telegram-")) {
        $functionName = $Endpoint
        $url = ""
        
        if (![string]::IsNullOrEmpty($ProjectId)) {
            try {
                $url = gcloud functions describe $functionName --region=$Region --format="value(serviceConfig.uri)" 2>$null
            } catch {
                # Fall back to main URL
                $url = $MainFunctionUrl
            }
        }
        
        if ([string]::IsNullOrEmpty($url)) {
            $url = $MainFunctionUrl
        }
    } else {
        $url = $MainFunctionUrl + $Endpoint
    }
    
    $success = $false
    $responseData = $null
    $statusCode = 0
    $errorMessage = ""
    
    try {
        Write-Host "   🔄 Testing: $Method $url" -ForegroundColor Gray
        
        if ($Method -eq "GET") {
            $response = Invoke-WebRequest -Uri $url -Method GET -UseBasicParsing -TimeoutSec 30
        } else {
            $jsonBody = if ($Body) { $Body | ConvertTo-Json -Depth 10 } else { "{}" }
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
                
                # Show key response fields for Telegram tests
                if ($responseData.configured -ne $null) {
                    Write-Host "   📱 Telegram Configured: $($responseData.configured)" -ForegroundColor $(if ($responseData.configured) {'Green'} else {'Red'})
                }
                if ($responseData.success -ne $null) {
                    Write-Host "   🎯 Success: $($responseData.success)" -ForegroundColor $(if ($responseData.success) {'Green'} else {'Red'})
                }
                if ($responseData.message) {
                    Write-Host "   💬 Message: $($responseData.message)" -ForegroundColor White
                }
                if ($responseData.bot_token_set -ne $null) {
                    Write-Host "   🤖 Bot Token Set: $($responseData.bot_token_set)" -ForegroundColor $(if ($responseData.bot_token_set) {'Green'} else {'Red'})
                }
                if ($responseData.chat_id_set -ne $null) {
                    Write-Host "   💬 Chat ID Set: $($responseData.chat_id_set)" -ForegroundColor $(if ($responseData.chat_id_set) {'Green'} else {'Red'})
                }
                if ($responseData.notifications_sent -ne $null) {
                    Write-Host "   📨 Notifications Sent: $($responseData.notifications_sent)" -ForegroundColor White
                }
                if ($responseData.telegram_configured -ne $null) {
                    Write-Host "   ⚙️ Telegram Integration: $($responseData.telegram_configured)" -ForegroundColor $(if ($responseData.telegram_configured) {'Green'} else {'Yellow'})
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
    Start-Sleep -Seconds 2
}

# Start Telegram testing
Write-Host "`n🚀 STARTING TELEGRAM TESTS..." -ForegroundColor Green

# Test 1: Main Function Telegram Status
Run-TelegramTest -TestName "Main Function Telegram Status" -Description "Check if main function reports Telegram configuration"

# Test 2: Main Function Debug (Telegram Info)
Run-TelegramTest -TestName "Main Function Debug Mode" -Endpoint "?debug=true" -Description "Get detailed Telegram configuration info"

# Test 3: Telegram Status Function
Run-TelegramTest -TestName "Telegram Status Endpoint" -Endpoint "telegram-status" -Description "Dedicated Telegram status check"

# Test 4: Telegram Test Notification
Run-TelegramTest -TestName "Send Test Notification" -Method "POST" -Endpoint "telegram-test" -Description "Send a test message to Telegram"

# Test 5: Get Alert Thresholds
Run-TelegramTest -TestName "Get Alert Thresholds" -Endpoint "telegram-thresholds" -Description "Retrieve current alert thresholds"

# Test 6: Update Alert Thresholds
Run-TelegramTest -TestName "Update Alert Thresholds" -Method "POST" -Endpoint "telegram-thresholds" -Body @{
    min_alpha_score = 30.0
    min_sell_score = 25.0
    min_eth_value = 0.1
    min_wallet_count = 5
} -Description "Update alert thresholds"

# Test 7: Simulate Buy Alert
Run-TelegramTest -TestName "Simulate Buy Alert" -Method "POST" -Endpoint "telegram-simulate" -Body @{
    alert_type = "buy"
} -Description "Simulate a buy alert notification"

# Test 8: Simulate Sell Alert
Run-TelegramTest -TestName "Simulate Sell Alert" -Method "POST" -Endpoint "telegram-simulate" -Body @{
    alert_type = "sell"
} -Description "Simulate a sell alert notification"

# Test 9: Analysis with Notifications Enabled
Run-TelegramTest -TestName "Analysis with Telegram Notifications" -Method "POST" -Body @{
    network = "ethereum"
    analysis_type = "buy"
    num_wallets = 50
    days_back = 1.0
    notifications = $true
} -Description "Run analysis with Telegram notifications enabled"

# Test 10: Analysis with Notifications Disabled
Run-TelegramTest -TestName "Analysis without Notifications" -Method "POST" -Body @{
    network = "base"
    analysis_type = "sell"
    num_wallets = 30
    days_back = 2.0
    notifications = $false
} -Description "Run analysis without sending notifications"

# Test 11: Error Scenario (should trigger error notification)
Run-TelegramTest -TestName "Error Scenario" -Method "POST" -Body @{
    network = "invalid_network"
    analysis_type = "buy"
    notifications = $true
} -Description "Trigger error to test error notifications" -ExpectedStatus 500

# Generate Test Report
Write-Host "`n📊 TELEGRAM TEST RESULTS SUMMARY" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

$passedTests = ($global:testResults | Where-Object { $_.Success }).Count
$totalTests = $global:testResults.Count
$successRate = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 1) } else { 0 }

Write-Host "✅ Passed: $passedTests" -ForegroundColor Green
Write-Host "❌ Failed: $($totalTests - $passedTests)" -ForegroundColor Red
Write-Host "📈 Success Rate: $successRate%" -ForegroundColor White
Write-Host "🕒 Total Tests: $totalTests" -ForegroundColor White

# Analyze Telegram configuration
$telegramConfigured = $false
$telegramWorking = $false

$statusTests = $global:testResults | Where-Object { $_.Name -like "*Status*" -and $_.Success }
foreach ($test in $statusTests) {
    if ($test.ResponseData.configured -eq $true -or $test.ResponseData.telegram_configured -eq $true) {
        $telegramConfigured = $true
    }
}

$notificationTests = $global:testResults | Where-Object { $_.Name -like "*Test Notification*" -and $_.Success }
foreach ($test in $notificationTests) {
    if ($test.ResponseData.success -eq $true) {
        $telegramWorking = $true
    }
}

Write-Host "`n📱 TELEGRAM INTEGRATION ANALYSIS:" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Configuration Status: $(if ($telegramConfigured) {'✅ CONFIGURED'} else {'❌ NOT CONFIGURED'})" -ForegroundColor $(if ($telegramConfigured) {'Green'} else {'Red'})
Write-Host "Notification Status: $(if ($telegramWorking) {'✅ WORKING'} else {'❌ NOT WORKING'})" -ForegroundColor $(if ($telegramWorking) {'Green'} else {'Red'})

# Show failed tests details
$failedTests = $global:testResults | Where-Object { -not $_.Success }
if ($failedTests.Count -gt 0) {
    Write-Host "`n❌ FAILED TESTS:" -ForegroundColor Red
    Write-Host "=================" -ForegroundColor Red
    
    foreach ($test in $failedTests) {
        Write-Host "• $($test.Name) (HTTP $($test.StatusCode)): $($test.ErrorMessage)" -ForegroundColor Red
    }
}

# Show successful notification tests
$successfulNotificationTests = $global:testResults | Where-Object { 
    ($_.Name -like "*Notification*" -or $_.Name -like "*Alert*" -or $_.Name -like "*Simulate*") -and 
    $_.Success -and 
    $_.ResponseData.success -eq $true 
}

if ($successfulNotificationTests.Count -gt 0) {
    Write-Host "`n✅ SUCCESSFUL TELEGRAM NOTIFICATIONS:" -ForegroundColor Green
    Write-Host "====================================" -ForegroundColor Green
    
    foreach ($test in $successfulNotificationTests) {
        Write-Host "• $($test.Name): $($test.ResponseData.message)" -ForegroundColor Green
    }
    
    Write-Host "`n📱 Check your Telegram app for the test messages!" -ForegroundColor Cyan
}

# Configuration analysis
$configTests = $global:testResults | Where-Object { $_.Name -like "*Status*" -and $_.Success -and $_.ResponseData }
if ($configTests.Count -gt 0) {
    Write-Host "`n⚙️ TELEGRAM CONFIGURATION DETAILS:" -ForegroundColor Yellow
    Write-Host "===================================" -ForegroundColor Yellow
    
    foreach ($test in $configTests) {
        $data = $test.ResponseData
        
        if ($data.bot_token_set -ne $null) {
            Write-Host "🤖 Bot Token: $(if ($data.bot_token_set) {'✅ Set'} else {'❌ Missing'})" -ForegroundColor $(if ($data.bot_token_set) {'Green'} else {'Red'})
        }
        if ($data.chat_id_set -ne $null) {
            Write-Host "💬 Chat ID: $(if ($data.chat_id_set) {'✅ Set'} else {'❌ Missing'})" -ForegroundColor $(if ($data.chat_id_set) {'Green'} else {'Red'})
        }
        if ($data.alert_thresholds) {
            Write-Host "🎯 Alert Thresholds:" -ForegroundColor White
            $data.alert_thresholds.PSObject.Properties | ForEach-Object {
                Write-Host "   • $($_.Name): $($_.Value)" -ForegroundColor Gray
            }
        }
        break  # Only show first successful config test
    }
}

# Analysis integration tests
$analysisTests = $global:testResults | Where-Object { $_.Name -like "*Analysis*" -and $_.Success }
if ($analysisTests.Count -gt 0) {
    Write-Host "`n📊 ANALYSIS WITH TELEGRAM INTEGRATION:" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    
    foreach ($test in $analysisTests) {
        $data = $test.ResponseData
        Write-Host "🔍 $($test.Name):" -ForegroundColor Yellow
        
        if ($data.network) { Write-Host "   Network: $($data.network)" -ForegroundColor White }
        if ($data.analysis_type) { Write-Host "   Type: $($data.analysis_type)" -ForegroundColor White }
        if ($data.total_transactions -ne $null) { Write-Host "   Transactions: $($data.total_transactions)" -ForegroundColor White }
        if ($data.notifications_sent -ne $null) { 
            Write-Host "   Notifications Sent: $($data.notifications_sent)" -ForegroundColor $(if ($data.notifications_sent -gt 0) {'Green'} else {'Yellow'})
        }
        if ($data.debug_info.notifications_configured -ne $null) {
            Write-Host "   Telegram Ready: $($data.debug_info.notifications_configured)" -ForegroundColor $(if ($data.debug_info.notifications_configured) {'Green'} else {'Red'})
        }
        Write-Host ""
    }
}

Write-Host "`n🎯 RECOMMENDATIONS:" -ForegroundColor Yellow
Write-Host "===================" -ForegroundColor Yellow

if ($telegramConfigured -and $telegramWorking) {
    Write-Host "✅ Telegram notifications are fully functional!" -ForegroundColor Green
    Write-Host "🚀 Your crypto analysis system is ready for production" -ForegroundColor Green
    Write-Host "📱 You'll receive notifications for high-value token discoveries" -ForegroundColor Cyan
    
    Write-Host "`n🔄 Next steps:" -ForegroundColor White
    Write-Host "1. Set up Cloud Scheduler for automated analysis" -ForegroundColor Gray
    Write-Host "2. Fine-tune alert thresholds based on your preferences" -ForegroundColor Gray
    Write-Host "3. Monitor your Telegram chat for real-time alerts" -ForegroundColor Gray
    
} elseif ($telegramConfigured -and !$telegramWorking) {
    Write-Host "⚠️ Telegram is configured but notifications aren't working" -ForegroundColor Yellow
    Write-Host "🔧 Troubleshooting steps:" -ForegroundColor White
    Write-Host "1. Verify your bot token is correct" -ForegroundColor Gray
    Write-Host "2. Check that your chat ID is accurate" -ForegroundColor Gray
    Write-Host "3. Ensure the bot can send messages to your chat" -ForegroundColor Gray
    Write-Host "4. Make sure the bot isn't blocked" -ForegroundColor Gray
    
} elseif (!$telegramConfigured) {
    Write-Host "❌ Telegram is not configured" -ForegroundColor Red
    Write-Host "📱 Setup required:" -ForegroundColor White
    Write-Host "1. Create a Telegram bot via @BotFather" -ForegroundColor Gray
    Write-Host "2. Get your chat ID from @userinfobot" -ForegroundColor Gray
    Write-Host "3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables" -ForegroundColor Gray
    Write-Host "4. Redeploy your Cloud Functions" -ForegroundColor Gray
} else {
    Write-Host "🔍 Mixed results - some features may be partially working" -ForegroundColor Yellow
    Write-Host "📋 Review the test results above for specific issues" -ForegroundColor White
}

# Show specific function URLs for manual testing
if (![string]::IsNullOrEmpty($ProjectId)) {
    Write-Host "`n🔗 FUNCTION URLs FOR MANUAL TESTING:" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    
    $functionNames = @(
        @{Name="crypto-analysis-function"; Description="Main analysis function"},
        @{Name="telegram-status"; Description="Telegram status check"},
        @{Name="telegram-test"; Description="Send test notification"},
        @{Name="telegram-thresholds"; Description="Manage alert thresholds"},
        @{Name="telegram-simulate"; Description="Simulate alerts"}
    )
    
    foreach ($func in $functionNames) {
        try {
            $url = gcloud functions describe $func.Name --region=$Region --format="value(serviceConfig.uri)" 2>$null
            if ($url) {
                Write-Host "🌐 $($func.Description): $url" -ForegroundColor White
            }
        } catch {
            # Function might not exist, skip
        }
    }
}

Write-Host "`n📱 TELEGRAM NOTIFICATION TYPES:" -ForegroundColor Yellow
Write-Host "===============================" -ForegroundColor Yellow
Write-Host "🆕 High Alpha Token Alerts - When promising tokens are discovered" -ForegroundColor Green
Write-Host "📉 Sell Pressure Warnings - When smart money is selling" -ForegroundColor Yellow
Write-Host "📊 Analysis Summaries - Overview of each analysis run" -ForegroundColor Blue
Write-Host "❌ Error Notifications - When analysis encounters problems" -ForegroundColor Red
Write-Host "🏥 System Status Updates - Health checks and diagnostics" -ForegroundColor Cyan
Write-Host "🧪 Test Messages - For verifying notification setup" -ForegroundColor Gray

Write-Host "`n⚙️ CUSTOMIZABLE ALERT THRESHOLDS:" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Yellow
Write-Host "• min_alpha_score: Minimum score for buy alerts (default: 25.0)" -ForegroundColor White
Write-Host "• min_sell_score: Minimum score for sell alerts (default: 20.0)" -ForegroundColor White
Write-Host "• min_eth_value: Minimum ETH value for alerts (default: 0.05)" -ForegroundColor White
Write-Host "• min_wallet_count: Minimum smart wallets required (default: 3)" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Adjust these via the telegram-thresholds endpoint" -ForegroundColor Cyan

Write-Host "`n🔧 EXAMPLE CURL COMMANDS:" -ForegroundColor Yellow
Write-Host "=========================" -ForegroundColor Yellow

if ($MainFunctionUrl) {
    Write-Host "# Send test notification" -ForegroundColor Gray
    Write-Host "curl -X POST '$MainFunctionUrl' \" -ForegroundColor Cyan
    Write-Host "  -H 'Content-Type: application/json' \" -ForegroundColor Cyan
    Write-Host "  -d '{""network"":""ethereum"",""analysis_type"":""buy"",""notifications"":true}'" -ForegroundColor Cyan
    
    Write-Host "`n# Check Telegram status" -ForegroundColor Gray
    Write-Host "curl '$MainFunctionUrl?debug=true'" -ForegroundColor Cyan
}

Write-Host "`n🎊 TELEGRAM TESTING COMPLETE!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

$overallStatus = if ($telegramConfigured -and $telegramWorking) {
    "🎉 FULLY FUNCTIONAL"
} elseif ($telegramConfigured) {
    "⚠️ NEEDS TROUBLESHOOTING" 
} else {
    "❌ REQUIRES SETUP"
}

Write-Host "Overall Telegram Status: $overallStatus" -ForegroundColor $(
    if ($telegramConfigured -and $telegramWorking) { 'Green' } 
    elseif ($telegramConfigured) { 'Yellow' } 
    else { 'Red' }
)
Write-Host "Success Rate: $successRate% ($passedTests/$totalTests tests passed)" -ForegroundColor White

if ($telegramWorking) {
    Write-Host "`n📱 Check your Telegram app now for test notifications!" -ForegroundColor Cyan
    Write-Host "🚀 Your crypto analysis system is ready for real-time alerts!" -ForegroundColor Green
} else {
    Write-Host "`n🔧 Follow the recommendations above to complete your setup" -ForegroundColor Yellow
}

Write-Host "=============================" -ForegroundColor Green