# Fixed Test Analysis Function and Monitor BigQuery Results
param(
    [Parameter(Mandatory=$true)]
    [string]$FunctionUrl,
    
    [string]$Network = "base",
    [string]$AnalysisType = "buy",
    [int]$NumWallets = 174,
    [float]$DaysBack = 1.0
)

Write-Host "🧪 TESTING CRYPTO ANALYSIS WITH BIGQUERY STORAGE" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Clean and validate URL
$CleanUrl = $FunctionUrl.Trim()
Write-Host "Function URL: $CleanUrl" -ForegroundColor White
Write-Host "Network: $Network" -ForegroundColor White
Write-Host "Analysis Type: $AnalysisType" -ForegroundColor White
Write-Host "Wallets: $NumWallets" -ForegroundColor White
Write-Host "Days Back: $DaysBack" -ForegroundColor White
Write-Host "=================================================" -ForegroundColor Cyan

# Test 1: Check function health and BigQuery config
Write-Host "`n1️⃣ Testing function health and BigQuery configuration..." -ForegroundColor Yellow

try {
    # Use the cleaned URL with explicit parameters
    $debugUrl = $CleanUrl + "?debug=true"
    $healthResponse = Invoke-RestMethod -Uri $debugUrl -Method GET -TimeoutSec 30 -ErrorAction Stop
    
    if ($healthResponse) {
        Write-Host "✅ Function is healthy" -ForegroundColor Green
        
        if ($healthResponse.debug_info) {
            Write-Host "📊 MongoDB configured: $($healthResponse.debug_info.mongo_configured)" -ForegroundColor White
            Write-Host "🔧 Alchemy configured: $($healthResponse.debug_info.alchemy_configured)" -ForegroundColor White
            Write-Host "📱 Telegram configured: $($healthResponse.telegram_configured)" -ForegroundColor White
            
            # Check BigQuery configuration in debug info
            if ($healthResponse.debug_info.bigquery_project_id) {
                Write-Host "🗄️ BigQuery Project: $($healthResponse.debug_info.bigquery_project_id)" -ForegroundColor Green
            }
            if ($healthResponse.debug_info.bigquery_dataset_id) {
                Write-Host "📊 BigQuery Dataset: $($healthResponse.debug_info.bigquery_dataset_id)" -ForegroundColor Green
            }
        } else {
            Write-Host "⚠️ Debug info not available, but function is responding" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "🔍 Trying basic health check without debug..." -ForegroundColor Yellow
    
    # Fallback to basic health check
    try {
        $basicHealth = Invoke-RestMethod -Uri $CleanUrl -Method GET -TimeoutSec 30
        Write-Host "✅ Basic health check successful" -ForegroundColor Green
        Write-Host "📊 Status: $($basicHealth.status)" -ForegroundColor White
        Write-Host "📱 Telegram: $($basicHealth.telegram_configured)" -ForegroundColor White
    } catch {
        Write-Host "❌ All health checks failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "🔍 Please verify the function URL and deployment" -ForegroundColor Yellow
        exit 1
    }
}

# Test 2: Run analysis and monitor BigQuery storage
Write-Host "`n2️⃣ Running analysis to test BigQuery storage..." -ForegroundColor Yellow

$analysisBody = @{
    network = $Network
    analysis_type = $AnalysisType
    num_wallets = $NumWallets
    days_back = $DaysBack
    notifications = $false
    debug = $false
} | ConvertTo-Json -Depth 3

Write-Host "📤 Sending analysis request..." -ForegroundColor Gray
Write-Host "Request: $Network $AnalysisType analysis with $NumWallets wallets" -ForegroundColor Gray

try {
    $analysisResponse = Invoke-RestMethod -Uri $CleanUrl -Method POST -Body $analysisBody -ContentType "application/json" -TimeoutSec 180 -ErrorAction Stop
    
    if ($analysisResponse.success) {
        Write-Host "✅ Analysis completed successfully!" -ForegroundColor Green
        Write-Host "📊 Network: $($analysisResponse.network)" -ForegroundColor White
        Write-Host "🔍 Analysis Type: $($analysisResponse.analysis_type)" -ForegroundColor White
        Write-Host "📝 Total Transactions: $($analysisResponse.total_transactions)" -ForegroundColor White
        Write-Host "🪙 Unique Tokens: $($analysisResponse.unique_tokens)" -ForegroundColor White
        Write-Host "💰 Total ETH Value: $($analysisResponse.total_eth_value)" -ForegroundColor White
        
        # Check if transfers were stored to BigQuery
        $transfersStored = 0
        
        # Check multiple possible locations for transfer storage count
        if ($analysisResponse.performance_metrics -and $analysisResponse.performance_metrics.transfers_stored) {
            $transfersStored = $analysisResponse.performance_metrics.transfers_stored
        } elseif ($analysisResponse.debug_info -and $analysisResponse.debug_info.transfers_stored) {
            $transfersStored = $analysisResponse.debug_info.transfers_stored
        } elseif ($analysisResponse.transfers_stored) {
            $transfersStored = $analysisResponse.transfers_stored
        }
        
        if ($transfersStored -gt 0) {
            Write-Host "✅ Transfers stored to BigQuery: $transfersStored" -ForegroundColor Green
        } else {
            Write-Host "⚠️ No transfer storage count found in response" -ForegroundColor Yellow
            Write-Host "   This might be normal if no qualifying transfers were found" -ForegroundColor Gray
        }
        
        # Show analysis timing
        if ($analysisResponse.performance_metrics -and $analysisResponse.performance_metrics.analysis_time) {
            Write-Host "⏱️ Analysis Time: $($analysisResponse.performance_metrics.analysis_time) seconds" -ForegroundColor White
        }
        
        # Show top tokens if any were found
        if ($analysisResponse.top_tokens -and $analysisResponse.top_tokens.Count -gt 0) {
            Write-Host "🏆 Top tokens found: $($analysisResponse.top_tokens.Count)" -ForegroundColor Green
            for ($i = 0; $i -lt [Math]::Min(3, $analysisResponse.top_tokens.Count); $i++) {
                $token = $analysisResponse.top_tokens[$i]
                if ($token.Count -ge 3) {
                    Write-Host "   $($i+1). $($token[0]) - Score: $($token[2])" -ForegroundColor Gray
                }
            }
        }
        
    } else {
        Write-Host "❌ Analysis failed: $($analysisResponse.error)" -ForegroundColor Red
        if ($analysisResponse.debug_info -and $analysisResponse.debug_info.analysis_traceback) {
            Write-Host "🔍 Error details:" -ForegroundColor Yellow
            $traceback = $analysisResponse.debug_info.analysis_traceback
            if ($traceback.Length -gt 500) {
                Write-Host $traceback.Substring(0, 500) + "..." -ForegroundColor Gray
            } else {
                Write-Host $traceback -ForegroundColor Gray
            }
        }
    }
    
} catch {
    Write-Host "❌ Analysis request failed: $($_.Exception.Message)" -ForegroundColor Red
    
    # Try to get more error details from the response
    if ($_.Exception.Response) {
        try {
            $errorStream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($errorStream)
            $errorBody = $reader.ReadToEnd()
            if ($errorBody) {
                Write-Host "🔍 Error response:" -ForegroundColor Yellow
                # Limit error output to avoid spam
                if ($errorBody.Length -gt 1000) {
                    Write-Host $errorBody.Substring(0, 1000) + "..." -ForegroundColor Gray
                } else {
                    Write-Host $errorBody -ForegroundColor Gray
                }
            }
        } catch {
            Write-Host "Could not read detailed error response" -ForegroundColor Gray
        }
    }
}

# Test 3: Run a second quick analysis to test data accumulation
Write-Host "`n3️⃣ Running second analysis to test data accumulation..." -ForegroundColor Yellow

$secondNetwork = if ($Network -eq "ethereum") { "base" } else { "ethereum" }
$secondType = if ($AnalysisType -eq "buy") { "sell" } else { "buy" }

$secondAnalysisBody = @{
    network = $secondNetwork
    analysis_type = $secondType
    num_wallets = 10
    days_back = 0.5
    notifications = $false
} | ConvertTo-Json -Depth 3

Write-Host "📤 Testing $secondNetwork $secondType analysis..." -ForegroundColor Gray

try {
    $secondResponse = Invoke-RestMethod -Uri $CleanUrl -Method POST -Body $secondAnalysisBody -ContentType "application/json" -TimeoutSec 120
    
    if ($secondResponse.success) {
        Write-Host "✅ Second analysis completed!" -ForegroundColor Green
        Write-Host "📊 Network: $($secondResponse.network)" -ForegroundColor White
        Write-Host "🔍 Type: $($secondResponse.analysis_type)" -ForegroundColor White
        Write-Host "📝 Transactions: $($secondResponse.total_transactions)" -ForegroundColor White
        
        # Check for additional transfers stored
        $additionalTransfers = 0
        if ($secondResponse.performance_metrics -and $secondResponse.performance_metrics.transfers_stored) {
            $additionalTransfers = $secondResponse.performance_metrics.transfers_stored
            Write-Host "💾 Additional transfers stored: $additionalTransfers" -ForegroundColor Green
        }
        
    } else {
        Write-Host "⚠️ Second analysis failed: $($secondResponse.error)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "⚠️ Second analysis failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Instructions for viewing results in BigQuery
Write-Host "`n🔍 VIEWING RESULTS IN BIGQUERY WEB CONSOLE:" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "1. Open BigQuery Console:" -ForegroundColor White
Write-Host "   https://console.cloud.google.com/bigquery?project=wtwalletranker" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Navigate to your data:" -ForegroundColor White
Write-Host "   └── wtwalletranker (project)" -ForegroundColor Gray
Write-Host "       └── crypto_analysis (dataset)" -ForegroundColor Gray
Write-Host "           └── transfers (table)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Click on 'transfers' table and select 'PREVIEW' to see recent data" -ForegroundColor White
Write-Host ""
Write-Host "4. Or copy/paste these SQL queries in the Query Editor:" -ForegroundColor White
Write-Host ""

# Use here-string to avoid backtick parsing issues
$query1 = @'
-- View recent transfers from your test
SELECT 
  wallet_address,
  token_symbol,
  transfer_type,
  cost_in_eth,
  network,
  timestamp,
  created_at
FROM `wtwalletranker.crypto_analysis.transfers` 
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY created_at DESC 
LIMIT 50;
'@

$query2 = @'
-- Count transfers by network and type
SELECT 
  network, 
  transfer_type, 
  COUNT(*) as transfer_count,
  SUM(cost_in_eth) as total_eth,
  COUNT(DISTINCT wallet_address) as unique_wallets
FROM `wtwalletranker.crypto_analysis.transfers` 
GROUP BY network, transfer_type
ORDER BY transfer_count DESC;
'@

Write-Host $query1 -ForegroundColor Cyan
Write-Host ""
Write-Host $query2 -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 EXPECTED DATA STRUCTURE:" -ForegroundColor Yellow
Write-Host "===========================" -ForegroundColor Yellow
Write-Host "• wallet_address: Ethereum/Base wallet addresses" -ForegroundColor White
Write-Host "• token_address: ERC20 token contract addresses" -ForegroundColor White
Write-Host "• transfer_type: 'buy' or 'sell'" -ForegroundColor White
Write-Host "• timestamp: When the blockchain transfer occurred" -ForegroundColor White
Write-Host "• cost_in_eth: ETH value of the transfer" -ForegroundColor White
Write-Host "• token_amount: Amount of tokens transferred" -ForegroundColor White
Write-Host "• network: 'ethereum' or 'base'" -ForegroundColor White
Write-Host "• created_at: When the record was inserted into BigQuery" -ForegroundColor White
Write-Host ""

Write-Host "🎯 TROUBLESHOOTING TIPS:" -ForegroundColor Red
Write-Host "========================" -ForegroundColor Red
Write-Host "If you don't see data in BigQuery:" -ForegroundColor White
Write-Host "1. Check that transfers_stored > 0 in the analysis results above" -ForegroundColor Gray
Write-Host "2. Verify the wtwalletranker project has the crypto_analysis dataset" -ForegroundColor Gray
Write-Host "3. Check Cloud Function logs for BigQuery-related errors:" -ForegroundColor Gray
Write-Host "   gcloud functions logs tail crypto-analysis-function --region=asia-southeast1" -ForegroundColor Cyan
Write-Host "4. Ensure your service account has BigQuery permissions in wtwalletranker" -ForegroundColor Gray
Write-Host ""

Write-Host "✅ TESTING COMPLETE!" -ForegroundColor Green
Write-Host "🔍 Go check BigQuery now: https://console.cloud.google.com/bigquery?project=wtwalletranker" -ForegroundColor Cyan