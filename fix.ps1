# Convert .env.production to proper YAML format
$envContent = Get-Content ".env.production" -Raw
$envLines = $envContent -split "`r?`n" | Where-Object { $_ -match "^[^#]" -and $_ -match "=" -and $_.Trim() }

$yamlContent = ""
foreach ($line in $envLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        $yamlContent += "$key`: `"$value`"`n"
    }
}

# Write with UTF8 no BOM
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText("env.yaml", $yamlContent, $utf8NoBom)

Write-Host "✅ Created env.yaml with proper encoding"