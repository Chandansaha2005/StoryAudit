# StoryAudit launcher for PowerShell

Clear-Host

Write-Host ""
Write-Host "========================================"
Write-Host "  STORYAUDIT - Story Consistency Checker"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.10+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Load .env file if it exists
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"')
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "✓ API Key loaded from .env" -ForegroundColor Green
}

Write-Host ""

# Menu loop
$running = $true
while ($running) {
    Write-Host "Choose an option:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1) Interactive Menu (Recommended)"
    Write-Host "  2) Analyze Story 1"
    Write-Host "  3) Analyze Story 2"
    Write-Host "  4) Analyze Story 3"
    Write-Host "  5) Analyze Story 4"
    Write-Host "  6) Analyze All Stories"
    Write-Host "  7) Set API Key"
    Write-Host "  8) View Results"
    Write-Host "  9) Exit"
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-9)"
    
    switch ($choice) {
        "1" {
            python menu.py
            break
        }
        "2" {
            if (-not $env:GOOGLE_API_KEY) {
                Write-Host ""
                Write-Host "ERROR: Google API Key not set!" -ForegroundColor Red
                Write-Host "Run option 7 first to set your API key"
                Write-Host ""
                Read-Host "Press Enter to continue"
            } else {
                python run.py --story-id 1 --verbose
                Read-Host "Press Enter to continue"
            }
        }
        "3" {
            if (-not $env:GOOGLE_API_KEY) {
                Write-Host ""
                Write-Host "ERROR: Google API Key not set!" -ForegroundColor Red
                Write-Host "Run option 7 first to set your API key"
                Write-Host ""
                Read-Host "Press Enter to continue"
            } else {
                python run.py --story-id 2 --verbose
                Read-Host "Press Enter to continue"
            }
        }
        "4" {
            if (-not $env:GOOGLE_API_KEY) {
                Write-Host ""
                Write-Host "ERROR: Google API Key not set!" -ForegroundColor Red
                Write-Host "Run option 7 first to set your API key"
                Write-Host ""
                Read-Host "Press Enter to continue"
            } else {
                python run.py --story-id 3 --verbose
                Read-Host "Press Enter to continue"
            }
        }
        "5" {
            if (-not $env:GOOGLE_API_KEY) {
                Write-Host ""
                Write-Host "ERROR: Google API Key not set!" -ForegroundColor Red
                Write-Host "Run option 7 first to set your API key"
                Write-Host ""
                Read-Host "Press Enter to continue"
            } else {
                python run.py --story-id 4 --verbose
                Read-Host "Press Enter to continue"
            }
        }
        "6" {
            if (-not $env:GOOGLE_API_KEY) {
                Write-Host ""
                Write-Host "ERROR: Google API Key not set!" -ForegroundColor Red
                Write-Host "Run option 7 first to set your API key"
                Write-Host ""
                Read-Host "Press Enter to continue"
            } else {
                python run.py --all --verbose
                Read-Host "Press Enter to continue"
            }
        }
        "7" {
            Write-Host ""
            Write-Host "Get a free API key from: https://ai.google.dev/" -ForegroundColor Yellow
            Write-Host ""
            $apiKey = Read-Host "Enter your Google API Key"
            
            if ($apiKey) {
                $envContent = "GOOGLE_API_KEY=`"$apiKey`""
                Set-Content -Path ".env" -Value $envContent
                [System.Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", $apiKey, "Process")
                Write-Host ""
                Write-Host "✓ API Key saved to .env file" -ForegroundColor Green
                Write-Host ""
                Read-Host "Press Enter to continue"
            }
        }
        "8" {
            Write-Host ""
            if (Test-Path "results.csv") {
                Write-Host "=== ANALYSIS RESULTS ===" -ForegroundColor Cyan
                Write-Host ""
                Get-Content "results.csv"
                Write-Host ""
            } else {
                Write-Host "No results file found. Analyze a story first." -ForegroundColor Yellow
            }
            Read-Host "Press Enter to continue"
        }
        "9" {
            Write-Host "Goodbye!"
            $running = $false
        }
        default {
            Write-Host "Invalid choice!" -ForegroundColor Red
            Read-Host "Press Enter to continue"
        }
    }
    
    Clear-Host
}
