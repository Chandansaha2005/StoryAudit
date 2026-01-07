@echo off
REM StoryAudit launcher for Windows

echo.
echo ========================================
echo   STORYAUDIT - Story Consistency Checker
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM Load .env file
if exist .env (
    for /f "tokens=*" %%a in ('type .env') do (
        if not "%%a"=="" (
            set %%a
        )
    )
)

REM Show menu
echo Choose an option:
echo.
echo  1) Interactive Menu (Recommended)
echo  2) Analyze Story 1
echo  3) Analyze Story 2
echo  4) Analyze Story 3
echo  5) Analyze Story 4
echo  6) Analyze All Stories
echo  7) Set API Key
echo  8) View Results
echo  9) Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" (
    python menu.py
) else if "%choice%"=="2" (
    if not defined GOOGLE_API_KEY (
        echo.
        echo ERROR: Google API Key not set!
        echo Run option 7 first to set your API key
        echo.
        pause
    ) else (
        python run.py --story-id 1 --verbose
        pause
    )
) else if "%choice%"=="3" (
    if not defined GOOGLE_API_KEY (
        echo.
        echo ERROR: Google API Key not set!
        echo Run option 7 first to set your API key
        echo.
        pause
    ) else (
        python run.py --story-id 2 --verbose
        pause
    )
) else if "%choice%"=="4" (
    if not defined GOOGLE_API_KEY (
        echo.
        echo ERROR: Google API Key not set!
        echo Run option 7 first to set your API key
        echo.
        pause
    ) else (
        python run.py --story-id 3 --verbose
        pause
    )
) else if "%choice%"=="5" (
    if not defined GOOGLE_API_KEY (
        echo.
        echo ERROR: Google API Key not set!
        echo Run option 7 first to set your API key
        echo.
        pause
    ) else (
        python run.py --story-id 4 --verbose
        pause
    )
) else if "%choice%"=="6" (
    if not defined GOOGLE_API_KEY (
        echo.
        echo ERROR: Google API Key not set!
        echo Run option 7 first to set your API key
        echo.
        pause
    ) else (
        python run.py --all --verbose
        pause
    )
) else if "%choice%"=="7" (
    echo.
    echo Get a free API key from: https://ai.google.dev/
    echo.
    setlocal enabledelayedexpansion
    set /p api_key="Enter your Google API Key: "
    (
        echo GOOGLE_API_KEY="!api_key!"
    ) > .env
    echo.
    echo API Key saved to .env file
    echo.
    pause
) else if "%choice%"=="8" (
    if exist results.csv (
        echo.
        echo === ANALYSIS RESULTS ===
        echo.
        type results.csv
        echo.
    ) else (
        echo No results file found. Analyze a story first.
    )
    echo.
    pause
) else if "%choice%"=="9" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice!
    pause
    goto :eof
)

goto :eof
