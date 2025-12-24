@echo off
REM Quick launcher for HolonicTrader Dashboard
REM Double-click this file to start the dashboard

echo ================================================
echo   HOLONIC TRADER - DASHBOARD LAUNCHER
echo ================================================
echo.
echo Starting dashboard...
echo.

REM Check if executable exists
if exist "dist\HolonicTrader\HolonicTrader.exe" (
    echo Running from executable...
    start "" "dist\HolonicTrader\HolonicTrader.exe"
) else (
    echo Executable not found. Running from Python...
    call .venv\Scripts\activate.bat
    python dashboard_gui.py
)
