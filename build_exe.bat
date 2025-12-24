@echo off
REM Build script for HolonicTrader Dashboard Executable
REM This creates a standalone .exe that can be run without Python installed

echo ================================================
echo   HOLONIC TRADER - EXECUTABLE BUILD SCRIPT
echo ================================================
echo.

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install PyInstaller if not already installed
echo.
echo [2/4] Checking PyInstaller installation...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
) else (
    echo PyInstaller already installed.
)

REM Clean previous build
echo.
echo [3/4] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build executable
echo.
echo [4/4] Building executable...
echo This may take 5-10 minutes...
pyinstaller --clean dashboard.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   BUILD COMPLETE!
echo ================================================
echo.
echo Executable location: dist\HolonicTrader\HolonicTrader.exe
echo.
echo To run the dashboard:
echo   1. Navigate to: dist\HolonicTrader\
echo   2. Double-click: HolonicTrader.exe
echo.
echo NOTE: The executable includes all dependencies.
echo       You can copy the entire 'HolonicTrader' folder
echo       to another computer and it will work!
echo.
pause
