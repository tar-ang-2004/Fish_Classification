@echo off
echo 🐟 Fish Classification Flask App
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if virtual environment exists
if exist "venv" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo 💡 No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ⬇️ Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo 🚀 Starting Fish Classification Web App...
echo.
echo 📝 Access the app at: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python run_app.py

echo.
echo 👋 Flask app stopped
pause
