@echo off
echo ========================================
echo StockAI - Starting Application
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install Python dependencies if needed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    pip install -q -r requirements.txt
)

REM Install backend dependencies
if not exist "backend-node\node_modules\" (
    echo Installing backend dependencies...
    cd backend-node
    call npm install --silent
    cd ..
)

REM Install frontend dependencies
if not exist "frontend\node_modules\" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install --silent
    cd ..
)

REM Start backend
echo Starting backend on http://localhost:5001...
cd backend-node
start /B node server.js > nul 2>&1
cd ..

REM Wait for backend
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting frontend on http://localhost:3000...
cd frontend
start /B npm start > nul 2>&1
cd ..

timeout /t 3 /nobreak > nul

echo.
echo ========================================
echo StockAI is running!
echo ========================================
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5001
echo.
echo Press Ctrl+C to stop
echo.

pause
