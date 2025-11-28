@echo off
echo Stopping StockAI...

REM Stop Node.js processes
taskkill /F /IM node.exe /T 2>nul
if %errorlevel% equ 0 echo Backend stopped

REM Stop npm processes
taskkill /F /IM npm.cmd /T 2>nul
if %errorlevel% equ 0 echo Frontend stopped

echo All services stopped
