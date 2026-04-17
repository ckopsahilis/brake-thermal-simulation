@echo off
setlocal

REM Run pure-Python simulation + MP4 generation using local venv interpreter
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py"
if errorlevel 1 (
    echo [ERROR] Python pipeline failed.
    exit /b 1
)

echo [OK] Python pipeline complete.
endlocal
