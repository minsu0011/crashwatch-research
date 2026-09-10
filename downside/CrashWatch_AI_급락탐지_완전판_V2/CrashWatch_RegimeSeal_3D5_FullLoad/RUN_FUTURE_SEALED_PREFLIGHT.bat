@echo off
setlocal
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" run_regime_gate_future_sealed_once.py
exit /b %errorlevel%

