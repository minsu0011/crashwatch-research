@echo off
setlocal
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" prepare_regime_future_sealed.py %*
exit /b %errorlevel%

