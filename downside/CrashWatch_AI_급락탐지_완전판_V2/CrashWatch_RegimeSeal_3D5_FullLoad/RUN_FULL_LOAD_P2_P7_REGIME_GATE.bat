@echo off
setlocal
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" run_regime_submodel_gate.py %*
exit /b %errorlevel%

