@echo off
setlocal
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" "run_long_horizon_generalization.py" --smoke
endlocal
