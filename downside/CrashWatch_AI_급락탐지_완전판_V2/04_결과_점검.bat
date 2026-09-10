@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u "04_결과_점검.py"
) else (
  python -u "04_결과_점검.py"
)
pause
