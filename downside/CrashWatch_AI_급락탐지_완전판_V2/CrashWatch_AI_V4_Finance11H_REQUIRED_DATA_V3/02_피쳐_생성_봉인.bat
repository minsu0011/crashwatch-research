@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u "02_피쳐_생성_봉인.py"
) else (
  python -u "02_피쳐_생성_봉인.py"
)
pause
