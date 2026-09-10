@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u "01_데이터_크롤링.py"
) else (
  python -u "01_데이터_크롤링.py"
)
pause
