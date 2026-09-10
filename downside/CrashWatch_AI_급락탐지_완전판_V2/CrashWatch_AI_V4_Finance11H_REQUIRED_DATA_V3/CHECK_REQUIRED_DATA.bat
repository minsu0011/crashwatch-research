@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist .venv_data\Scripts\python.exe (
  .venv_data\Scripts\python.exe 06B_필수데이터_검증.py
) else (
  python 06B_필수데이터_검증.py
)
endlocal
pause
