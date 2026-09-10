@echo off
chcp 65001 > nul
cd /d "%~dp0"
if exist ".venv_data\Scripts\python.exe" (
  ".venv_data\Scripts\python.exe" 04G_집중Nested_안전중지요청.py
) else (
  python 04G_집중Nested_안전중지요청.py
)
pause
