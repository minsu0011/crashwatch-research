@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u "03_이탈테스트_봉인인증.py"
) else (
  python -u "03_이탈테스트_봉인인증.py"
)
pause
