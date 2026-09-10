@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title CrashWatch 전체공정 원터치

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3.11"
) else (
  set "PY=python"
)
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto fail
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist ".env" copy /Y ".env.example" ".env" >nul
"%PYTHON%" -u 전체공정_실행.py
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo [실패] crashwatch_ai_data\logs를 확인하십시오.
pause
exit /b 1
