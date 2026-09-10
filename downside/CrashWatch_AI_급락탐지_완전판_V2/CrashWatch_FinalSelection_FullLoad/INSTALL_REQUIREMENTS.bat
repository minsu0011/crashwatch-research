@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo Installation completed.
pause
