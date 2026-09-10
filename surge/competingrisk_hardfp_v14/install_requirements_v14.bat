@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements_v14.txt
if errorlevel 1 exit /b 1
echo V14 requirements installed.
endlocal
