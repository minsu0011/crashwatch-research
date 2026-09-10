@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: PREDICT_LATEST_WINDOWS.bat INPUT.parquet [OUTPUT.csv]
  exit /b 2
)
set "CW_INPUT=%~1"
set "CW_OUTPUT=%~2"
if "%CW_OUTPUT%"=="" set "CW_OUTPUT=%~dpn1_crashwatch_predictions.csv"
.venv\Scripts\python.exe crashwatch_predict.py --input "%CW_INPUT%" --output "%CW_OUTPUT%" --latest-only
if errorlevel 1 exit /b 1
echo Saved: %CW_OUTPUT%
pause
