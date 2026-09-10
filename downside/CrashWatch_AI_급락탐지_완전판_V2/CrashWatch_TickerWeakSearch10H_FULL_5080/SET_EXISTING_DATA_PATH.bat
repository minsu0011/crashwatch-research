@echo off
setlocal
cd /d "%~dp0"
echo Enter the full path of the existing crashwatch_ai_data folder.
set /p DATA_PATH=Path: 
if "%DATA_PATH%"=="" (
  echo Empty path. Nothing changed.
  pause
  exit /b 1
)
> .env echo CRASHWATCH_DATA_DIR="%DATA_PATH%"
echo Saved to %CD%\.env
pause
